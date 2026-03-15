from __future__ import annotations

import argparse
import contextlib
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import (
    PairedMelDataset,
    collect_paired_files,
    compute_shared_stats,
    denormalize_tensor,
    load_stats,
    save_stats,
    split_pairs,
)
from .losses import CompositeMelLoss
from .metrics import lsd_metric, mae_metric, ssim_metric
from .models import ProgressiveMelRefiner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a residual mel-spectrogram refiner.")
    parser.add_argument("--low-dir", type=Path, required=True, help="Directory with RVQ-VAE coarse mel .npy files.")
    parser.add_argument("--high-dir", type=Path, required=True, help="Directory with target high-detail mel .npy files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store checkpoints and stats.")
    parser.add_argument("--stats-path", type=Path, default=None, help="Optional path to an existing shared stats JSON.")
    parser.add_argument("--norm", choices=("none", "zscore"), default="zscore")
    parser.add_argument("--use-deltas", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--group-separator", type=str, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--num-subbands", type=int, default=4)
    parser.add_argument("--coarse-weight", type=float, default=0.2)
    parser.add_argument("--grad-weight", type=float, default=0.3)
    parser.add_argument("--hf-weight", type=float, default=0.2)
    parser.add_argument("--ssim-weight", type=float, default=0.1)
    parser.add_argument("--detail-freq-boost", type=float, default=1.5)
    parser.add_argument("--scheduler", choices=("cosine", "plateau", "none"), default="cosine")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--plateau-patience", type=int, default=8)
    parser.add_argument("--early-stopping-patience", type=int, default=20)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loader(
    dataset: PairedMelDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )


def resolve_stats(
    norm: str,
    stats_path: Path | None,
    output_dir: Path,
    train_pairs,
) -> tuple[dict[str, float] | None, Path | None]:
    if norm == "none":
        return None, None

    resolved_path = stats_path or (output_dir / "stats.json")
    if resolved_path.exists():
        stats = load_stats(resolved_path)
    else:
        stats = compute_shared_stats(train_pairs)
        save_stats(stats, resolved_path)
    return stats, resolved_path


def create_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer):
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        )
    if args.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=args.plateau_patience,
            min_lr=args.min_lr,
        )
    return None


def save_checkpoint(
    path: Path,
    model: ProgressiveMelRefiner,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    args: argparse.Namespace,
    stats: dict[str, float] | None,
    best_val_loss: float,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "args": vars(args),
        "stats": stats,
        "best_val_loss": best_val_loss,
        "model_config": {
            "in_channels": 3 if args.use_deltas else 1,
            "base_channels": args.base_channels,
            "num_subbands": args.num_subbands,
        },
    }
    torch.save(checkpoint, path)


def run_epoch(
    model: ProgressiveMelRefiner,
    loader: DataLoader,
    criterion: CompositeMelLoss,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    coarse_weight: float,
    stats: dict[str, float] | None,
    amp_enabled: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    totals = {
        "loss": 0.0,
        "l1": 0.0,
        "grad": 0.0,
        "hf": 0.0,
        "ssim_loss": 0.0,
        "coarse_l1": 0.0,
        "mae": 0.0,
        "ssim": 0.0,
        "lsd": 0.0,
    }
    count = 0

    autocast_context = (
        torch.cuda.amp.autocast if amp_enabled and device.type == "cuda" else contextlib.nullcontext
    )

    for batch in loader:
        model_input = batch["input"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        batch_size = model_input.size(0)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with autocast_context():
            outputs = model(model_input)
            prediction = outputs["prediction"]
            loss, components = criterion(prediction, target)
            coarse_l1 = F.l1_loss(outputs["coarse_prediction"], target)
            total_loss = loss + (coarse_weight * coarse_l1)

        if training:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

        prediction_eval = denormalize_tensor(prediction.detach(), stats)
        target_eval = denormalize_tensor(target.detach(), stats)

        totals["loss"] += float(total_loss.detach().item()) * batch_size
        totals["l1"] += float(components["l1"].item()) * batch_size
        totals["grad"] += float(components["grad"].item()) * batch_size
        totals["hf"] += float(components["hf"].item()) * batch_size
        totals["ssim_loss"] += float(components["ssim_loss"].item()) * batch_size
        totals["coarse_l1"] += float(coarse_l1.detach().item()) * batch_size
        totals["mae"] += float(mae_metric(prediction_eval, target_eval).item()) * batch_size
        totals["ssim"] += float(ssim_metric(prediction_eval, target_eval).item()) * batch_size
        totals["lsd"] += float(lsd_metric(prediction_eval, target_eval).item()) * batch_size
        count += batch_size

    return {name: value / max(count, 1) for name, value in totals.items()}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    set_seed(args.seed)
    device = choose_device(args.device)

    all_pairs = collect_paired_files(args.low_dir, args.high_dir)
    train_pairs, val_pairs = split_pairs(
        all_pairs,
        val_ratio=args.val_ratio,
        seed=args.seed,
        group_separator=args.group_separator,
    )
    stats, resolved_stats_path = resolve_stats(args.norm, args.stats_path, args.output_dir, train_pairs)

    train_dataset = PairedMelDataset(train_pairs, stats=stats, use_deltas=args.use_deltas)
    val_dataset = PairedMelDataset(val_pairs, stats=stats, use_deltas=args.use_deltas)

    pin_memory = device.type == "cuda"
    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    input_channels = 3 if args.use_deltas else 1
    model = ProgressiveMelRefiner(
        in_channels=input_channels,
        base_channels=args.base_channels,
        num_subbands=args.num_subbands,
    ).to(device)

    criterion = CompositeMelLoss(
        grad_weight=args.grad_weight,
        hf_weight=args.hf_weight,
        ssim_weight=args.ssim_weight,
        detail_freq_boost=args.detail_freq_boost,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = create_scheduler(args, optimizer)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    print(f"Device: {device}")
    print(f"Training pairs: {len(train_pairs)} | Validation pairs: {len(val_pairs)}")
    print(f"Input channels: {input_channels} | Shared stats: {resolved_stats_path or 'disabled'}")

    best_val_loss = math.inf
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            coarse_weight=args.coarse_weight,
            stats=stats,
            amp_enabled=args.amp,
        )

        with torch.no_grad():
            val_metrics = run_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                optimizer=None,
                scaler=None,
                device=device,
                coarse_weight=args.coarse_weight,
                stats=stats,
                amp_enabled=args.amp,
            )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_metrics['loss']:.5f} | "
            f"val_loss={val_metrics['loss']:.5f} | "
            f"val_mae={val_metrics['mae']:.5f} | "
            f"val_ssim={val_metrics['ssim']:.5f} | "
            f"val_lsd={val_metrics['lsd']:.5f} | "
            f"lr={current_lr:.2e}"
        )

        save_checkpoint(
            args.output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            args=args,
            stats=stats,
            best_val_loss=best_val_loss,
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                args.output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                args=args,
                stats=stats,
                best_val_loss=best_val_loss,
            )
        else:
            patience_counter += 1

        if patience_counter >= args.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best validation loss {best_val_loss:.5f} was reached at epoch {best_epoch}."
            )
            break

    print(f"Best checkpoint: {args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
