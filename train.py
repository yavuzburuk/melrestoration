from __future__ import annotations

import argparse
import contextlib
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

if __package__:
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
else:
    from data import (
        PairedMelDataset,
        collect_paired_files,
        compute_shared_stats,
        denormalize_tensor,
        load_stats,
        save_stats,
        split_pairs,
    )
    from losses import CompositeMelLoss
    from metrics import lsd_metric, mae_metric, ssim_metric
    from models import ProgressiveMelRefiner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a residual mel-spectrogram refiner.")
    parser.add_argument("--low-dir", type=Path, required=True, help="Directory with RVQ-VAE coarse mel .npy files.")
    parser.add_argument("--high-dir", type=Path, required=True, help="Directory with target high-detail mel .npy files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store checkpoints and stats.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume training from a saved checkpoint.")
    parser.add_argument("--stats-path", type=Path, default=None, help="Optional path to an existing shared stats JSON.")
    parser.add_argument("--norm", choices=("none", "zscore"), default="zscore")
    parser.add_argument("--use-deltas", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--pairing-mode",
        choices=("relative", "basename"),
        default="relative",
        help="Use 'relative' for matching subfolder paths or 'basename' for matching filenames only.",
    )
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


def create_grad_scaler(device: torch.device, enabled: bool):
    amp_enabled = enabled and device.type == "cuda"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(device.type, enabled=amp_enabled)
    return torch.cuda.amp.GradScaler(enabled=amp_enabled)


def autocast_context_manager(device: torch.device, enabled: bool):
    amp_enabled = enabled and device.type == "cuda"
    if not amp_enabled:
        return contextlib.nullcontext
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return lambda: torch.amp.autocast(device_type=device.type, dtype=torch.float16)
    return torch.cuda.amp.autocast


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


def _serialize_arg_value(value):
    if isinstance(value, Path):
        return str(value)
    return value


def serialize_args(args: argparse.Namespace) -> dict[str, object]:
    return {key: _serialize_arg_value(value) for key, value in vars(args).items()}


def configure_from_checkpoint(args: argparse.Namespace, checkpoint: dict) -> None:
    model_config = checkpoint.get("model_config", {})
    checkpoint_use_deltas = bool(checkpoint.get("args", {}).get("use_deltas", model_config.get("in_channels", 1) > 1))
    checkpoint_base_channels = int(model_config.get("base_channels", args.base_channels))
    checkpoint_num_subbands = int(model_config.get("num_subbands", args.num_subbands))

    if (
        args.use_deltas != checkpoint_use_deltas
        or args.base_channels != checkpoint_base_channels
        or args.num_subbands != checkpoint_num_subbands
    ):
        print("Resuming with model settings stored in the checkpoint.")

    args.use_deltas = checkpoint_use_deltas
    args.base_channels = checkpoint_base_channels
    args.num_subbands = checkpoint_num_subbands


def resolve_training_stats(
    args: argparse.Namespace,
    train_pairs,
    checkpoint: dict | None,
) -> tuple[dict[str, float] | None, Path | None]:
    if checkpoint is not None and "stats" in checkpoint:
        stats = checkpoint["stats"]
        if stats is None:
            return None, None

        resolved_path = args.stats_path or (args.output_dir / "stats.json")
        if not resolved_path.exists():
            save_stats(stats, resolved_path)
        return stats, resolved_path

    return resolve_stats(args.norm, args.stats_path, args.output_dir, train_pairs)


def initialize_metrics_csv(path: Path, resume_epoch: int = 0) -> None:
    fieldnames = ["epoch", "train_loss", "val_loss", "val_mae", "val_ssim", "val_lsd", "lr"]

    if resume_epoch <= 0 or not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
        return

    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = [row for row in reader if int(row["epoch"]) <= resume_epoch]

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_metrics_csv(
    path: Path,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    lr: float,
) -> None:
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "val_loss", "val_mae", "val_ssim", "val_lsd", "lr"],
        )
        writer.writerow(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "val_ssim": val_metrics["ssim"],
                "val_lsd": val_metrics["lsd"],
                "lr": lr,
            }
        )


def save_checkpoint(
    path: Path,
    model: ProgressiveMelRefiner,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler | None,
    epoch: int,
    args: argparse.Namespace,
    stats: dict[str, float] | None,
    best_val_loss: float,
    best_epoch: int,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "args": serialize_args(args),
        "stats": stats,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
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

    autocast_context = autocast_context_manager(device, amp_enabled)

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
    resume_checkpoint = (
        torch.load(args.resume, map_location=device, weights_only=False) if args.resume is not None else None
    )
    if resume_checkpoint is not None:
        configure_from_checkpoint(args, resume_checkpoint)

    all_pairs = collect_paired_files(args.low_dir, args.high_dir, pairing_mode=args.pairing_mode)
    train_pairs, val_pairs = split_pairs(
        all_pairs,
        val_ratio=args.val_ratio,
        seed=args.seed,
        group_separator=args.group_separator,
    )
    stats, resolved_stats_path = resolve_training_stats(args, train_pairs, resume_checkpoint)

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
    scaler = create_grad_scaler(device, args.amp)
    metrics_csv_path = args.output_dir / "metrics.csv"

    print(f"Device: {device}")
    print(f"Training pairs: {len(train_pairs)} | Validation pairs: {len(val_pairs)}")
    print(f"Input channels: {input_channels} | Shared stats: {resolved_stats_path or 'disabled'}")

    start_epoch = 0
    best_val_loss = math.inf
    best_epoch = 0
    patience_counter = 0

    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])

        scheduler_state_dict = resume_checkpoint.get("scheduler_state_dict")
        if scheduler is not None and scheduler_state_dict is not None:
            scheduler.load_state_dict(scheduler_state_dict)

        scaler_state_dict = resume_checkpoint.get("scaler_state_dict")
        if scaler is not None and scaler_state_dict is not None and scaler.is_enabled():
            scaler.load_state_dict(scaler_state_dict)

        start_epoch = int(resume_checkpoint.get("epoch", 0))
        best_val_loss = float(resume_checkpoint.get("best_val_loss", math.inf))
        best_epoch = int(resume_checkpoint.get("best_epoch", start_epoch if math.isfinite(best_val_loss) else 0))
        print(f"Resuming from checkpoint {args.resume} at epoch {start_epoch}.")

    if args.epochs <= start_epoch:
        raise ValueError(
            f"--epochs must be greater than the checkpoint epoch when resuming. "
            f"Got --epochs {args.epochs} and checkpoint epoch {start_epoch}."
        )

    initialize_metrics_csv(metrics_csv_path, resume_epoch=start_epoch)

    for epoch in range(start_epoch + 1, args.epochs + 1):
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

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                args.output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                args=args,
                stats=stats,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
            )
        else:
            patience_counter += 1

        save_checkpoint(
            args.output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            args=args,
            stats=stats,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
        )
        append_metrics_csv(metrics_csv_path, epoch, train_metrics, val_metrics, current_lr)

        if patience_counter >= args.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best validation loss {best_val_loss:.5f} was reached at epoch {best_epoch}."
            )
            break

    print(f"Best checkpoint: {args.output_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
