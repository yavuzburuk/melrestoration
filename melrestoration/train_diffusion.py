from __future__ import annotations

import argparse
import contextlib
import csv
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    try:
        from melrestoration.data import (
            PairedMelDataset,
            collect_paired_files,
            compute_shared_stats,
            denormalize_tensor,
            load_stats,
            save_stats,
            split_pairs,
        )
        from melrestoration.diffusion import ConditionalDiffusionUNet, GaussianDiffusion
        from melrestoration.metrics import lsd_metric, mae_metric, ssim_metric
    except ModuleNotFoundError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from data import (  # type: ignore[no-redef]
            PairedMelDataset,
            collect_paired_files,
            compute_shared_stats,
            denormalize_tensor,
            load_stats,
            save_stats,
            split_pairs,
        )
        from diffusion import ConditionalDiffusionUNet, GaussianDiffusion  # type: ignore[no-redef]
        from metrics import lsd_metric, mae_metric, ssim_metric  # type: ignore[no-redef]
else:
    from .data import (
        PairedMelDataset,
        collect_paired_files,
        compute_shared_stats,
        denormalize_tensor,
        load_stats,
        save_stats,
        split_pairs,
    )
    from .diffusion import ConditionalDiffusionUNet, GaussianDiffusion
    from .metrics import lsd_metric, mae_metric, ssim_metric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a conditional diffusion mel-spectrogram refiner.")
    parser.add_argument("--low-dir", type=Path, required=True, help="Directory with VAE/RVQ-VAE coarse mel .npy files.")
    parser.add_argument("--high-dir", type=Path, required=True, help="Directory with target high-detail mel .npy files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store checkpoints and stats.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume training from a saved diffusion checkpoint.")
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
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--channel-mults", type=str, default="1,2,4,4")
    parser.add_argument("--time-channels", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--use-attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-schedule", choices=("cosine", "linear"), default="cosine")
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--target-mode", choices=("residual", "mel"), default="residual")
    parser.add_argument("--noise-loss", choices=("mse", "l1"), default="mse")
    parser.add_argument("--x0-loss-weight", type=float, default=0.1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--scheduler", choices=("cosine", "plateau", "none"), default="cosine")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--plateau-patience", type=int, default=8)
    parser.add_argument("--early-stopping-patience", type=int, default=30)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def parse_channel_mults(value: str) -> tuple[int, ...]:
    try:
        multipliers = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid --channel-mults value: {value}") from exc
    if not multipliers or any(multiplier < 1 for multiplier in multipliers):
        raise ValueError("--channel-mults must be a comma-separated list of positive integers.")
    return multipliers


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
    diffusion_config = checkpoint.get("diffusion_config", {})
    checkpoint_args = checkpoint.get("args", {})

    args.use_deltas = bool(checkpoint_args.get("use_deltas", model_config.get("cond_channels", 1) > 1))
    args.base_channels = int(model_config.get("base_channels", args.base_channels))
    args.channel_mults = ",".join(str(item) for item in model_config.get("channel_mults", parse_channel_mults(args.channel_mults)))
    args.time_channels = model_config.get("time_channels", args.time_channels)
    args.dropout = float(model_config.get("dropout", args.dropout))
    args.use_attention = bool(model_config.get("use_attention", args.use_attention))
    args.target_mode = str(checkpoint_args.get("target_mode", args.target_mode))
    args.timesteps = int(diffusion_config.get("timesteps", args.timesteps))
    args.beta_schedule = str(diffusion_config.get("beta_schedule", args.beta_schedule))
    args.beta_start = float(diffusion_config.get("beta_start", args.beta_start))
    args.beta_end = float(diffusion_config.get("beta_end", args.beta_end))


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
    fieldnames = [
        "epoch",
        "train_loss",
        "train_noise_loss",
        "train_x0_l1",
        "val_loss",
        "val_noise_loss",
        "val_x0_l1",
        "val_mae",
        "val_ssim",
        "val_lsd",
        "lr",
    ]

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
            fieldnames=[
                "epoch",
                "train_loss",
                "train_noise_loss",
                "train_x0_l1",
                "val_loss",
                "val_noise_loss",
                "val_x0_l1",
                "val_mae",
                "val_ssim",
                "val_lsd",
                "lr",
            ],
        )
        writer.writerow(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_noise_loss": train_metrics["noise_loss"],
                "train_x0_l1": train_metrics["x0_l1"],
                "val_loss": val_metrics["loss"],
                "val_noise_loss": val_metrics["noise_loss"],
                "val_x0_l1": val_metrics["x0_l1"],
                "val_mae": val_metrics["mae"],
                "val_ssim": val_metrics["ssim"],
                "val_lsd": val_metrics["lsd"],
                "lr": lr,
            }
        )


def save_checkpoint(
    path: Path,
    model: ConditionalDiffusionUNet,
    diffusion: GaussianDiffusion,
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
        "model_config": model.config(),
        "diffusion_config": diffusion.config(),
    }
    torch.save(checkpoint, path)


def diffusion_training_target(low: torch.Tensor, target: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "residual":
        return target - low
    if mode == "mel":
        return target
    raise ValueError(f"Unsupported diffusion target mode: {mode}")


def restored_from_diffusion_target(low: torch.Tensor, predicted_x0: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "residual":
        return low + predicted_x0
    if mode == "mel":
        return predicted_x0
    raise ValueError(f"Unsupported diffusion target mode: {mode}")


def diffusion_noise_loss(prediction: torch.Tensor, target: torch.Tensor, loss_kind: str) -> torch.Tensor:
    if loss_kind == "mse":
        return F.mse_loss(prediction, target)
    if loss_kind == "l1":
        return F.l1_loss(prediction, target)
    raise ValueError(f"Unsupported noise loss: {loss_kind}")


def run_epoch(
    model: ConditionalDiffusionUNet,
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    stats: dict[str, float] | None,
    amp_enabled: bool,
    target_mode: str,
    noise_loss_kind: str,
    x0_loss_weight: float,
    grad_accum_steps: int,
    grad_clip: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    grad_accum_steps = max(1, grad_accum_steps)

    totals = {
        "loss": 0.0,
        "noise_loss": 0.0,
        "x0_l1": 0.0,
        "mae": 0.0,
        "ssim": 0.0,
        "lsd": 0.0,
    }
    count = 0
    autocast_context = autocast_context_manager(device, amp_enabled)

    if training:
        optimizer.zero_grad(set_to_none=True)

    total_batches = len(loader)
    for step, batch in enumerate(loader, start=1):
        condition = batch["input"].to(device, non_blocking=True)
        low = batch["low"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        batch_size = condition.size(0)

        with torch.set_grad_enabled(training):
            with autocast_context():
                x_start = diffusion_training_target(low, target, target_mode)
                timesteps = torch.randint(0, diffusion.timesteps, (batch_size,), device=device, dtype=torch.long)
                noise = torch.randn_like(x_start)
                noisy = diffusion.q_sample(x_start, timesteps, noise)
                predicted_noise = model(noisy, condition, timesteps)
                noise_loss = diffusion_noise_loss(predicted_noise, noise, noise_loss_kind)
                predicted_x0 = diffusion.predict_x_start_from_noise(noisy, timesteps, predicted_noise)
                restored = restored_from_diffusion_target(low, predicted_x0, target_mode)
                x0_l1 = F.l1_loss(restored, target)
                total_loss = noise_loss + (x0_loss_weight * x0_l1)

        if training:
            scaled_loss = total_loss / grad_accum_steps
            if scaler is not None and scaler.is_enabled():
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            should_step = step % grad_accum_steps == 0 or step == total_batches
            if should_step:
                if grad_clip > 0:
                    if scaler is not None and scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                if scaler is not None and scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        restored_eval = denormalize_tensor(restored.detach(), stats)
        target_eval = denormalize_tensor(target.detach(), stats)

        totals["loss"] += float(total_loss.detach().item()) * batch_size
        totals["noise_loss"] += float(noise_loss.detach().item()) * batch_size
        totals["x0_l1"] += float(x0_l1.detach().item()) * batch_size
        totals["mae"] += float(mae_metric(restored_eval, target_eval).item()) * batch_size
        totals["ssim"] += float(ssim_metric(restored_eval, target_eval).item()) * batch_size
        totals["lsd"] += float(lsd_metric(restored_eval, target_eval).item()) * batch_size
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

    cond_channels = 3 if args.use_deltas else 1
    channel_mults = parse_channel_mults(args.channel_mults)
    model = ConditionalDiffusionUNet(
        cond_channels=cond_channels,
        base_channels=args.base_channels,
        channel_mults=channel_mults,
        time_channels=args.time_channels,
        dropout=args.dropout,
        use_attention=args.use_attention,
    ).to(device)
    diffusion = GaussianDiffusion(
        timesteps=args.timesteps,
        beta_schedule=args.beta_schedule,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = create_scheduler(args, optimizer)
    scaler = create_grad_scaler(device, args.amp)
    metrics_csv_path = args.output_dir / "diffusion_metrics.csv"

    print(f"Device: {device}")
    print(f"Training pairs: {len(train_pairs)} | Validation pairs: {len(val_pairs)}")
    print(f"Condition channels: {cond_channels} | Shared stats: {resolved_stats_path or 'disabled'}")
    print(f"Target mode: {args.target_mode} | Timesteps: {args.timesteps} | Schedule: {args.beta_schedule}")

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
            diffusion=diffusion,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            stats=stats,
            amp_enabled=args.amp,
            target_mode=args.target_mode,
            noise_loss_kind=args.noise_loss,
            x0_loss_weight=args.x0_loss_weight,
            grad_accum_steps=args.grad_accum_steps,
            grad_clip=args.grad_clip,
        )

        val_metrics = run_epoch(
            model=model,
            diffusion=diffusion,
            loader=val_loader,
            optimizer=None,
            scaler=None,
            device=device,
            stats=stats,
            amp_enabled=args.amp,
            target_mode=args.target_mode,
            noise_loss_kind=args.noise_loss,
            x0_loss_weight=args.x0_loss_weight,
            grad_accum_steps=1,
            grad_clip=0.0,
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
                args.output_dir / "best_diffusion.pt",
                model=model,
                diffusion=diffusion,
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
            args.output_dir / "last_diffusion.pt",
            model=model,
            diffusion=diffusion,
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

    print(f"Best checkpoint: {args.output_dir / 'best_diffusion.pt'}")


if __name__ == "__main__":
    main()
