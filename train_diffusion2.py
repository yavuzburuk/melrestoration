from __future__ import annotations

import argparse
import contextlib
import copy
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
        load_mel,
        load_stats,
        normalize,
        save_stats,
        split_pairs,
    )
    from .diffusion import ConditionalDiffusionUNet, GaussianDiffusion
    from .metrics import lsd_metric, mae_metric, ssim_metric
else:
    from data import (
        PairedMelDataset,
        collect_paired_files,
        compute_shared_stats,
        denormalize_tensor,
        load_mel,
        load_stats,
        normalize,
        save_stats,
        split_pairs,
    )
    from diffusion import ConditionalDiffusionUNet, GaussianDiffusion
    from metrics import lsd_metric, mae_metric, ssim_metric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a v2 conditional diffusion mel refiner with EMA and sampled validation.")
    parser.add_argument("--low-dir", type=Path, required=True, help="Directory with VAE/RVQ-VAE coarse mel .npy files.")
    parser.add_argument("--high-dir", type=Path, required=True, help="Directory with target high-detail mel .npy files.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to store checkpoints and stats.")
    parser.add_argument("--resume", type=Path, default=None, help="Resume training from a saved v2 diffusion checkpoint.")
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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--base-channels", type=int, default=96)
    parser.add_argument("--channel-mults", type=str, default="1,2,4,4")
    parser.add_argument("--time-channels", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--use-attention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-schedule", choices=("cosine", "linear"), default="cosine")
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--target-mode", choices=("residual", "mel"), default="residual")
    parser.add_argument("--diffusion-target-norm", choices=("none", "zscore"), default="zscore")
    parser.add_argument("--target-stats-path", type=Path, default=None)
    parser.add_argument("--noise-loss", choices=("mse", "l1"), default="mse")
    parser.add_argument("--x0-loss-weight", type=float, default=2.0)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--ema-update-after-step", type=int, default=0)
    parser.add_argument("--sampled-val-every", type=int, default=5, help="Run sampled validation every N epochs. Use 0 to disable.")
    parser.add_argument("--sampled-val-batches", type=int, default=2, help="Number of validation batches to sample.")
    parser.add_argument("--sampled-val-sample-steps", type=int, default=50)
    parser.add_argument("--sampled-val-strength", type=float, default=0.25)
    parser.add_argument("--sampled-val-x0-clip", type=float, default=4.0)
    parser.add_argument("--sampled-val-eta", type=float, default=0.0)
    parser.add_argument("--scheduler", choices=("cosine", "plateau", "none"), default="cosine")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--plateau-patience", type=int, default=8)
    parser.add_argument("--early-stopping-patience", type=int, default=40)
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
    args.diffusion_target_norm = str(
        checkpoint_args.get(
            "diffusion_target_norm",
            "zscore" if checkpoint.get("target_stats") is not None else "none",
        )
    )
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


def compute_diffusion_target_stats(
    pairs,
    shared_stats: dict[str, float] | None,
    target_mode: str,
) -> dict[str, float]:
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    for pair in pairs:
        low = normalize(load_mel(pair.low_path), shared_stats).astype(np.float64, copy=False)
        high = normalize(load_mel(pair.high_path), shared_stats).astype(np.float64, copy=False)
        if target_mode == "residual":
            target = high - low
        elif target_mode == "mel":
            target = high
        else:
            raise ValueError(f"Unsupported diffusion target mode: {target_mode}")

        total_sum += float(target.sum())
        total_sq_sum += float(np.square(target).sum())
        total_count += int(target.size)

    if total_count == 0:
        raise ValueError("Cannot compute diffusion target statistics for an empty dataset.")

    mean = total_sum / total_count
    variance = max((total_sq_sum / total_count) - (mean * mean), 1e-12)
    return {"mean": float(mean), "std": float(math.sqrt(variance))}


def resolve_diffusion_target_stats(
    args: argparse.Namespace,
    train_pairs,
    shared_stats: dict[str, float] | None,
    checkpoint: dict | None,
) -> tuple[dict[str, float] | None, Path | None]:
    if checkpoint is not None and "target_stats" in checkpoint:
        target_stats = checkpoint["target_stats"]
        if target_stats is None:
            return None, None

        resolved_path = args.target_stats_path or (args.output_dir / "diffusion_target_stats.json")
        if not resolved_path.exists():
            save_stats(target_stats, resolved_path)
        return target_stats, resolved_path

    if args.diffusion_target_norm == "none":
        return None, None

    resolved_path = args.target_stats_path or (args.output_dir / "diffusion_target_stats.json")
    if resolved_path.exists():
        target_stats = load_stats(resolved_path)
    else:
        target_stats = compute_diffusion_target_stats(train_pairs, shared_stats, args.target_mode)
        save_stats(target_stats, resolved_path)
    return target_stats, resolved_path


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
        "sampled_val_mae",
        "sampled_val_ssim",
        "sampled_val_lsd",
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
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def append_metrics_csv(
    path: Path,
    epoch: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
    sampled_metrics: dict[str, float] | None,
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
                "sampled_val_mae",
                "sampled_val_ssim",
                "sampled_val_lsd",
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
                "sampled_val_mae": "" if sampled_metrics is None else sampled_metrics["mae"],
                "sampled_val_ssim": "" if sampled_metrics is None else sampled_metrics["ssim"],
                "sampled_val_lsd": "" if sampled_metrics is None else sampled_metrics["lsd"],
                "lr": lr,
            }
        )


def create_ema_model(model: ConditionalDiffusionUNet) -> ConditionalDiffusionUNet:
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    return ema_model


@torch.no_grad()
def copy_model_to_ema(ema_model: ConditionalDiffusionUNet, model: ConditionalDiffusionUNet) -> None:
    ema_model.load_state_dict(model.state_dict())


@torch.no_grad()
def update_ema_model(
    ema_model: ConditionalDiffusionUNet,
    model: ConditionalDiffusionUNet,
    decay: float,
) -> None:
    ema_state = ema_model.state_dict()
    model_state = model.state_dict()
    for name, ema_value in ema_state.items():
        model_value = model_state[name]
        if torch.is_floating_point(ema_value):
            ema_value.mul_(decay).add_(model_value.to(device=ema_value.device, dtype=ema_value.dtype), alpha=1.0 - decay)
        else:
            ema_value.copy_(model_value)


def diffusion_training_target(low: torch.Tensor, target: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "residual":
        return target - low
    if mode == "mel":
        return target
    raise ValueError(f"Unsupported diffusion target mode: {mode}")


def normalize_diffusion_target(
    target: torch.Tensor,
    target_stats: dict[str, float] | None,
) -> torch.Tensor:
    if target_stats is None:
        return target
    return (target - float(target_stats["mean"])) / max(float(target_stats["std"]), 1e-6)


def denormalize_diffusion_target(
    target: torch.Tensor,
    target_stats: dict[str, float] | None,
) -> torch.Tensor:
    if target_stats is None:
        return target
    return target * max(float(target_stats["std"]), 1e-6) + float(target_stats["mean"])


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
    ema_model: ConditionalDiffusionUNet | None,
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler | None,
    device: torch.device,
    stats: dict[str, float] | None,
    target_stats: dict[str, float] | None,
    amp_enabled: bool,
    target_mode: str,
    noise_loss_kind: str,
    x0_loss_weight: float,
    grad_accum_steps: int,
    grad_clip: float,
    ema_decay: float,
    ema_update_after_step: int,
    global_step: int,
) -> tuple[dict[str, float], int]:
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
                raw_x_start = diffusion_training_target(low, target, target_mode)
                x_start = normalize_diffusion_target(raw_x_start, target_stats)
                timesteps = torch.randint(0, diffusion.timesteps, (batch_size,), device=device, dtype=torch.long)
                noise = torch.randn_like(x_start)
                noisy = diffusion.q_sample(x_start, timesteps, noise)
                predicted_noise = model(noisy, condition, timesteps)
                noise_loss = diffusion_noise_loss(predicted_noise, noise, noise_loss_kind)
                predicted_x0 = diffusion.predict_x_start_from_noise(noisy, timesteps, predicted_noise)
                raw_predicted_x0 = denormalize_diffusion_target(predicted_x0, target_stats)
                restored = restored_from_diffusion_target(low, raw_predicted_x0, target_mode)
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

                global_step += 1
                if ema_model is not None:
                    if global_step <= ema_update_after_step:
                        copy_model_to_ema(ema_model, model)
                    else:
                        update_ema_model(ema_model, model, decay=ema_decay)

        restored_eval = denormalize_tensor(restored.detach(), stats)
        target_eval = denormalize_tensor(target.detach(), stats)

        totals["loss"] += float(total_loss.detach().item()) * batch_size
        totals["noise_loss"] += float(noise_loss.detach().item()) * batch_size
        totals["x0_l1"] += float(x0_l1.detach().item()) * batch_size
        totals["mae"] += float(mae_metric(restored_eval, target_eval).item()) * batch_size
        totals["ssim"] += float(ssim_metric(restored_eval, target_eval).item()) * batch_size
        totals["lsd"] += float(lsd_metric(restored_eval, target_eval).item()) * batch_size
        count += batch_size

    return {name: value / max(count, 1) for name, value in totals.items()}, global_step


def build_initial_diffusion_target(
    low: torch.Tensor,
    target_mode: str,
    target_stats: dict[str, float] | None,
) -> torch.Tensor:
    if target_mode == "residual":
        initial = torch.zeros_like(low)
    elif target_mode == "mel":
        initial = low
    else:
        raise ValueError(f"Unsupported diffusion target mode: {target_mode}")
    return normalize_diffusion_target(initial, target_stats)


def resolve_start_timestep(diffusion: GaussianDiffusion, strength: float) -> int:
    strength = max(0.0, min(float(strength), 1.0))
    return int(round((diffusion.timesteps - 1) * strength))


@torch.no_grad()
def run_sampled_validation(
    model: ConditionalDiffusionUNet,
    diffusion: GaussianDiffusion,
    loader: DataLoader,
    device: torch.device,
    stats: dict[str, float] | None,
    target_stats: dict[str, float] | None,
    target_mode: str,
    max_batches: int,
    sample_steps: int,
    strength: float,
    x0_clip: float,
    eta: float,
) -> dict[str, float]:
    model.eval()
    totals = {"mae": 0.0, "ssim": 0.0, "lsd": 0.0}
    count = 0
    start_timestep = resolve_start_timestep(diffusion, strength)
    x0_clip_value = None if x0_clip <= 0 else x0_clip

    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break

        condition = batch["input"].to(device, non_blocking=True)
        low = batch["low"].to(device, non_blocking=True)
        target = batch["target"].to(device, non_blocking=True)
        initial_x = build_initial_diffusion_target(low, target_mode, target_stats)
        sampled_x0 = diffusion.sample_ddim(
            model,
            condition,
            tuple(low.shape),
            steps=sample_steps,
            eta=eta,
            initial_x=initial_x,
            start_timestep=start_timestep,
            x0_clip=x0_clip_value,
        )
        raw_sampled_x0 = denormalize_diffusion_target(sampled_x0, target_stats)
        restored = restored_from_diffusion_target(low, raw_sampled_x0, target_mode)

        restored_eval = denormalize_tensor(restored, stats)
        target_eval = denormalize_tensor(target, stats)
        batch_size = condition.size(0)
        totals["mae"] += float(mae_metric(restored_eval, target_eval).item()) * batch_size
        totals["ssim"] += float(ssim_metric(restored_eval, target_eval).item()) * batch_size
        totals["lsd"] += float(lsd_metric(restored_eval, target_eval).item()) * batch_size
        count += batch_size

    if count == 0:
        return {"mae": math.inf, "ssim": 0.0, "lsd": math.inf}
    return {name: value / count for name, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: ConditionalDiffusionUNet,
    ema_model: ConditionalDiffusionUNet | None,
    diffusion: GaussianDiffusion,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler | None,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
    stats: dict[str, float] | None,
    target_stats: dict[str, float] | None,
    best_val_loss: float,
    best_epoch: int,
    best_sampled_mae: float,
    best_sampled_epoch: int,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "ema_model_state_dict": ema_model.state_dict() if ema_model is not None else None,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "args": serialize_args(args),
        "stats": stats,
        "target_stats": target_stats,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "best_sampled_mae": best_sampled_mae,
        "best_sampled_epoch": best_sampled_epoch,
        "model_config": model.config(),
        "diffusion_config": diffusion.config(),
    }
    torch.save(checkpoint, path)


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
    target_stats, resolved_target_stats_path = resolve_diffusion_target_stats(
        args,
        train_pairs,
        shared_stats=stats,
        checkpoint=resume_checkpoint,
    )

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
    ema_model = create_ema_model(model) if args.ema else None

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = create_scheduler(args, optimizer)
    scaler = create_grad_scaler(device, args.amp)
    metrics_csv_path = args.output_dir / "diffusion2_metrics.csv"

    print(f"Device: {device}")
    print(f"Training pairs: {len(train_pairs)} | Validation pairs: {len(val_pairs)}")
    print(f"Condition channels: {cond_channels} | Shared stats: {resolved_stats_path or 'disabled'}")
    print(f"Diffusion target stats: {resolved_target_stats_path or 'disabled'}")
    print(f"Target mode: {args.target_mode} | Timesteps: {args.timesteps} | Schedule: {args.beta_schedule}")
    print(f"EMA: {'enabled' if ema_model is not None else 'disabled'} | Sampled validation every {args.sampled_val_every} epoch(s)")

    start_epoch = 0
    global_step = 0
    best_val_loss = math.inf
    best_epoch = 0
    best_sampled_mae = math.inf
    best_sampled_epoch = 0
    patience_counter = 0

    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        if ema_model is not None:
            ema_state_dict = resume_checkpoint.get("ema_model_state_dict")
            if ema_state_dict is not None:
                ema_model.load_state_dict(ema_state_dict)
            else:
                copy_model_to_ema(ema_model, model)

        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])

        scheduler_state_dict = resume_checkpoint.get("scheduler_state_dict")
        if scheduler is not None and scheduler_state_dict is not None:
            scheduler.load_state_dict(scheduler_state_dict)

        scaler_state_dict = resume_checkpoint.get("scaler_state_dict")
        if scaler is not None and scaler_state_dict is not None and scaler.is_enabled():
            scaler.load_state_dict(scaler_state_dict)

        start_epoch = int(resume_checkpoint.get("epoch", 0))
        global_step = int(resume_checkpoint.get("global_step", 0))
        best_val_loss = float(resume_checkpoint.get("best_val_loss", math.inf))
        best_epoch = int(resume_checkpoint.get("best_epoch", start_epoch if math.isfinite(best_val_loss) else 0))
        best_sampled_mae = float(resume_checkpoint.get("best_sampled_mae", math.inf))
        best_sampled_epoch = int(resume_checkpoint.get("best_sampled_epoch", 0))
        print(f"Resuming from checkpoint {args.resume} at epoch {start_epoch}.")

    if args.epochs <= start_epoch:
        raise ValueError(
            f"--epochs must be greater than the checkpoint epoch when resuming. "
            f"Got --epochs {args.epochs} and checkpoint epoch {start_epoch}."
        )

    initialize_metrics_csv(metrics_csv_path, resume_epoch=start_epoch)

    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_metrics, global_step = run_epoch(
            model=model,
            ema_model=ema_model,
            diffusion=diffusion,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            stats=stats,
            target_stats=target_stats,
            amp_enabled=args.amp,
            target_mode=args.target_mode,
            noise_loss_kind=args.noise_loss,
            x0_loss_weight=args.x0_loss_weight,
            grad_accum_steps=args.grad_accum_steps,
            grad_clip=args.grad_clip,
            ema_decay=args.ema_decay,
            ema_update_after_step=args.ema_update_after_step,
            global_step=global_step,
        )

        eval_model = ema_model if ema_model is not None else model
        val_metrics, _ = run_epoch(
            model=eval_model,
            ema_model=None,
            diffusion=diffusion,
            loader=val_loader,
            optimizer=None,
            scaler=None,
            device=device,
            stats=stats,
            target_stats=target_stats,
            amp_enabled=args.amp,
            target_mode=args.target_mode,
            noise_loss_kind=args.noise_loss,
            x0_loss_weight=args.x0_loss_weight,
            grad_accum_steps=1,
            grad_clip=0.0,
            ema_decay=args.ema_decay,
            ema_update_after_step=args.ema_update_after_step,
            global_step=global_step,
        )

        sampled_metrics = None
        should_sample_validate = args.sampled_val_every > 0 and epoch % args.sampled_val_every == 0
        if should_sample_validate:
            sampled_metrics = run_sampled_validation(
                model=eval_model,
                diffusion=diffusion,
                loader=val_loader,
                device=device,
                stats=stats,
                target_stats=target_stats,
                target_mode=args.target_mode,
                max_batches=args.sampled_val_batches,
                sample_steps=args.sampled_val_sample_steps,
                strength=args.sampled_val_strength,
                x0_clip=args.sampled_val_x0_clip,
                eta=args.sampled_val_eta,
            )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics["loss"])
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        message = (
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_metrics['loss']:.5f} | "
            f"val_loss={val_metrics['loss']:.5f} | "
            f"val_mae={val_metrics['mae']:.5f} | "
            f"val_ssim={val_metrics['ssim']:.5f} | "
            f"val_lsd={val_metrics['lsd']:.5f}"
        )
        if sampled_metrics is not None:
            message += (
                f" | sampled_mae={sampled_metrics['mae']:.5f} | "
                f"sampled_ssim={sampled_metrics['ssim']:.5f} | "
                f"sampled_lsd={sampled_metrics['lsd']:.5f}"
            )
        message += f" | lr={current_lr:.2e}"
        print(message)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                args.output_dir / "best_diffusion2.pt",
                model=model,
                ema_model=ema_model,
                diffusion=diffusion,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                args=args,
                stats=stats,
                target_stats=target_stats,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                best_sampled_mae=best_sampled_mae,
                best_sampled_epoch=best_sampled_epoch,
            )
        else:
            patience_counter += 1

        if sampled_metrics is not None and sampled_metrics["mae"] < best_sampled_mae:
            best_sampled_mae = sampled_metrics["mae"]
            best_sampled_epoch = epoch
            save_checkpoint(
                args.output_dir / "best_sampled_diffusion2.pt",
                model=model,
                ema_model=ema_model,
                diffusion=diffusion,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                args=args,
                stats=stats,
                target_stats=target_stats,
                best_val_loss=best_val_loss,
                best_epoch=best_epoch,
                best_sampled_mae=best_sampled_mae,
                best_sampled_epoch=best_sampled_epoch,
            )

        save_checkpoint(
            args.output_dir / "last_diffusion2.pt",
            model=model,
            ema_model=ema_model,
            diffusion=diffusion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            args=args,
            stats=stats,
            target_stats=target_stats,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
            best_sampled_mae=best_sampled_mae,
            best_sampled_epoch=best_sampled_epoch,
        )
        append_metrics_csv(metrics_csv_path, epoch, train_metrics, val_metrics, sampled_metrics, current_lr)

        if patience_counter >= args.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch}. "
                f"Best validation loss {best_val_loss:.5f} was reached at epoch {best_epoch}."
            )
            break

    print(f"Best denoising-loss checkpoint: {args.output_dir / 'best_diffusion2.pt'}")
    if math.isfinite(best_sampled_mae):
        print(f"Best sampled-validation checkpoint: {args.output_dir / 'best_sampled_diffusion2.pt'}")


if __name__ == "__main__":
    main()
