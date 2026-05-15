from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .data import compute_delta_features, load_mel, normalize
    from .diffusion import ConditionalDiffusionUNet, GaussianDiffusion
else:
    from data import compute_delta_features, load_mel, normalize
    from diffusion import ConditionalDiffusionUNet, GaussianDiffusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run v2 conditional diffusion mel refinement with EMA-aware inference.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Single .npy file or directory with .npy files.")
    parser.add_argument("--output", type=Path, required=True, help="Output .npy path or output directory.")
    parser.add_argument("--weights", choices=("ema", "model"), default="ema", help="Use EMA weights when available.")
    parser.add_argument("--sampler", choices=("ddim", "ddpm"), default="ddim")
    parser.add_argument("--sample-steps", type=int, default=75, help="DDIM steps. Ignored when --sampler ddpm is used.")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM stochasticity. Use 0 for deterministic sampling.")
    parser.add_argument(
        "--init-mode",
        choices=("condition", "random"),
        default="condition",
        help="Use 'condition' to denoise from the low mel/zero residual, or 'random' for full generation.",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.25,
        help="Noise strength for --init-mode condition. Lower values stay closer to the input.",
    )
    parser.add_argument(
        "--x0-clip",
        type=float,
        default=4.0,
        help="Clamp predicted diffusion target in normalized space during sampling. Use 0 to disable.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clip-min", type=float, default=None, help="Optional output clamp in original mel scale.")
    parser.add_argument("--clip-max", type=float, default=None, help="Optional output clamp in original mel scale.")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def choose_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(
    checkpoint: dict,
    device: torch.device,
    weights: str,
) -> tuple[
    ConditionalDiffusionUNet,
    GaussianDiffusion,
    dict[str, float] | None,
    dict[str, float] | None,
    bool,
    str,
    str,
]:
    model_config = checkpoint["model_config"]
    model = ConditionalDiffusionUNet(
        cond_channels=int(model_config["cond_channels"]),
        base_channels=int(model_config["base_channels"]),
        channel_mults=tuple(int(item) for item in model_config["channel_mults"]),
        time_channels=int(model_config["time_channels"]),
        dropout=float(model_config["dropout"]),
        use_attention=bool(model_config["use_attention"]),
    )

    selected_weights = "model"
    state_dict = checkpoint["model_state_dict"]
    if weights == "ema":
        ema_state_dict = checkpoint.get("ema_model_state_dict")
        if ema_state_dict is not None:
            state_dict = ema_state_dict
            selected_weights = "ema"

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    diffusion_config = checkpoint["diffusion_config"]
    diffusion = GaussianDiffusion(
        timesteps=int(diffusion_config["timesteps"]),
        beta_schedule=str(diffusion_config["beta_schedule"]),
        beta_start=float(diffusion_config["beta_start"]),
        beta_end=float(diffusion_config["beta_end"]),
    ).to(device)

    args = checkpoint.get("args", {})
    use_deltas = bool(args.get("use_deltas", model_config["cond_channels"] > 1))
    target_mode = str(args.get("target_mode", "residual"))
    stats = checkpoint.get("stats")
    target_stats = checkpoint.get("target_stats")
    return model, diffusion, stats, target_stats, use_deltas, target_mode, selected_weights


def build_condition_tensor(
    array: np.ndarray,
    stats: dict[str, float] | None,
    use_deltas: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    low = normalize(array.astype(np.float32, copy=False), stats)
    channels = [low]
    if use_deltas:
        delta_time, delta_freq = compute_delta_features(low)
        channels.extend([delta_time, delta_freq])

    condition = np.stack(channels, axis=0).astype(np.float32, copy=False)
    condition_tensor = torch.from_numpy(condition[None, ...])
    low_tensor = torch.from_numpy(low[None, None, ...].astype(np.float32, copy=False))
    return condition_tensor, low_tensor


def denormalize_array(array: np.ndarray, stats: dict[str, float] | None) -> np.ndarray:
    if stats is None:
        return array
    return array * max(float(stats["std"]), 1e-6) + float(stats["mean"])


def normalize_diffusion_target(target: torch.Tensor, target_stats: dict[str, float] | None) -> torch.Tensor:
    if target_stats is None:
        return target
    return (target - float(target_stats["mean"])) / max(float(target_stats["std"]), 1e-6)


def denormalize_diffusion_target(target: torch.Tensor, target_stats: dict[str, float] | None) -> torch.Tensor:
    if target_stats is None:
        return target
    return target * max(float(target_stats["std"]), 1e-6) + float(target_stats["mean"])


def iter_inputs(input_path: Path) -> list[tuple[Path, Path]]:
    if input_path.is_file():
        return [(input_path, Path(input_path.name))]
    return [(path, path.relative_to(input_path)) for path in sorted(input_path.rglob("*.npy"))]


def restore_prediction(low: torch.Tensor, sampled_x0: torch.Tensor, target_mode: str) -> torch.Tensor:
    if target_mode == "residual":
        return low + sampled_x0
    if target_mode == "mel":
        return sampled_x0
    raise ValueError(f"Unsupported diffusion target mode in checkpoint: {target_mode}")


def build_initial_diffusion_target(
    low: torch.Tensor,
    target_mode: str,
    target_stats: dict[str, float] | None,
    init_mode: str,
) -> torch.Tensor | None:
    if init_mode == "random":
        return None
    if target_mode == "residual":
        initial = torch.zeros_like(low)
    elif target_mode == "mel":
        initial = low
    else:
        raise ValueError(f"Unsupported diffusion target mode in checkpoint: {target_mode}")
    return normalize_diffusion_target(initial, target_stats)


def resolve_start_timestep(diffusion: GaussianDiffusion, strength: float, init_mode: str) -> int | None:
    if init_mode == "random":
        return None
    strength = max(0.0, min(float(strength), 1.0))
    return int(round((diffusion.timesteps - 1) * strength))


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, diffusion, stats, target_stats, use_deltas, target_mode, selected_weights = build_model(
        checkpoint,
        device,
        weights=args.weights,
    )
    x0_clip = None if args.x0_clip is None or args.x0_clip <= 0 else args.x0_clip

    items = iter_inputs(args.input)
    if not items:
        raise FileNotFoundError(f"No .npy files found in {args.input}.")

    print(f"Using {selected_weights} weights from {args.checkpoint}.")
    for input_path, relative_path in items:
        mel = load_mel(input_path)
        condition, low = build_condition_tensor(mel, stats=stats, use_deltas=use_deltas)
        condition = condition.to(device)
        low = low.to(device)

        shape = (1, 1, condition.shape[-2], condition.shape[-1])
        initial_x = build_initial_diffusion_target(low, target_mode, target_stats, args.init_mode)
        start_timestep = resolve_start_timestep(diffusion, args.strength, args.init_mode)
        with torch.no_grad():
            if args.sampler == "ddpm":
                sampled_x0 = diffusion.sample_ddpm(
                    model,
                    condition,
                    shape,
                    initial_x=initial_x,
                    start_timestep=start_timestep,
                    x0_clip=x0_clip,
                )
            else:
                sampled_x0 = diffusion.sample_ddim(
                    model,
                    condition,
                    shape,
                    steps=args.sample_steps,
                    eta=args.eta,
                    initial_x=initial_x,
                    start_timestep=start_timestep,
                    x0_clip=x0_clip,
                )
            raw_sampled_x0 = denormalize_diffusion_target(sampled_x0, target_stats)
            prediction = restore_prediction(low, raw_sampled_x0, target_mode)[0, 0].detach().cpu().numpy()

        prediction = denormalize_array(prediction, stats).astype(np.float32, copy=False)
        if args.clip_min is not None or args.clip_max is not None:
            clip_min = -np.inf if args.clip_min is None else args.clip_min
            clip_max = np.inf if args.clip_max is None else args.clip_max
            prediction = np.clip(prediction, clip_min, clip_max).astype(np.float32, copy=False)

        if args.input.is_file():
            output_path = args.output
        else:
            output_path = args.output / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, prediction)

    print(f"Saved {len(items)} diffusion-refined mel file(s) to {args.output}.")


if __name__ == "__main__":
    main()
