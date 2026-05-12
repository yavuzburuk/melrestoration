from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from .data import compute_delta_features, load_mel, normalize
from .diffusion import ConditionalDiffusionUNet, GaussianDiffusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run conditional diffusion mel refinement from a saved checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Single .npy file or directory with .npy files.")
    parser.add_argument("--output", type=Path, required=True, help="Output .npy path or output directory.")
    parser.add_argument("--sampler", choices=("ddim", "ddpm"), default="ddim")
    parser.add_argument("--sample-steps", type=int, default=50, help="DDIM steps. Ignored when --sampler ddpm is used.")
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM stochasticity. Use 0 for deterministic sampling.")
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
) -> tuple[ConditionalDiffusionUNet, GaussianDiffusion, dict[str, float] | None, bool, str]:
    model_config = checkpoint["model_config"]
    model = ConditionalDiffusionUNet(
        cond_channels=int(model_config["cond_channels"]),
        base_channels=int(model_config["base_channels"]),
        channel_mults=tuple(int(item) for item in model_config["channel_mults"]),
        time_channels=int(model_config["time_channels"]),
        dropout=float(model_config["dropout"]),
        use_attention=bool(model_config["use_attention"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
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
    return model, diffusion, stats, use_deltas, target_mode


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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model, diffusion, stats, use_deltas, target_mode = build_model(checkpoint, device)

    items = iter_inputs(args.input)
    if not items:
        raise FileNotFoundError(f"No .npy files found in {args.input}.")

    for input_path, relative_path in items:
        mel = load_mel(input_path)
        condition, low = build_condition_tensor(mel, stats=stats, use_deltas=use_deltas)
        condition = condition.to(device)
        low = low.to(device)

        shape = (1, 1, condition.shape[-2], condition.shape[-1])
        with torch.no_grad():
            if args.sampler == "ddpm":
                sampled_x0 = diffusion.sample_ddpm(model, condition, shape)
            else:
                sampled_x0 = diffusion.sample_ddim(
                    model,
                    condition,
                    shape,
                    steps=args.sample_steps,
                    eta=args.eta,
                )
            prediction = restore_prediction(low, sampled_x0, target_mode)[0, 0].detach().cpu().numpy()

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
