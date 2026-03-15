from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .data import compute_delta_features, load_mel, normalize
from .models import ProgressiveMelRefiner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mel refinement inference from a saved checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Single .npy file or directory with .npy files.")
    parser.add_argument("--output", type=Path, required=True, help="Output .npy path or output directory.")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def choose_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(checkpoint: dict, device: torch.device) -> tuple[ProgressiveMelRefiner, dict[str, float] | None, bool]:
    model_config = checkpoint["model_config"]
    model = ProgressiveMelRefiner(
        in_channels=model_config["in_channels"],
        base_channels=model_config["base_channels"],
        num_subbands=model_config["num_subbands"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    args = checkpoint.get("args", {})
    use_deltas = bool(args.get("use_deltas", model_config["in_channels"] > 1))
    stats = checkpoint.get("stats")
    return model, stats, use_deltas


def build_input_tensor(array: np.ndarray, stats: dict[str, float] | None, use_deltas: bool) -> torch.Tensor:
    low = normalize(array.astype(np.float32, copy=False), stats)
    channels = [low]
    if use_deltas:
        delta_time, delta_freq = compute_delta_features(low)
        channels.extend([delta_time, delta_freq])

    model_input = np.stack(channels, axis=0).astype(np.float32, copy=False)
    return torch.from_numpy(model_input[None, ...])


def denormalize_array(array: np.ndarray, stats: dict[str, float] | None) -> np.ndarray:
    if stats is None:
        return array
    return array * max(float(stats["std"]), 1e-6) + float(stats["mean"])


def iter_inputs(input_path: Path) -> list[tuple[Path, Path]]:
    if input_path.is_file():
        return [(input_path, Path(input_path.name))]
    return [(path, path.relative_to(input_path)) for path in sorted(input_path.rglob("*.npy"))]


def main() -> None:
    args = parse_args()
    device = choose_device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model, stats, use_deltas = build_model(checkpoint, device)

    items = iter_inputs(args.input)
    if not items:
        raise FileNotFoundError(f"No .npy files found in {args.input}.")

    for input_path, relative_path in items:
        mel = load_mel(input_path)
        model_input = build_input_tensor(mel, stats=stats, use_deltas=use_deltas)
        model_input = model_input.to(device)

        with torch.no_grad():
            outputs = model(model_input)
            prediction = outputs["prediction"][0, 0].detach().cpu().numpy()

        prediction = denormalize_array(prediction, stats).astype(np.float32, copy=False)

        if args.input.is_file():
            output_path = args.output
        else:
            output_path = args.output / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, prediction)

    print(f"Saved {len(items)} refined mel file(s) to {args.output}.")


if __name__ == "__main__":
    main()
