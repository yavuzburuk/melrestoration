from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PairedMelPath:
    relative_path: Path
    low_path: Path
    high_path: Path


def load_mel(path: Path | str) -> np.ndarray:
    array = np.load(path)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D mel array in {path}, got shape {array.shape}.")
    return np.asarray(array, dtype=np.float32)


def _format_path_examples(paths: list[Path], root: Path, limit: int = 5) -> str:
    if not paths:
        return "none"
    examples = [str(path.relative_to(root)) for path in paths[:limit]]
    if len(paths) > limit:
        examples.append("...")
    return ", ".join(examples)


def _duplicate_filenames(paths: list[Path]) -> list[str]:
    counts: dict[str, int] = {}
    for path in paths:
        counts[path.name] = counts.get(path.name, 0) + 1
    return sorted(name for name, count in counts.items() if count > 1)


def _raise_no_pairs_error(
    low_root: Path,
    high_root: Path,
    low_files: list[Path],
    high_files: list[Path],
    pairing_mode: str,
) -> None:
    shared_names = sorted({path.name for path in low_files} & {path.name for path in high_files})
    basename_hint = (
        " Matching filenames were found, but not matching relative paths. "
        "Try --pairing-mode basename if the same filenames are stored under different subfolders."
        if pairing_mode == "relative" and shared_names
        else ""
    )
    raise FileNotFoundError(
        f"No paired .npy files found between {low_root} and {high_root}. "
        f"Pairing mode: {pairing_mode}. "
        f"Found {len(low_files)} low .npy file(s) and {len(high_files)} high .npy file(s). "
        f"Low examples: {_format_path_examples(low_files, low_root)}. "
        f"High examples: {_format_path_examples(high_files, high_root)}. "
        "Relative pairing requires identical paths under both roots, for example "
        "data/low/blues/example.npy and data/high/blues/example.npy."
        f"{basename_hint}"
    )


def collect_paired_files(
    low_dir: Path | str,
    high_dir: Path | str,
    pairing_mode: str = "relative",
) -> list[PairedMelPath]:
    low_root = Path(low_dir)
    high_root = Path(high_dir)
    if pairing_mode not in {"relative", "basename"}:
        raise ValueError("--pairing-mode must be either 'relative' or 'basename'.")

    if not low_root.exists():
        raise FileNotFoundError(f"Low directory does not exist: {low_root}")
    if not high_root.exists():
        raise FileNotFoundError(f"High directory does not exist: {high_root}")

    low_files = sorted(low_root.rglob("*.npy"))
    high_files = sorted(high_root.rglob("*.npy"))
    pairs: list[PairedMelPath] = []

    if pairing_mode == "relative":
        for low_path in low_files:
            relative_path = low_path.relative_to(low_root)
            high_path = high_root / relative_path
            if high_path.exists():
                pairs.append(PairedMelPath(relative_path=relative_path, low_path=low_path, high_path=high_path))
    else:
        duplicate_low_names = _duplicate_filenames(low_files)
        duplicate_high_names = _duplicate_filenames(high_files)
        if duplicate_low_names or duplicate_high_names:
            raise ValueError(
                "--pairing-mode basename requires unique filenames in both directories. "
                f"Duplicate low filenames: {duplicate_low_names[:5] or 'none'}. "
                f"Duplicate high filenames: {duplicate_high_names[:5] or 'none'}."
            )

        high_by_name = {path.name: path for path in high_files}
        for low_path in low_files:
            high_path = high_by_name.get(low_path.name)
            if high_path is not None:
                pairs.append(PairedMelPath(relative_path=Path(low_path.name), low_path=low_path, high_path=high_path))

    if not pairs:
        _raise_no_pairs_error(low_root, high_root, low_files, high_files, pairing_mode)

    return pairs


def compute_shared_stats(pairs: Iterable[PairedMelPath]) -> dict[str, float]:
    total_sum = 0.0
    total_sq_sum = 0.0
    total_count = 0

    for pair in pairs:
        for path in (pair.low_path, pair.high_path):
            array = load_mel(path).astype(np.float64, copy=False)
            total_sum += float(array.sum())
            total_sq_sum += float(np.square(array).sum())
            total_count += int(array.size)

    if total_count == 0:
        raise ValueError("Cannot compute statistics for an empty dataset.")

    mean = total_sum / total_count
    variance = max((total_sq_sum / total_count) - (mean * mean), 1e-12)
    std = math.sqrt(variance)
    return {"mean": float(mean), "std": float(std)}


def save_stats(stats: dict[str, float], path: Path | str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)


def load_stats(path: Path | str) -> dict[str, float]:
    with open(path, "r", encoding="utf-8") as handle:
        stats = json.load(handle)
    return {"mean": float(stats["mean"]), "std": float(stats["std"])}


def _group_key(relative_path: Path, separator: str | None) -> str:
    if separator:
        return relative_path.stem.split(separator)[0]
    if relative_path.parent != Path("."):
        return str(relative_path.parent)
    return relative_path.stem


def split_pairs(
    pairs: list[PairedMelPath],
    val_ratio: float,
    seed: int,
    group_separator: str | None = None,
) -> tuple[list[PairedMelPath], list[PairedMelPath]]:
    groups: dict[str, list[PairedMelPath]] = {}
    for pair in pairs:
        key = _group_key(pair.relative_path, group_separator)
        groups.setdefault(key, []).append(pair)

    keys = list(groups.keys())
    rng = random.Random(seed)
    rng.shuffle(keys)

    if len(keys) == 1:
        raise ValueError(
            "Only one data group was found. Provide a better --group-separator or organize files into "
            "separate parent directories so train/validation splitting can avoid leakage."
        )

    target_val_groups = max(1, int(round(len(keys) * val_ratio)))
    val_keys = set(keys[:target_val_groups])

    train_pairs = [pair for key in keys if key not in val_keys for pair in groups[key]]
    val_pairs = [pair for key in keys if key in val_keys for pair in groups[key]]

    if not train_pairs or not val_pairs:
        raise ValueError(
            "Train/validation split failed. Adjust --val-ratio or provide a better --group-separator "
            "to avoid collapsing all files into one group."
        )

    return train_pairs, val_pairs


def normalize(array: np.ndarray, stats: dict[str, float] | None) -> np.ndarray:
    if stats is None:
        return array
    std = max(stats["std"], 1e-6)
    return (array - stats["mean"]) / std


def denormalize_tensor(tensor: torch.Tensor, stats: dict[str, float] | None) -> torch.Tensor:
    if stats is None:
        return tensor
    return tensor * max(stats["std"], 1e-6) + stats["mean"]


def compute_delta_features(mel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta_time = np.zeros_like(mel, dtype=np.float32)
    delta_freq = np.zeros_like(mel, dtype=np.float32)
    delta_time[:, 1:] = mel[:, 1:] - mel[:, :-1]
    delta_freq[1:, :] = mel[1:, :] - mel[:-1, :]
    return delta_time, delta_freq


class PairedMelDataset(Dataset):
    def __init__(
        self,
        pairs: list[PairedMelPath],
        stats: dict[str, float] | None = None,
        use_deltas: bool = True,
    ) -> None:
        self.pairs = pairs
        self.stats = stats
        self.use_deltas = use_deltas

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        pair = self.pairs[index]
        low = normalize(load_mel(pair.low_path), self.stats)
        high = normalize(load_mel(pair.high_path), self.stats)

        channels = [low]
        if self.use_deltas:
            delta_time, delta_freq = compute_delta_features(low)
            channels.extend([delta_time, delta_freq])

        model_input = np.stack(channels, axis=0).astype(np.float32, copy=False)
        low_tensor = torch.from_numpy(low[None, ...])
        high_tensor = torch.from_numpy(high[None, ...])
        input_tensor = torch.from_numpy(model_input)

        return {
            "input": input_tensor,
            "low": low_tensor,
            "target": high_tensor,
            "path": str(pair.relative_path),
        }
