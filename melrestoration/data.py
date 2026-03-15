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


def collect_paired_files(low_dir: Path | str, high_dir: Path | str) -> list[PairedMelPath]:
    low_root = Path(low_dir)
    high_root = Path(high_dir)
    pairs: list[PairedMelPath] = []

    for low_path in sorted(low_root.rglob("*.npy")):
        relative_path = low_path.relative_to(low_root)
        high_path = high_root / relative_path
        if high_path.exists():
            pairs.append(PairedMelPath(relative_path=relative_path, low_path=low_path, high_path=high_path))

    if not pairs:
        raise FileNotFoundError(
            f"No paired .npy files found between {low_root} and {high_root}. "
            "Expected matching relative paths in both directories."
        )

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
