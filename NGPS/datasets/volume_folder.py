from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass(frozen=True)
class VolumeItem:
    case_id: str
    noisy_path: Path


@dataclass(frozen=True)
class EvaluationVolumeItem:
    case_id: str
    noisy_path: Path
    clean_path: Path


def list_volume_items(data_root: str | Path, split: str) -> List[VolumeItem]:
    """List only noisy volumes from ``DATA_ROOT/split/noisy/*.npy``."""

    root = Path(data_root)
    noisy_dir = root / split / "noisy"
    if not noisy_dir.exists():
        raise FileNotFoundError(f"No noisy directory found: {noisy_dir}")

    items = []
    for noisy_path in sorted(noisy_dir.glob("*.npy")):
        load_volume(noisy_path)
        items.append(VolumeItem(noisy_path.stem, noisy_path))
    if not items:
        raise RuntimeError(f"No .npy volumes found in {noisy_dir}")
    return items


def list_evaluation_items(data_root: str | Path, split: str) -> List[EvaluationVolumeItem]:
    """Explicitly attach and validate clean references for evaluation only."""

    noisy_items = list_volume_items(data_root, split)
    clean_dir = Path(data_root) / split / "clean"
    evaluation_items: List[EvaluationVolumeItem] = []
    for item in noisy_items:
        clean_path = clean_dir / item.noisy_path.name
        if not clean_path.is_file():
            raise FileNotFoundError(f"Missing evaluation-only clean reference: {clean_path}")
        try:
            clean = load_volume(clean_path)
        except Exception as exc:
            raise ValueError(f"Invalid evaluation-only clean reference: {clean_path}") from exc
        noisy = load_volume(item.noisy_path)
        if clean.shape != noisy.shape:
            raise ValueError(f"Evaluation shape mismatch for {item.case_id}: noisy={noisy.shape}, clean={clean.shape}")
        evaluation_items.append(EvaluationVolumeItem(item.case_id, item.noisy_path, clean_path))
    return evaluation_items


def load_volume(path: str | Path) -> np.ndarray:
    """Load a normalized volume, rejecting invalid data instead of clipping it."""

    volume = np.load(path, allow_pickle=False)
    if volume.ndim != 3:
        raise ValueError(f"Expected volume shaped (D, H, W), got {volume.shape}: {path}")
    if not np.issubdtype(volume.dtype, np.floating):
        raise ValueError(f"Expected floating-point volume, got {volume.dtype}: {path}")
    if not np.isfinite(volume).all():
        raise ValueError(f"Volume contains non-finite values: {path}")
    minimum = float(volume.min())
    maximum = float(volume.max())
    if minimum < 0.0 or maximum > 1.0:
        raise ValueError(f"Expected values in [0, 1], got [{minimum}, {maximum}]: {path}")
    return volume.astype(np.float32, copy=False)
