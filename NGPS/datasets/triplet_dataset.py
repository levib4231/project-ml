from __future__ import annotations

from numbers import Integral, Real
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ngps.config import NGPSConfig
from ngps.guide import guide_filter_volume

from .volume_folder import VolumeItem, load_volume


def _center_indices(depth: int, central_fraction: float, margin: int) -> List[int]:
    lo = margin
    hi = depth - margin
    if (isinstance(central_fraction, bool) or not isinstance(central_fraction, Real)
            or not math.isfinite(central_fraction) or not 0 < central_fraction <= 1.0):
        raise ValueError("central_fraction must be in (0, 1].")
    if hi <= lo:
        raise ValueError(f"Depth {depth} is insufficient for slice triplets; need at least {2 * margin + 1} slices.")
    if central_fraction < 1.0:
        keep = max(1, int((hi - lo) * central_fraction))
        center = (lo + hi) // 2
        start = max(lo, center - keep // 2)
        end = min(hi, start + keep)
        start = max(lo, end - keep)
        return list(range(start, end))
    return list(range(lo, hi))


def _crop(arrays: Sequence[np.ndarray], crop_size: int, rng: np.random.Generator | None = None) -> List[np.ndarray]:
    arrays = list(arrays)
    if crop_size <= 0:
        return arrays
    height, width = arrays[0].shape
    if height < crop_size or width < crop_size:
        raise ValueError(f"Spatial shape {(height, width)} is smaller than crop_size={crop_size}.")
    rng = np.random.default_rng() if rng is None else rng
    top = int(rng.integers(0, height - crop_size + 1))
    left = int(rng.integers(0, width - crop_size + 1))
    return [arr[top : top + crop_size, left : left + crop_size] for arr in arrays]


class VolumeTripletDataset(Dataset):
    """Adjacent-slice triplets for neighboring-slice self-supervision.

    Positive crop sizes produce aligned squares without padding noisy targets.
    With crop_size=0, full slices are returned and must have batch-compatible shapes.
    """

    def __init__(
        self,
        items: Sequence[VolumeItem],
        cfg: NGPSConfig,
        crop_size: int = 256,
        central_fraction: float = 1.0,
        cache_guides: bool = True,
        use_guides: bool = True,
        seed: int = 0,
    ) -> None:
        if isinstance(crop_size, bool) or not isinstance(crop_size, Integral) or crop_size < 0:
            raise ValueError("crop_size must be an integer >= 0.")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("seed must be a nonnegative integer.")
        # Validate even when no subjects are provided.
        _center_indices(3, central_fraction, margin=1)
        self.items = list(items)
        self.cfg = cfg
        self.crop_size = crop_size
        self.cache_guides = cache_guides
        self.use_guides = use_guides
        self.seed = int(seed)
        self.samples: List[Tuple[int, int]] = []
        self._raw_cache: Dict[Path, np.ndarray] = {}
        self._guide_cache: Dict[Path, np.ndarray] = {}

        for item_idx, item in enumerate(self.items):
            depth, height, width = load_volume(item.noisy_path).shape
            if depth < 3:
                raise ValueError(f"Subject {item.case_id} ({item.noisy_path}) has insufficient depth {depth}; need at least 3 slices.")
            if crop_size > 0 and min(height, width) < crop_size:
                raise ValueError(f"Subject {item.case_id} ({item.noisy_path}) spatial shape {(height, width)} is smaller than crop_size={crop_size}.")
            for z in _center_indices(depth, central_fraction=central_fraction, margin=1):
                self.samples.append((item_idx, z))
        if not self.samples:
            raise RuntimeError("No valid slice triplets were found.")

    def __len__(self) -> int:
        return len(self.samples)

    def _raw(self, item: VolumeItem) -> np.ndarray:
        if item.noisy_path not in self._raw_cache:
            self._raw_cache[item.noisy_path] = load_volume(item.noisy_path)
        return self._raw_cache[item.noisy_path]

    def _guide(self, item: VolumeItem, raw: np.ndarray) -> np.ndarray:
        if not self.cache_guides:
            return guide_filter_volume(raw, self.cfg)
        if item.noisy_path not in self._guide_cache:
            self._guide_cache[item.noisy_path] = guide_filter_volume(raw, self.cfg)
        return self._guide_cache[item.noisy_path]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        item_idx, z = self.samples[index]
        item = self.items[item_idx]
        raw = self._raw(item)
        guide = self._guide(item, raw) if self.use_guides else raw
        prev_z, next_z = z - 1, z + 1

        arrays = _crop(
            [
                raw[z],
                raw[prev_z],
                raw[next_z],
                guide[z],
                guide[prev_z],
                guide[next_z],
            ],
            self.crop_size,
            np.random.default_rng(self.seed + int(index)),
        )
        return tuple(torch.from_numpy(np.ascontiguousarray(arr)).unsqueeze(0).float() for arr in arrays)
