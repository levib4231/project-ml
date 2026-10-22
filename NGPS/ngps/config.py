from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml


def _integer(name: str, value: Any, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")


def _nonnegative(name: str, value: Any, *, positive: bool = False) -> None:
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        bound = "> 0" if positive else ">= 0"
        raise ValueError(f"{name} must be a finite number {bound}.")


def _mapping(value: Any, name: str, allowed: set[str]) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown {name} keys: {sorted(map(str, unknown))}")


@dataclass(frozen=True)
class NGPSConfig:
    """Hyperparameters used by guide generation, matching, and the backbone."""

    patch_size: int = 7
    search_window: int = 15
    topk: int = 4
    mask_threshold: float = 0.05
    bilateral_d: int = 5
    bilateral_sigma_color: float = 35.0
    bilateral_sigma_space: float = 50.0
    median_kernel: int = 5
    rc_weight: float = 0.5
    nafnet_width: int = 32
    nafnet_enc_blocks: Tuple[int, ...] = (2, 2, 4, 8)
    nafnet_middle_blocks: int = 8
    nafnet_dec_blocks: Tuple[int, ...] = (2, 2, 2, 2)

    def __post_init__(self) -> None:
        for name in ("patch_size", "search_window", "topk", "bilateral_d", "median_kernel", "nafnet_width"):
            _integer(name, getattr(self, name), 1)
        _integer("nafnet_middle_blocks", self.nafnet_middle_blocks, 0)
        for name in ("nafnet_enc_blocks", "nafnet_dec_blocks"):
            blocks = getattr(self, name)
            if not isinstance(blocks, (list, tuple)) or not blocks:
                raise ValueError(f"{name} must be a nonempty sequence of block counts.")
            for count in blocks:
                _integer(name, count, 0)
        if len(self.nafnet_enc_blocks) != len(self.nafnet_dec_blocks):
            raise ValueError("nafnet_enc_blocks and nafnet_dec_blocks must have equal lengths.")
        for name in ("mask_threshold", "rc_weight"):
            _nonnegative(name, getattr(self, name))
        for name in ("bilateral_sigma_color", "bilateral_sigma_space"):
            _nonnegative(name, getattr(self, name), positive=True)
        if self.patch_size < 1 or self.patch_size % 2 != 1:
            raise ValueError("patch_size must be odd.")
        if self.search_window < 1 or self.search_window % 2 != 1:
            raise ValueError("search_window must be odd.")
        if self.topk < 1 or self.topk > self.search_window * self.search_window:
            raise ValueError("topk must be in [1, search_window ** 2].")
        if self.median_kernel < 1 or self.median_kernel % 2 != 1:
            raise ValueError("median_kernel must be odd.")


@dataclass(frozen=True)
class TrainingConfig:
    """Common training options; crop_size=0 requires batch-compatible slice shapes."""

    epochs: int = 10
    batch_size: int = 4
    crop_size: int = 256
    lr: float = 2e-4
    weight_decay: float = 1e-5
    num_workers: int = 0
    seed: int = 42

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size"):
            _integer(name, getattr(self, name), 1)
        for name in ("crop_size", "num_workers", "seed"):
            _integer(name, getattr(self, name), 0)
        _nonnegative("lr", self.lr, positive=True)
        _nonnegative("weight_decay", self.weight_decay)


def _update_dataclass(cls: type, values: Dict[str, Any]):
    names = {f.name for f in fields(cls)}
    _mapping(values, cls.__name__, names)
    selected = dict(values)
    for name in ("nafnet_enc_blocks", "nafnet_dec_blocks"):
        if name in selected:
            if not isinstance(selected[name], (list, tuple)):
                raise ValueError(f"{name} must be a sequence of block counts.")
            selected[name] = tuple(selected[name])
    return cls(**selected)


def load_config(path: str | Path) -> tuple[NGPSConfig, TrainingConfig, Dict[str, Any]]:
    """Load a YAML config and return typed method/training configs plus raw dict."""

    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    _mapping(raw, "config", {"ngps", "training", "nsn2n"})
    nsn2n = raw.get("nsn2n", {})
    _mapping(nsn2n, "nsn2n", {"lambda_ic"})
    if "lambda_ic" in nsn2n:
        _nonnegative("lambda_ic", nsn2n["lambda_ic"])
    method = _update_dataclass(NGPSConfig, raw.get("ngps", {}))
    train = _update_dataclass(TrainingConfig, raw.get("training", {}))
    return method, train, raw
