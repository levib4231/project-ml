from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


@dataclass(frozen=True)
class RicianNoiseConfig:
    """Explicit parameters for magnitude-MRI Rician corruption."""

    sigma: float
    sigma_reference: str = "absolute"
    data_min: float = 0.0
    data_max: float = 1.0
    clip_output: bool = False

    def __post_init__(self) -> None:
        if not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError("sigma must be a finite positive number.")
        if self.sigma_reference not in {"absolute", "data_range_fraction"}:
            raise ValueError("sigma_reference must be 'absolute' or 'data_range_fraction'.")
        if not np.isfinite(self.data_min) or not np.isfinite(self.data_max) or self.data_max <= self.data_min:
            raise ValueError("data_max must be greater than data_min.")

    @property
    def effective_sigma(self) -> float:
        if self.sigma_reference == "absolute":
            return float(self.sigma)
        return float(self.sigma * (self.data_max - self.data_min))


def derive_subject_seed(base_seed: int, subject_id: str, noise_label: str) -> int:
    payload = f"{int(base_seed)}:{subject_id}:{noise_label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], byteorder="big", signed=False)


def simulate_rician(clean: np.ndarray, config: RicianNoiseConfig, *, seed: int) -> np.ndarray:
    """Return ``sqrt((x+n_real)^2 + n_imag^2)`` for independent Gaussian noise."""

    signal = np.asarray(clean)
    if not np.issubdtype(signal.dtype, np.number) or not np.isfinite(signal).all():
        raise ValueError("clean input must contain finite numeric values.")
    if np.any(signal < 0):
        raise ValueError("Rician magnitude simulation requires a non-negative clean signal.")
    rng = np.random.default_rng(seed)
    sigma = config.effective_sigma
    real_noise = rng.normal(0.0, sigma, size=signal.shape)
    imaginary_noise = rng.normal(0.0, sigma, size=signal.shape)
    noisy = np.sqrt((signal.astype(np.float64) + real_noise) ** 2 + imaginary_noise**2)
    if config.clip_output:
        noisy = np.clip(noisy, config.data_min, config.data_max)
    return noisy.astype(np.float32)
