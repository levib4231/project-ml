"""Actual baseline objectives/filters, not labels for the NGPS objective."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from skimage.restoration import denoise_nl_means

from .losses import masked_mse


def mask_pixels(raw, generator=None, fraction=0.05):
    """Noise2Void-style random replacement; loss only on hidden coordinates.

    Replacements come from a nonzero horizontal offset in the ORIGINAL input.
    This is a masked-pixel training baseline, not an architectural blind spot.
    """
    if raw.shape[-1] < 2 or not 0 < fraction <= 1:
        raise ValueError("Masking requires width >= 2 and fraction in (0,1]")
    # CPU RNG makes the mask stream independent of accelerator RNG implementation.
    # Keep the mask stream on CPU so a CPU Generator is reproducible on both CPU
    # and CUDA; move only the finished mask to the input device for the operation.
    mask = torch.rand(raw.shape, generator=generator) < fraction
    flat = mask.reshape(raw.shape[0], -1)
    for row in flat:
        if not row.any():
            row[torch.randint(row.numel(), (1,), generator=generator)] = True
    mask = mask.to(raw.device)
    offset = int(torch.randint(1, raw.shape[-1], (1,), generator=generator))
    replaced = torch.where(mask, torch.roll(raw, shifts=offset, dims=-1), raw)
    return replaced, mask.to(device=raw.device, dtype=raw.dtype)


def blindspot_loss(model, raw, generator=None):
    corrupted, mask = mask_pixels(raw, generator)
    return masked_mse(model(corrupted), raw, mask)


def adjacent_loss(model, center, prev, next_):
    prediction = model(center)
    return 0.5 * (F.mse_loss(prediction, prev) + F.mse_loss(prediction, next_))


def nlm_volume(volume, sigma):
    """Slice-wise NLM with fixed h=0.8*sigma; no clean-reference tuning."""
    return np.stack([denoise_nl_means(
        image, h=0.8 * sigma, sigma=sigma, patch_size=5, patch_distance=6,
        fast_mode=True, preserve_range=True, channel_axis=None,
    ) for image in volume]).astype(np.float32)
