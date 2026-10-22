from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import NGPSConfig
from .guide import make_misalignment_mask


@dataclass
class NGPSTargets:
    target_prev: torch.Tensor
    target_next: torch.Tensor
    mask_prev: torch.Tensor
    mask_next: torch.Tensor
    cost_prev: torch.Tensor
    cost_next: torch.Tensor


class NeighborGuidedPatchSampler(nn.Module):
    """Dense local Top-K retrieval for Neighbor-Guided Patch Sampling.

    For every pixel p in the center slice, the sampler compares the guide patch
    around p to guide patches in a local window of the neighboring slice. The
    Top-K matched coordinates are then used to retrieve raw center-pixel values
    from that neighbor. This keeps structural matching and raw noisy target
    retrieval decoupled.
    """

    def __init__(self, cfg: NGPSConfig = NGPSConfig()) -> None:
        super().__init__()
        self.cfg = cfg
        self.pad = cfg.search_window // 2
        self.num_shifts = cfg.search_window * cfg.search_window
        kernel = torch.ones((self.num_shifts, 1, cfg.patch_size, cfg.patch_size), dtype=torch.float32)
        self.register_buffer("ssd_kernel", kernel, persistent=False)

    @staticmethod
    def _validate_image(x: torch.Tensor, name: str) -> None:
        if (not isinstance(x, torch.Tensor) or x.ndim != 4 or x.shape[1] != 1
                or min(x.shape[0], x.shape[2], x.shape[3]) < 1):
            raise ValueError(f"{name} must have shape (B, 1, H, W).")
        if not torch.isfinite(x).all():
            raise ValueError(f"{name} contains non-finite values.")

    def _shift_stack(self, image: torch.Tensor) -> torch.Tensor:
        """Stack all local-window shifts along the channel dimension."""

        self._validate_image(image, "image")
        _, _, height, width = image.shape
        padded = F.pad(image, (self.pad, self.pad, self.pad, self.pad), mode="replicate")
        shifts = []
        for dy in range(self.cfg.search_window):
            for dx in range(self.cfg.search_window):
                shifts.append(padded[:, :, dy : dy + height, dx : dx + width])
        return torch.cat(shifts, dim=1)

    def _ssd_stack(self, guide_center: torch.Tensor, guide_neighbor: torch.Tensor) -> torch.Tensor:
        # Extend BOTH images before shifting and computing differences. Padding
        # an already-computed SSD map with zeros artificially rewards boundaries.
        radius = self.cfg.patch_size // 2
        center = F.pad(guide_center, (radius,) * 4, mode="replicate")
        neighbor = F.pad(guide_neighbor, (radius,) * 4, mode="replicate")
        diff_sq = (center - self._shift_stack(neighbor)).pow(2)
        kernel = self.ssd_kernel.to(device=diff_sq.device, dtype=diff_sq.dtype)
        return F.conv2d(diff_sq, kernel, groups=self.num_shifts)

    @torch.no_grad()
    def forward_direction(
        self,
        guide_center: torch.Tensor,
        guide_neighbor: torch.Tensor,
        raw_neighbor: torch.Tensor,
        return_cost: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Return the NGPS target for one neighbor direction.

        If ``return_cost`` is true, the second return value is the Top-K mean SSD
        cost divided by patch area. This is useful for reliability diagnostics
        or calibrated match-cost gating.
        """

        for name, image in [("center", guide_center), ("neighbor", guide_neighbor), ("raw", raw_neighbor)]:
            self._validate_image(image, name)
            if image.shape != guide_center.shape:
                raise ValueError("Sampler inputs must have matching shapes")
            if image.device != guide_center.device or image.dtype != guide_center.dtype:
                raise ValueError("Sampler inputs must share device and dtype")
        ssd = self._ssd_stack(guide_center, guide_neighbor)
        top_values, top_indices = torch.topk(ssd, k=self.cfg.topk, dim=1, largest=False)
        shifted_raw = self._shift_stack(raw_neighbor)
        top_raw = torch.gather(shifted_raw, dim=1, index=top_indices)
        target = top_raw.mean(dim=1, keepdim=True).detach()
        if not return_cost:
            return target
        cost = top_values.mean(dim=1, keepdim=True) / float(self.cfg.patch_size**2)
        return target, cost.detach()

    @torch.no_grad()
    def forward(
        self,
        guide_center: torch.Tensor,
        guide_prev: torch.Tensor,
        guide_next: torch.Tensor,
        raw_prev: torch.Tensor,
        raw_next: torch.Tensor,
    ) -> NGPSTargets:
        target_prev, cost_prev = self.forward_direction(guide_center, guide_prev, raw_prev, return_cost=True)
        target_next, cost_next = self.forward_direction(guide_center, guide_next, raw_next, return_cost=True)
        mask_prev = make_misalignment_mask(guide_center, guide_prev, self.cfg.mask_threshold)
        mask_next = make_misalignment_mask(guide_center, guide_next, self.cfg.mask_threshold)
        return NGPSTargets(target_prev, target_next, mask_prev.detach(), mask_next.detach(), cost_prev, cost_next)
