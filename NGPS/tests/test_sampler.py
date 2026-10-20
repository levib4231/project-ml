from __future__ import annotations

import unittest

import torch

from ngps.config import NGPSConfig
from ngps.sampler import NeighborGuidedPatchSampler


class SamplerTests(unittest.TestCase):
    def test_patch_cost_uses_consistent_boundary_padding(self) -> None:
        cfg = NGPSConfig(patch_size=3, search_window=1, topk=1)
        sampler = NeighborGuidedPatchSampler(cfg)
        center = torch.zeros(1, 1, 5, 5)
        neighbor = torch.ones(1, 1, 5, 5)
        costs = sampler._ssd_stack(center, neighbor)
        torch.testing.assert_close(costs, torch.full_like(costs, 9.0))


if __name__ == "__main__":
    unittest.main()
