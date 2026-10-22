"""Small independent numerical checks for matching, objectives, and guides."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from scipy.ndimage import median_filter
import torch

from ngps.config import NGPSConfig
from ngps.guide import guide_filter_slice, guide_filter_volume
from ngps.losses import NGPSHybridLoss, NSN2NLoss
from ngps.sampler import NeighborGuidedPatchSampler


def brute_force_ssd(center, neighbor, patch_size, search_window):
    """Sum scalar differences at independently clamped image coordinates.

    Deliberately no padding, convolution, unfolding, or sampler helpers.
    """
    batch, _, height, width = center.shape
    result = torch.empty(batch, search_window ** 2, height, width, dtype=center.dtype)
    radius, search_radius = patch_size // 2, search_window // 2
    clamp_y = lambda y: max(0, min(height - 1, y))
    clamp_x = lambda x: max(0, min(width - 1, x))
    for b in range(batch):
        for y in range(height):
            for x in range(width):
                for index in range(search_window ** 2):
                    dy, dx = divmod(index, search_window)
                    dy, dx = dy - search_radius, dx - search_radius
                    total = 0.0
                    for py in range(-radius, radius + 1):
                        for px in range(-radius, radius + 1):
                            a = float(center[b, 0, clamp_y(y + py), clamp_x(x + px)])
                            z = float(neighbor[b, 0, clamp_y(y + py + dy), clamp_x(x + px + dx)])
                            total += (a - z) ** 2
                    result[b, index, y, x] = total
    return result


class SamplerOracleTests(unittest.TestCase):
    def test_all_shift_ssds_raw_topk_both_directions_and_detach(self):
        # Integer-valued doubles keep SSDs exact, including boundary ties.
        for height, width in ((2, 3), (1, 2), (2, 1), (1, 1)):
            for patch_size in (1, 3, 5):
                for topk in (1, 4, 9):
                    with self.subTest(shape=(height, width), patch=patch_size, topk=topk):
                        cfg = NGPSConfig(patch_size=patch_size, search_window=3, topk=topk)
                        sampler = NeighborGuidedPatchSampler(cfg)
                        base = torch.arange(2 * height * width, dtype=torch.float64).reshape(2, 1, height, width)
                        center = ((base * 3 + 1) % 11).requires_grad_()
                        prev = ((base * 7 + 2) % 13).requires_grad_()
                        next_ = ((base * 5 + 4) % 17).requires_grad_()
                        raw_prev = (100 + base * 19).requires_grad_()
                        raw_next = (-200 - base * 23).requires_grad_()
                        result = sampler(center, prev, next_, raw_prev, raw_next)
                        for direction, neighbor, raw in (("prev", prev, raw_prev), ("next", next_, raw_next)):
                            oracle = brute_force_ssd(center.detach(), neighbor.detach(), patch_size, 3)
                            actual_ssd = sampler._ssd_stack(center, neighbor).detach()
                            torch.testing.assert_close(actual_ssd, oracle, rtol=0, atol=0)
                            # Only borrow tie ordering, after checking EVERY candidate
                            # independently. Equal SSDs need not imply equal raw values.
                            indices = actual_ssd.topk(topk, dim=1, largest=False).indices
                            selected = oracle.gather(1, indices)
                            torch.testing.assert_close(selected.sort(dim=1).values,
                                                       oracle.sort(dim=1).values[:, :topk], rtol=0, atol=0)
                            expected_target = torch.empty_like(center)
                            for b in range(2):
                                for y in range(height):
                                    for x in range(width):
                                        values = []
                                        for index in indices[b, :, y, x].tolist():
                                            dy, dx = divmod(index, 3)
                                            yy = max(0, min(height - 1, y + dy - 1))
                                            xx = max(0, min(width - 1, x + dx - 1))
                                            values.append(float(raw.detach()[b, 0, yy, xx]))
                                        expected_target[b, 0, y, x] = sum(values) / topk
                            target = getattr(result, f"target_{direction}")
                            cost = getattr(result, f"cost_{direction}")
                            mask = getattr(result, f"mask_{direction}")
                            torch.testing.assert_close(target, expected_target, rtol=0, atol=0)
                            torch.testing.assert_close(cost, selected.mean(1, keepdim=True) / patch_size ** 2,
                                                       rtol=0, atol=0)
                            torch.testing.assert_close(mask, ((center - neighbor).abs() > cfg.mask_threshold).double())
                            torch.testing.assert_close(sampler.forward_direction(center, neighbor, raw), target)
                            for output in (target, cost, mask):
                                self.assertEqual(output.shape, center.shape)
                                self.assertFalse(output.requires_grad)
                                self.assertIsNone(output.grad_fn)


def image(values, requires_grad=False):
    return torch.tensor(values, dtype=torch.float64).reshape(1, 1, 1, -1).requires_grad_(requires_grad)


class LossArithmeticTests(unittest.TestCase):
    def test_asymmetric_masks_arithmetic_and_gradients(self):
        # Prev has one dynamic pixel, next has two: normalizing each direction
        # separately is observably different from pooling their active pixels.
        cases = (
            (True, (6, 73 / 4, 41 / 6, 83 / 3),
             [-23 / 4, -7 / 3, 1 / 4, -10 / 3], [0, -1 / 3, 1 / 6, -1 / 2], [3 / 4, 0, -3 / 4, 0]),
            (False, (4, 57 / 8, 4, 105 / 8),
             [-15 / 8, -5 / 4, 1 / 4, -15 / 8], [0, -1 / 4, 1 / 8, -3 / 8], [3 / 8, 0, -3 / 8, 0]),
        )
        for active, expected, grad_center, grad_prev, grad_next in cases:
            with self.subTest(normalize_by_active_pixels=active):
                center = image([1, 2, 3, 4], True)
                prev, next_ = image([2, 0, 4, 1], True), image([4, 3, 0, 2], True)
                loss = NGPSHybridLoss(lambda_rc=0.5, normalize_by_active_pixels=active)(
                    center, image([0, 4, 1, 8]), image([3, 0, 5, 1]),
                    image([5, 1, 7, 2]), image([0, 6, 2, 9]),
                    image([1, 0, 0, 0]), image([0, 1, 0, 1]), prev, next_)
                # Squared-error sums: static 24,8; dynamic 16,41; RC 14,18.
                n2n, dynamic, rc, total = expected
                for value, wanted in ((loss.n2n, n2n), (loss.dynamic, dynamic),
                                      (loss.recon, n2n + dynamic), (loss.rc, rc),
                                      (loss.total, total), (loss.ic, 0)):
                    self.assertAlmostEqual(value.item(), wanted, places=12)
                loss.total.backward()
                for tensor, gradient in ((center, grad_center), (prev, grad_prev), (next_, grad_next)):
                    torch.testing.assert_close(tensor.grad, image(gradient), rtol=1e-12, atol=1e-12)

    def test_nsn2n_ic_detaches_neighbor_prediction_targets(self):
        class Quadratic(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float64))
                self.outputs = []

            def forward(self, x):
                out = self.weight * x.square()
                out.retain_grad()
                self.outputs.append(out)
                return out

        model = Quadratic()
        center, prev, next_ = image([2], True), image([1], True), image([3], True)
        loss = NSN2NLoss(lambda_rc=0, lambda_ic=1)(model, center, prev, next_, image([1]), image([1]))
        # f((1+3)/2)=8, detached (f(1)+f(3))/2=10: IC=4.
        self.assertEqual(loss.ic.item(), 4)
        self.assertEqual(loss.total.item(), 4)
        loss.ic.backward()
        self.assertEqual(model.weight.grad.item(), -16)  # 2*(8-10)*2**2
        self.assertIsNone(center.grad)
        self.assertEqual(prev.grad.item(), -16)
        self.assertEqual(next_.grad.item(), -16)
        for out in model.outputs[:3]:
            self.assertIsNone(out.grad)
        self.assertEqual(model.outputs[3].grad.item(), -4)


class GuideRegressionTests(unittest.TestCase):
    def test_nonconstant_out_of_range_float32_matches_explicit_filters(self):
        cfg = NGPSConfig(bilateral_d=3, bilateral_sigma_color=90,
                         bilateral_sigma_space=2, median_kernel=3)
        yy, xx = np.mgrid[:7, :9]
        source = (-0.7 + 0.22 * xx + 0.08 * yy + 0.13 * ((xx + yy) % 3)).astype(np.float32)
        source[3, 4] = 2.5
        original = source.copy()
        bilateral = cv2.bilateralFilter(source, d=3, sigmaColor=90 / 255.0,
                                        sigmaSpace=2, borderType=cv2.BORDER_REFLECT_101)
        expected = median_filter(bilateral, size=3, mode="reflect").astype(np.float32)
        actual = guide_filter_slice(source, cfg)
        self.assertEqual(actual.dtype, np.float32)
        self.assertEqual(actual.shape, source.shape)
        self.assertLess(actual.min(), 0)
        self.assertGreater(actual.max(), 1)
        self.assertGreater(float(np.ptp(actual)), 1)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(source, original)
        volume = np.stack([source, source[::-1].copy()])
        np.testing.assert_array_equal(guide_filter_volume(volume, cfg),
                                      np.stack([expected, guide_filter_slice(volume[1], cfg)]))

    def test_invalid_guide_inputs(self):
        valid = np.zeros((3, 4), dtype=np.float32)
        invalid = [valid.astype(np.float64), valid.astype(np.uint8), valid[0],
                   valid[None], np.array(1, dtype=np.float32)]
        for value in (np.nan, np.inf, -np.inf):
            bad = valid.copy()
            bad[1, 2] = value
            invalid.append(bad)
        for bad in invalid:
            with self.subTest(shape=bad.shape, dtype=bad.dtype):
                with self.assertRaisesRegex(ValueError, "finite float32 2D"):
                    guide_filter_slice(bad)
        for bad in (valid, valid[None, None]):
            with self.assertRaisesRegex(ValueError, "volume must have shape"):
                guide_filter_volume(bad)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
