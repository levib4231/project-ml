from __future__ import annotations

import unittest

import numpy as np

from datasets.noise import RicianNoiseConfig, derive_subject_seed, simulate_rician


class RicianNoiseTests(unittest.TestCase):
    def test_fixed_seed_is_deterministic_and_different_seed_changes_output(self) -> None:
        clean = np.full((5, 8, 8), 0.4, dtype=np.float32)
        cfg = RicianNoiseConfig(sigma=0.05, sigma_reference="data_range_fraction")
        first = simulate_rician(clean, cfg, seed=123)
        second = simulate_rician(clean, cfg, seed=123)
        changed = simulate_rician(clean, cfg, seed=124)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, changed))
        self.assertEqual(first.shape, clean.shape)
        self.assertEqual(first.dtype, np.float32)

    def test_zero_signal_matches_rayleigh_second_moment(self) -> None:
        clean = np.zeros((300_000,), dtype=np.float32)
        sigma = 0.1
        noisy = simulate_rician(clean, RicianNoiseConfig(sigma=sigma), seed=7)
        self.assertAlmostEqual(float(np.mean(noisy**2)), 2.0 * sigma**2, delta=3e-4)

    def test_convention_and_clipping_are_explicit(self) -> None:
        clean = np.ones((1000,), dtype=np.float32)
        fractional = RicianNoiseConfig(
            sigma=0.1,
            sigma_reference="data_range_fraction",
            data_min=-1.0,
            data_max=1.0,
            clip_output=True,
        )
        self.assertEqual(fractional.effective_sigma, 0.2)
        noisy = simulate_rician(clean, fractional, seed=1)
        self.assertLessEqual(float(noisy.max()), 1.0)
        self.assertNotEqual(derive_subject_seed(42, "subject01", "0.05"), derive_subject_seed(42, "subject02", "0.05"))

    def test_rejects_invalid_parameters_and_input(self) -> None:
        with self.assertRaises(ValueError):
            RicianNoiseConfig(sigma=0.0)
        with self.assertRaises(ValueError):
            RicianNoiseConfig(sigma=0.1, sigma_reference="percent_of_mean")
        with self.assertRaisesRegex(ValueError, "non-negative"):
            simulate_rician(np.array([-0.1], dtype=np.float32), RicianNoiseConfig(sigma=0.1), seed=1)


if __name__ == "__main__":
    unittest.main()
