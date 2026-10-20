from __future__ import annotations

import unittest

import numpy as np

from ngps.aggregation import aggregate_subject_rows
from ngps.metrics import METRIC_NAMES, compute_subject_metrics, fsim_proxy


class MetricTests(unittest.TestCase):
    def test_metric_names_do_not_claim_reference_fsim(self) -> None:
        self.assertIn("fsim_proxy", METRIC_NAMES)
        self.assertNotIn("fsim", METRIC_NAMES)

    def test_identity_is_better_than_perturbation(self) -> None:
        reference = np.tile(np.linspace(0, 1, 16, dtype=np.float32), (3, 16, 1))
        perturbed = np.clip(reference + 0.1, 0, 1)
        identical = compute_subject_metrics(reference, reference)
        changed = compute_subject_metrics(reference, perturbed)
        self.assertGreater(identical["psnr"], changed["psnr"])
        self.assertGreater(identical["ssim"], changed["ssim"])
        self.assertGreaterEqual(fsim_proxy(reference[0], reference[0]), fsim_proxy(reference[0], perturbed[0]))
        self.assertLessEqual(identical["hfen"], changed["hfen"])
        self.assertLessEqual(identical["gmsd"], changed["gmsd"])

    def test_subject_aggregation_uses_subjects_as_units(self) -> None:
        rows = [
            {"method": "m", "noise_level": "n", "subject_id": "a", "psnr": 10.0},
            {"method": "m", "noise_level": "n", "subject_id": "b", "psnr": 14.0},
        ]
        aggregates = aggregate_subject_rows(rows, metric_names=("psnr",))
        self.assertEqual(len(aggregates), 1)
        self.assertEqual(aggregates[0]["subject_count"], 2)
        self.assertEqual(aggregates[0]["psnr_mean"], 12.0)
        self.assertAlmostEqual(aggregates[0]["psnr_std"], np.sqrt(8.0))


if __name__ == "__main__":
    unittest.main()
