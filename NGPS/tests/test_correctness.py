from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from datasets.manifest import assign_splits, load_manifest, manifest_hash, stable_seed, validate_manifest
from datasets.preparation import gaussian_noise, normalize, prepare
from datasets.volume_folder import validate_volume
from ngps.baselines import adjacent_loss, blindspot_loss, mask_pixels, nlm_volume
from ngps.config import NGPSConfig
from ngps.guide import guide_filter_volume, make_misalignment_mask
from ngps.evaluation import evaluate
from ngps.inference import infer_volume, load_model
from ngps.losses import NGPSHybridLoss, NSN2NLoss, masked_mse
from ngps.metrics import METRICS, aggregate_subjects, gradient_magnitude_similarity, volume_metrics
from ngps.nafnet import build_nafnet
from ngps.sampler import NeighborGuidedPatchSampler
from ngps.training import mask_diagnostics, train
from scripts.synthetic_pipeline import make_sources, run


torch.set_num_threads(1)


class DataTests(unittest.TestCase):
    def test_deterministic_split_and_seeds(self):
        ids = [f"subject{i}" for i in range(10)]
        splits = assign_splits(ids)
        self.assertEqual(splits, assign_splits(reversed(ids)))
        self.assertEqual([list(splits.values()).count(s) for s in ("train", "val", "test")], [6, 2, 2])
        self.assertNotEqual(stable_seed(42, "a", "noise:0.03"), stable_seed(42, "a", "noise:0.06"))
        self.assertNotEqual(stable_seed(42, "a", "noise:0.03"), stable_seed(42, "b", "noise:0.03"))

    def test_versioned_source_manifest_metadata(self):
        manifest = load_manifest(ROOT / "configs/subjects_v1.json")
        self.assertEqual({r["split"] for r in manifest["subjects"]}, {"train", "val", "test"})
        self.assertTrue(all(len(r["source_sha256"]) == 64 for r in manifest["subjects"]))
        self.assertEqual([sum(r["split"] == split for r in manifest["subjects"])
                          for split in ("train", "val", "test")], [6, 2, 2])

    def test_manifest_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = make_sources(Path(tmp) / "sources")
            original = load_manifest(path)
            self.assertEqual(manifest_hash(original), manifest_hash(copy.deepcopy(original)))
            for field in ("subject_id", "source"):
                bad = copy.deepcopy(original)
                bad["subjects"][1][field] = bad["subjects"][0][field]
                with self.assertRaises(ValueError):
                    validate_manifest(bad, path)
            bad = copy.deepcopy(original)
            bad["subjects"][0]["split"] = "test" if bad["subjects"][0]["split"] != "test" else "train"
            with self.assertRaises(ValueError):
                validate_manifest(bad, path)
            link = path.parent / "alias.npy"
            link.symlink_to(path.parent / original["subjects"][0]["source"])
            bad = copy.deepcopy(original)
            bad["subjects"][1]["source"] = "alias.npy"
            with self.assertRaises(ValueError):
                validate_manifest(bad, path)
            link.unlink()
            link.hardlink_to(path.parent / original["subjects"][0]["source"])
            with self.assertRaises(ValueError):
                validate_manifest(bad, path)

    def test_data_validation_and_normalization(self):
        clean = np.zeros((3, 8, 8), dtype=np.float32)
        for bad in (clean.astype(np.float64), clean[0], clean[:, :0], clean + np.nan, clean + np.inf):
            with self.assertRaises(ValueError):
                validate_volume(bad)
        with self.assertRaises(ValueError):
            validate_volume(clean - 0.1, unit_range=True)
        np.testing.assert_array_equal(validate_volume(clean - 0.1), clean - 0.1)
        np.testing.assert_array_equal(normalize(clean + 50, "fixed-range", 0, 100), clean + 0.5)
        for policy, lower, upper in [("already-unit", None, None), ("fixed-range", 0, 10), ("fixed-range", 2, 2)]:
            with self.assertRaises(ValueError):
                normalize(clean + 50, policy, lower, upper)

    def test_noise_is_deterministic_unclipped(self):
        clean = np.zeros((10, 20, 20), dtype=np.float32)
        for sigma in (0.03, 0.06, 0.10):
            a = gaussian_noise(clean, sigma, 10)
            np.testing.assert_array_equal(a, gaussian_noise(clean, sigma, 10))
            self.assertFalse(np.array_equal(a, gaussian_noise(clean, sigma, 11)))
            self.assertEqual(a.dtype, np.float32)
            self.assertLess(a.min(), 0)
            self.assertAlmostEqual(float(a.std()), sigma, delta=sigma * 0.05)


class MethodTests(unittest.TestCase):
    def setUp(self):
        self.cfg = NGPSConfig(patch_size=3, search_window=3, topk=2, median_kernel=3,
                              nafnet_width=4, nafnet_enc_blocks=(1,), nafnet_dec_blocks=(1,), nafnet_middle_blocks=1)

    def test_guide_mask_and_diagnostics(self):
        image = np.full((3, 8, 8), -0.2, dtype=np.float32)
        guide = guide_filter_volume(image, self.cfg)
        np.testing.assert_allclose(guide, image, atol=1e-6)
        zero = torch.zeros(1, 1, 8, 8)
        mask = make_misalignment_mask(zero, zero + 0.1, 0.05)
        self.assertTrue(torch.all(mask == 1))
        self.assertTrue(torch.all(make_misalignment_mask(zero, zero + 0.05, 0.05) == 0))
        stats = mask_diagnostics(mask, zero)
        self.assertEqual(stats["dynamic_active_fraction"], 0.5)
        self.assertEqual(stats["dynamic_empty_fraction"], 0.5)
        self.assertEqual(stats["static_empty_fraction"], 0.5)

    def test_sampler_shapes_bounds_and_boundary_cost(self):
        sampler = NeighborGuidedPatchSampler(self.cfg)
        for shape in [(2, 1, 8, 9), (1, 1, 1, 2)]:
            center = torch.zeros(shape)
            neighbor = torch.ones(shape)
            target, cost = sampler.forward_direction(center, neighbor, neighbor * 3, True)
            self.assertEqual(target.shape, shape)
            torch.testing.assert_close(target, neighbor * 3)
            # No artificially cheap zero-padded border patches.
            torch.testing.assert_close(cost, neighbor)
            raw = torch.rand(shape)
            result = sampler.forward_direction(center, neighbor, raw)
            self.assertGreaterEqual(float(result.min()), float(raw.min()))
            self.assertLessEqual(float(result.max()), float(raw.max()))
        with self.assertRaises(ValueError):
            sampler.forward_direction(center, neighbor, torch.ones(2, 1, 3, 3))

    def test_global_residual_and_odd_shape(self):
        model = build_nafnet(self.cfg)
        torch.nn.init.zeros_(model.ending.weight)
        torch.nn.init.zeros_(model.ending.bias)
        for shape in [(1, 1, 7, 9), (1, 1, 1, 1)]:
            x = torch.rand(shape)
            torch.testing.assert_close(model(x), x)

    def test_losses_empty_and_active(self):
        pred = torch.ones(1, 1, 4, 4, requires_grad=True)
        zero, one = torch.zeros_like(pred), torch.ones_like(pred)
        self.assertEqual(float(masked_mse(pred, zero, zero).detach()), 0)
        self.assertEqual(float(masked_mse(pred, zero, one).detach()), 1)
        for mask in (zero, one):
            loss = NGPSHybridLoss()(pred, zero, zero, zero, zero, mask, mask, pred, pred)
            self.assertEqual(float(loss.total.detach()), 1)
            loss.total.backward(retain_graph=True)
        self.assertTrue(torch.isfinite(pred.grad).all())
        model = torch.nn.Conv2d(1, 1, 1)
        loss = NSN2NLoss()(model, pred.detach(), zero, zero, one, one).total
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_real_baselines(self):
        raw = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8) / 64
        corrupted, mask = mask_pixels(raw, torch.Generator().manual_seed(5), fraction=1)
        self.assertTrue(torch.all(corrupted != raw))
        self.assertTrue(torch.all(mask == 1))
        model = torch.nn.Conv2d(1, 1, 1)
        loss = blindspot_loss(model, raw, torch.Generator().manual_seed(7))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        identity = torch.nn.Identity()
        self.assertAlmostEqual(float(adjacent_loss(identity, raw, raw + 1, raw + 1)), 1)
        result = nlm_volume(np.random.default_rng(4).random((3, 8, 8), dtype=np.float32), 0.1)
        self.assertEqual(result.shape, (3, 8, 8))
        self.assertTrue(np.isfinite(result).all())

    def test_subject_aggregation_and_metrics(self):
        rows = [{"subject_id": "a", "slices": 1, **dict.fromkeys(METRICS, 1.0)},
                {"subject_id": "b", "slices": 100, **dict.fromkeys(METRICS, 3.0)}]
        result = aggregate_subjects(rows)
        self.assertEqual(result["PSNR"]["mean"], 2)
        self.assertEqual(result["PSNR"]["std_population"], 1)
        with self.assertRaises(ValueError):
            aggregate_subjects([rows[0], rows[0]])
        clean = np.random.default_rng(5).random((3, 8, 8), dtype=np.float32)
        scores = volume_metrics(clean, clean + np.float32(0.1))
        self.assertAlmostEqual(scores["PSNR"], 20, places=4)
        self.assertNotIn("FSIM", scores)
        self.assertEqual(gradient_magnitude_similarity(np.zeros((8, 8)), np.zeros((8, 8))), 1)
        with self.assertRaises(ValueError):
            volume_metrics(clean, clean[:1])
        with self.assertRaises(ValueError):
            masked_mse(torch.zeros(1, 1, 2, 2), torch.zeros(1, 2, 2, 2), torch.ones(1, 1, 2, 2))


class PipelineTests(unittest.TestCase):
    def test_synthetic_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run(tmp)
            self.assertEqual(len(report["subjects"]), 2)
            self.assertEqual(report["aggregate_by_sigma"]["0.06"]["PSNR"]["n_subjects"], 2)
            self.assertTrue(all(r["inference_seconds"] > 0 for r in report["subjects"]))
            self.assertTrue((Path(tmp) / "model_metrics_aggregate.csv").exists())

    def test_all_objectives_never_open_clean_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = make_sources(root / "sources")
            manifest = prepare(source, root / "prepared", "already-unit")
            original_load = np.load
            def noisy_only(path, *args, **kwargs):
                self.assertTrue(Path(path).name.startswith("noisy_"), f"Training opened non-noisy file: {path}")
                return original_load(path, *args, **kwargs)
            adjacent_model = None
            for method in ("ngps", "nsn2n", "blindspot", "adjacent"):
                with patch("numpy.load", side_effect=noisy_only):
                    ckpt = train(manifest, ROOT / "configs/synthetic_cpu.yaml", root / method, method=method)
                model, metadata = load_model(ckpt, "cpu")
                if method == "adjacent":
                    adjacent_model = model
                self.assertEqual(metadata["epoch"], 1)
                self.assertEqual(metadata["selection"]["rule"], "fixed_final_epoch")
                self.assertEqual(len(metadata["validation_history"]), 1)
                for key in ("commit", "device", "seed", "manifest_hash", "config"):
                    self.assertIn(key, metadata)
                pred = infer_volume(model, np.zeros((3, 8, 8), dtype=np.float32), "cpu")
                self.assertEqual(pred.shape, (3, 8, 8))
                json.loads((root / method / "metadata.json").read_text())
            # Same seed/config/data produces identical parameters on CPU.
            repeated = train(manifest, ROOT / "configs/synthetic_cpu.yaml", root / "repeat", method="adjacent")
            repeated_model, _ = load_model(repeated, "cpu")
            self.assertIsNotNone(adjacent_model)
            for a, b in zip(adjacent_model.parameters(), repeated_model.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            with self.assertRaises(ValueError):
                evaluate(manifest, root / "wrong_sigma", "model", repeated, 0.03)
            altered = load_manifest(manifest)
            altered["normalization"]["audit_note"] = "changed protocol"
            altered_path = manifest.parent / "altered.json"
            altered_path.write_text(json.dumps(altered))
            with self.assertRaises(ValueError):
                evaluate(altered_path, root / "wrong_manifest", "model", repeated, 0.06)
            # Hash checks fail even when a tampered noisy volume remains valid float32.
            row = next(r for r in altered["subjects"] if r["split"] == "train")
            noisy_path = manifest.parent / row["noisy"]["0.06"]["path"]
            np.save(noisy_path, np.zeros(row["shape"], dtype=np.float32))
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                train(manifest, ROOT / "configs/synthetic_cpu.yaml", root / "tampered")


if __name__ == "__main__":
    unittest.main()
