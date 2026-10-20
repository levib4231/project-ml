from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from datasets.triplet_dataset import VolumeTripletDataset
from datasets.volume_folder import list_evaluation_items, list_volume_items, load_volume
from ngps.config import NGPSConfig


class DatasetIsolationTests(unittest.TestCase):
    def test_training_inventory_does_not_open_or_attach_clean_references(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            noisy_dir = root / "train" / "noisy"
            clean_dir = root / "train" / "clean"
            noisy_dir.mkdir(parents=True)
            clean_dir.mkdir(parents=True)
            np.save(noisy_dir / "case.npy", np.full((3, 8, 8), 0.5, dtype=np.float32))
            (clean_dir / "case.npy").write_text("not a NumPy file")

            items = list_volume_items(root, "train")
            self.assertFalse(hasattr(items[0], "clean_path"))
            dataset = VolumeTripletDataset(items, NGPSConfig(patch_size=3, search_window=3), crop_size=4)
            sample = dataset[0]
            self.assertEqual(len(sample), 6)
            with self.assertRaisesRegex(ValueError, "evaluation-only"):
                list_evaluation_items(root, "train")

    def test_loader_rejects_out_of_range_values_instead_of_clipping(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.npy"
            np.save(path, np.array([[[1.1]]], dtype=np.float32))
            with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
                load_volume(path)

    def test_crop_is_reproducible_for_seed_epoch_and_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            noisy_dir = root / "train" / "noisy"
            noisy_dir.mkdir(parents=True)
            volume = np.arange(3 * 10 * 10, dtype=np.float32).reshape(3, 10, 10)
            volume /= volume.max()
            np.save(noisy_dir / "case.npy", volume)
            items = list_volume_items(root, "train")
            cfg = NGPSConfig(patch_size=3, search_window=3)
            first = VolumeTripletDataset(items, cfg, crop_size=4, seed=9)
            second = VolumeTripletDataset(items, cfg, crop_size=4, seed=9)
            np.testing.assert_array_equal(first[0][0].numpy(), second[0][0].numpy())
            first.set_epoch(1)
            self.assertFalse(np.array_equal(first[0][0].numpy(), second[0][0].numpy()))


if __name__ == "__main__":
    unittest.main()
