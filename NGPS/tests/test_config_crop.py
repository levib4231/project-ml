from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml

from ngps.config import NGPSConfig, TrainingConfig, load_config
from datasets.triplet_dataset import VolumeTripletDataset, _crop
from datasets.volume_folder import VolumeItem


class ConfigTests(unittest.TestCase):
    def load(self, value):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(yaml.safe_dump(value))
            return load_config(path)

    def test_unknown_keys(self):
        for value in ({"trainng": {}}, {"nsn2n": {"lambda_typo": 1}},
                      {"ngps": {"typo": 1}}, {"training": {"typo": 1}}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Unknown"):
                self.load(value)

    def test_non_mapping(self):
        for value in (None, False, 0, "text", [], [1], {"ngps": None},
                      {"training": []}, {"nsn2n": 1}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "mapping"):
                self.load(value)

    def test_invalid_training_scalars(self):
        for name in ("epochs", "batch_size", "crop_size", "num_workers", "seed"):
            for value in (1.5, True, "2", float("nan"), float("inf"), -1):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    TrainingConfig(**{name: value})
        for name in ("lr", "weight_decay"):
            for value in (float("nan"), float("inf"), -1, "1", True):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    TrainingConfig(**{name: value})
        for name in ("epochs", "batch_size", "lr"):
            with self.assertRaises(ValueError):
                TrainingConfig(**{name: 0})

    def test_invalid_method_scalars(self):
        for name in ("patch_size", "search_window", "topk", "bilateral_d", "median_kernel",
                     "nafnet_width", "nafnet_middle_blocks"):
            for value in (1.5, True, "3", float("inf"), float("nan"), -1):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    NGPSConfig(**{name: value})
        for name in ("mask_threshold", "rc_weight", "bilateral_sigma_color", "bilateral_sigma_space"):
            for value in (float("nan"), float("inf"), -1, "1", True):
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    NGPSConfig(**{name: value})
        for value in (None, 2, [1.5], [True], [-1]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.load({"ngps": {"nafnet_enc_blocks": value}})
        for value in (float("nan"), float("inf"), -1, "1", True):
            with self.assertRaises(ValueError):
                self.load({"nsn2n": {"lambda_ic": value}})

    def test_valid_defaults_and_configs(self):
        method, train, raw = self.load({})
        self.assertEqual(method, NGPSConfig())
        self.assertEqual(train, TrainingConfig())
        self.assertEqual(raw, {})
        for path in (ROOT / "configs").glob("*.yaml"):
            load_config(path)
        self.load({"nsn2n": {"lambda_ic": 0}, "training": {"crop_size": 0}})


class CropTests(unittest.TestCase):
    def test_aligned_rectangular_and_equal_crops(self):
        for shape in ((8, 12), (4, 12), (12, 4), (4, 4)):
            with self.subTest(shape=shape):
                source = np.arange(np.prod(shape)).reshape(shape)
                cropped = _crop([source, source + 1000, source + 2000], 4)
                for i, arr in enumerate(cropped):
                    self.assertEqual(arr.shape, (4, 4))
                    np.testing.assert_array_equal(arr - i * 1000, cropped[0])
                top, left = np.argwhere(source == cropped[0][0, 0])[0]
                np.testing.assert_array_equal(cropped[0], source[top:top + 4, left:left + 4])

    def make_items(self, root, shapes):
        items = []
        for index, shape in enumerate(shapes):
            path = root / f"subject{index}.npy"
            volume = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            np.save(path, volume)
            items.append(VolumeItem(path.stem, path))
        return items

    def test_mixed_sizes_batch_and_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            shapes = [(3, 4, 9), (3, 10, 4), (3, 8, 11)]
            items = self.make_items(Path(tmp), shapes)
            dataset = VolumeTripletDataset(items, NGPSConfig(), crop_size=4, use_guides=False, seed=17)
            repeat = VolumeTripletDataset(items, NGPSConfig(), crop_size=4, use_guides=False, seed=17)
            for index in range(len(dataset)):
                for left, right in zip(dataset[index], repeat[index]):
                    torch.testing.assert_close(left, right)
            batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=3)))
            for tensor in batch:
                self.assertEqual(tuple(tensor.shape), (3, 1, 4, 4))
            for i, (_, h, w) in enumerate(shapes):
                torch.testing.assert_close(batch[0][i] - batch[1][i], torch.full((1, 4, 4), float(h * w)))
                torch.testing.assert_close(batch[2][i] - batch[0][i], torch.full((1, 4, 4), float(h * w)))
            for i in range(3):
                torch.testing.assert_close(batch[i], batch[i + 3])

    def test_too_small_and_insufficient_depth_fail_at_construction(self):
        for shape, message in (((3, 3, 9), "subject1.*spatial shape.*crop_size=4"),
                               ((3, 9, 3), "subject1.*spatial shape.*crop_size=4"),
                               ((2, 9, 9), "subject1.*insufficient depth.*3 slices")):
            with tempfile.TemporaryDirectory() as tmp:
                items = self.make_items(Path(tmp), [(3, 8, 8), shape])
                with self.assertRaisesRegex(ValueError, message):
                    VolumeTripletDataset(items, NGPSConfig(), crop_size=4, use_guides=False)

    def test_full_slices_and_invalid_options(self):
        with tempfile.TemporaryDirectory() as tmp:
            items = self.make_items(Path(tmp), [(3, 4, 9)])
            dataset = VolumeTripletDataset(items, NGPSConfig(), crop_size=0, use_guides=False)
            self.assertEqual(tuple(dataset[0][0].shape), (1, 4, 9))
            for value in (0, -1, 1.1, float("nan"), float("inf"), "1", True):
                for subjects in (items, []):
                    with self.assertRaisesRegex(ValueError, "central_fraction"):
                        VolumeTripletDataset(subjects, NGPSConfig(), central_fraction=value)
            for value in (-1, 1.5, True):
                with self.assertRaisesRegex(ValueError, "crop_size"):
                    VolumeTripletDataset(items, NGPSConfig(), crop_size=value)


if __name__ == "__main__":
    unittest.main()
