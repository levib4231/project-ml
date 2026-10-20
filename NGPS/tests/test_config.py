from __future__ import annotations

import unittest

from ngps.config import NGPSConfig, TrainingConfig


class ConfigTests(unittest.TestCase):
    def test_training_positive_value_validation(self) -> None:
        for name, value in (
            ("epochs", 0),
            ("batch_size", 0),
            ("crop_size", 0),
            ("lr", 0.0),
            ("num_workers", -1),
        ):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    TrainingConfig(**{name: value})

    def test_method_dimension_validation(self) -> None:
        with self.assertRaises(ValueError):
            NGPSConfig(patch_size=0)
        with self.assertRaises(ValueError):
            NGPSConfig(search_window=-1)


if __name__ == "__main__":
    unittest.main()
