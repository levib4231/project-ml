from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from datasets.manifest import build_dataset_manifest, write_json


class ManifestTests(unittest.TestCase):
    def _subject(self, root: Path, subject_id: str, shape: tuple[int, ...] = (3, 4, 5)) -> Path:
        volume = np.linspace(0.0, 1.0, int(np.prod(shape)), dtype=np.float32).reshape(shape)
        volume_path = root / f"subject{subject_id}_clean.npy"
        np.save(volume_path, volume)
        (root / f"subject{subject_id}_t1w_p4.mnc.gz").write_bytes(b"fixture source")
        return volume_path

    def test_manifest_is_deterministic_and_records_required_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._subject(root, "01")
            self._subject(root, "02")
            manifest = build_dataset_manifest(
                root,
                dataset_name="fixture-unverified",
                preprocessing_version="test-v1",
                provenance_note="fixture only",
            )
            first = root / "first.json"
            second = root / "second.json"
            write_json(first, manifest)
            write_json(second, manifest)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            subject = json.loads(first.read_text())["subjects"][0]
            self.assertEqual(subject["shape"], [3, 4, 5])
            self.assertEqual(subject["dtype"], "float32")
            self.assertEqual(len(subject["sha256"]), 64)
            self.assertEqual(len(subject["source_sha256"]), 64)
            self.assertIn("foreground", subject)
            self.assertIsNone(subject["split"])

    def test_rejects_shape_mismatch_duplicate_ids_and_missing_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._subject(root, "01")
            self._subject(root, "02", shape=(4, 4, 5))
            with self.assertRaisesRegex(ValueError, "shape mismatch"):
                build_dataset_manifest(root, dataset_name="fixture")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "a"
            second = root / "b"
            first.mkdir()
            second.mkdir()
            self._subject(first, "01")
            self._subject(second, "01")
            with self.assertRaisesRegex(ValueError, "Duplicate subject"):
                build_dataset_manifest(root, dataset_name="fixture", recursive=True)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.save(root / "subject01_clean.npy", np.zeros((3, 4, 5), dtype=np.float32))
            with self.assertRaisesRegex(FileNotFoundError, "source"):
                build_dataset_manifest(root, dataset_name="fixture")


if __name__ == "__main__":
    unittest.main()
