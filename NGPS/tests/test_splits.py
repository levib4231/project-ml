from __future__ import annotations

import unittest

from datasets.splits import assign_subject_splits, validate_split_manifest


class SplitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = {
            "schema_version": 1,
            "subjects": [{"subject_id": f"subject{index:02d}", "split": None} for index in range(10)],
        }

    def test_explicit_counts_are_deterministic_and_subject_disjoint(self) -> None:
        first = assign_subject_splits(self.manifest, seed=42, counts={"train": 6, "val": 2, "test": 2})
        second = assign_subject_splits(self.manifest, seed=42, counts={"train": 6, "val": 2, "test": 2})
        changed = assign_subject_splits(self.manifest, seed=43, counts={"train": 6, "val": 2, "test": 2})
        self.assertEqual(first, second)
        self.assertNotEqual(first["split"]["subject_ids"], changed["split"]["subject_ids"])
        self.assertEqual(first["split"]["counts"], {"train": 6, "val": 2, "test": 2})
        validate_split_manifest(first)

    def test_rejects_bad_counts_and_duplicate_subjects(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum"):
            assign_subject_splits(self.manifest, seed=1, counts={"train": 7, "val": 2, "test": 2})
        duplicate = {"subjects": [{"subject_id": "a"}, {"subject_id": "a"}]}
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            assign_subject_splits(duplicate, seed=1, counts={"train": 1, "val": 1, "test": 0})


if __name__ == "__main__":
    unittest.main()
