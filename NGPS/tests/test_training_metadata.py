from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from ngps.run_metadata import RunRecorder


class RunMetadataTests(unittest.TestCase):
    def test_run_manifest_contains_required_reproducibility_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run_manifest.json"
            recorder = RunRecorder(
                output,
                command=["train_ngps.py", "--dry-run"],
                config={"training": {"seed": 42}},
                seed=42,
                device="cpu",
                dataset_manifest_path=None,
            )
            recorder.finish("dry_run", extra={"data_order_seed": 42})
            payload = json.loads(output.read_text())
            for key in (
                "schema_version",
                "status",
                "git_commit",
                "git_dirty",
                "config",
                "seed",
                "software_versions",
                "device",
                "started_at_utc",
                "ended_at_utc",
                "wall_time_seconds",
            ):
                self.assertIn(key, payload)
            self.assertEqual(payload["status"], "dry_run")


if __name__ == "__main__":
    unittest.main()
