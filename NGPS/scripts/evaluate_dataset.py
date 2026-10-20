from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.volume_folder import load_volume
from ngps.aggregation import aggregate_subject_rows
from ngps.metrics import METRIC_NAMES, compute_subject_metrics


def _prediction_spec(value: str) -> tuple[str, str, Path]:
    parts = value.split(":", 2)
    if len(parts) != 3 or not all(parts):
        raise argparse.ArgumentTypeError("Prediction must be METHOD:NOISE_LEVEL:DIRECTORY")
    return parts[0], parts[1], Path(parts[2])


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    preferred = ["row_type", "method", "noise_level", "subject_id", "subject_count", *METRIC_NAMES]
    extras = sorted({key for row in rows for key in row} - set(preferred))
    fieldnames = [name for name in preferred if any(name in row for row in rows)] + extras
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate predictions by subject, then aggregate across subjects.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", default=None, help="Override manifest path_root for clean references.")
    parser.add_argument("--split", default="test", choices=("val", "test"))
    parser.add_argument("--prediction", action="append", type=_prediction_spec, required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_root = Path(args.data_root) if args.data_root else Path(manifest["path_root"])
    subjects = [subject for subject in manifest["subjects"] if subject.get("split") == args.split]
    if not subjects:
        raise ValueError(f"Manifest contains no subjects assigned to {args.split}.")

    subject_rows: list[dict[str, Any]] = []
    for method, noise_level, prediction_dir in args.prediction:
        for subject in subjects:
            subject_id = str(subject["subject_id"])
            clean = load_volume(data_root / subject["volume_path"])
            prediction = load_volume(prediction_dir / f"{subject_id}.npy")
            metrics = compute_subject_metrics(clean, prediction)
            subject_rows.append(
                {
                    "row_type": "subject",
                    "method": method,
                    "noise_level": noise_level,
                    "subject_id": subject_id,
                    **metrics,
                }
            )
    aggregate_rows = aggregate_subject_rows(subject_rows)
    rows = [*subject_rows, *aggregate_rows]
    _write_csv(Path(args.output_csv), rows)
    payload = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "split": args.split,
        "aggregation": "slice metrics within subject, then subject means across subjects",
        "uncertainty": "two-sided 95% Student-t confidence interval across subject means",
        "evaluation_region": "full image",
        "metric_scope": "2D slice-wise",
        "metric_warning": "fsim_proxy is a project-local gradient proxy, not validated reference FSIM",
        "subject_rows": subject_rows,
        "aggregate_rows": aggregate_rows,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(aggregate_rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
