from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.volume_folder import load_volume
from ngps.metrics import METRIC_NAMES, compute_slice_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one denoised volume slice-wise, then aggregate within subject.")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--clean", required=True, help="Evaluation-only clean reference.")
    parser.add_argument("--output-csv", default=None, help="Optional per-slice intermediate metrics.")
    parser.add_argument("--output-json", default=None, help="Optional subject summary and evaluation protocol.")
    args = parser.parse_args()

    pred = load_volume(args.pred)
    clean = load_volume(args.clean)
    if pred.shape != clean.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, clean={clean.shape}")

    rows = []
    for index in range(pred.shape[0]):
        rows.append({"slice": index, **compute_slice_metrics(clean[index], pred[index])})
    summary = {name: sum(row[name] for row in rows) / len(rows) for name in METRIC_NAMES}
    payload = {
        "aggregation": "slice metrics averaged within one subject",
        "evaluation_region": "full image",
        "metric_scope": "2D slice-wise",
        "subject_metrics": summary,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output_csv:
        destination = Path(args.output_csv)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=["slice", *METRIC_NAMES])
            writer.writeheader()
            writer.writerows(rows)
    if args.output_json:
        destination = Path(args.output_json)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
