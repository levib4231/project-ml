from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from datasets.volume_folder import load_volume
from ngps.metrics import METRICS


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate denoised .npy volumes.")
    parser.add_argument("--pred", required=True)
    parser.add_argument("--clean", required=True)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    pred = load_volume(args.pred)
    clean = load_volume(args.clean, unit_range=True)
    if pred.shape != clean.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, clean={clean.shape}")

    rows = []
    for z in range(pred.shape[0]):
        rows.append(
            {
                "slice": z,
                **{name: fn(clean[z], pred[z]) for name, fn in METRICS.items()},
            }
        )
    summary = {k: float(np.mean([row[k] for row in rows])) for k in METRICS}
    print(summary)
    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        with Path(args.output_csv).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
