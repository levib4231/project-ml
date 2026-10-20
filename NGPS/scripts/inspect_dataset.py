from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.splits import validate_split_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize and validate a generated dataset manifest.")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    subjects = payload.get("subjects", [])
    if not subjects:
        raise ValueError("Manifest contains no subjects.")
    ids = [subject["subject_id"] for subject in subjects]
    if len(ids) != len(set(ids)):
        raise ValueError("Manifest contains duplicate subject identifiers.")
    if payload.get("split") is not None:
        validate_split_manifest(payload)
    summary = {
        "dataset_name": payload.get("dataset_name"),
        "identity_status": payload.get("identity_status"),
        "subject_count": len(subjects),
        "shapes": sorted({tuple(subject["shape"]) for subject in subjects}),
        "dtypes": sorted({subject["dtype"] for subject in subjects}),
        "subjects": ids,
        "split_counts": payload.get("split", {}).get("counts") if payload.get("split") else None,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
