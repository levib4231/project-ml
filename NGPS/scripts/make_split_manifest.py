from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.manifest import write_json
from datasets.splits import assign_subject_splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Assign deterministic, subject-disjoint dataset splits.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--counts", type=int, nargs=3, metavar=("TRAIN", "VAL", "TEST"))
    group.add_argument("--ratios", type=float, nargs=3, metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    names = ("train", "val", "test")
    counts = dict(zip(names, args.counts)) if args.counts else None
    ratios = dict(zip(names, args.ratios)) if args.ratios else None
    result = assign_subject_splits(manifest, seed=args.seed, counts=counts, ratios=ratios)
    print(json.dumps(result["split"], indent=2, sort_keys=True))
    if not args.dry_run:
        write_json(args.output, result)
        print(f"Wrote split manifest to {args.output}")


if __name__ == "__main__":
    main()
