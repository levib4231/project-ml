from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.manifest import build_dataset_manifest, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect local clean/source pairs and write a deterministic manifest.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-name", required=True, help="Use an explicitly provisional name if identity is unresolved.")
    parser.add_argument("--preprocessing-version", default="unknown")
    parser.add_argument("--provenance-note", default="unverified")
    parser.add_argument("--foreground-threshold", type=float, default=0.0)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--allow-missing-source", action="store_true")
    args = parser.parse_args()

    manifest = build_dataset_manifest(
        args.data_dir,
        dataset_name=args.dataset_name,
        preprocessing_version=args.preprocessing_version,
        provenance_note=args.provenance_note,
        foreground_threshold=args.foreground_threshold,
        recursive=args.recursive,
        require_source=not args.allow_missing_source,
    )
    write_json(args.output, manifest)
    print(f"Wrote {manifest['subject_count']} subjects to {args.output}")


if __name__ == "__main__":
    main()
