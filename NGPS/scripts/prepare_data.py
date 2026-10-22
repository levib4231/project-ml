from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.preparation import prepare


def main():
    parser = argparse.ArgumentParser(description="Prepare manifest subjects with explicit normalization and Gaussian noise")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--normalization", required=True, choices=["already-unit", "fixed-range"])
    parser.add_argument("--lower", type=float)
    parser.add_argument("--upper", type=float)
    args = parser.parse_args()
    print(prepare(args.manifest, args.output_root, args.normalization, args.lower, args.upper))


if __name__ == "__main__":
    main()
