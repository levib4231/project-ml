import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ngps.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description="Held-out subject evaluation; JSON and per-subject/aggregate CSV")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--method", choices=["model", "nlm"], required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--sigma", type=float, choices=[0.03, 0.06, 0.10])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = evaluate(args.manifest, args.output_prefix, args.method, args.checkpoint, args.sigma, args.device)
    print(report["aggregate_by_sigma"])


if __name__ == "__main__":
    main()
