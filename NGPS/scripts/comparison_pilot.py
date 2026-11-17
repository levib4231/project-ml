"""Run a bounded, exploratory noisy-only convergence/loss comparison."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ngps.comparison import comparison_pilot
from ngps.device import preferred_device

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default=preferred_device())
    parser.add_argument("--steps-per-epoch", type=int, default=32)
    parser.add_argument("--epochs", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--budget-seconds", type=float, default=600)
    args = parser.parse_args()
    print(comparison_pilot(args.manifest, args.config, args.output_dir, args.device,
                           args.steps_per_epoch, tuple(args.epochs), args.budget_seconds))
