"""Checkpoint 2 CLI shared by local, Colab and Kaggle execution."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ngps.checkpoint2 import (benchmark, export_diagnostics, freeze, pilot,
                             prepare_verified, train_matrix)
from ngps.device import preferred_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--manifest', required=True)
    prepare.add_argument('--verification', required=True)
    prepare.add_argument('--output-dir', required=True)
    for name in ('pilot', 'freeze'):
        command = commands.add_parser(name)
        command.add_argument('--manifest', required=True)
        command.add_argument('--config', required=True)
        command.add_argument('--output-dir', required=True)
        if name == 'pilot':
            command.add_argument('--device', default=preferred_device())
        else:
            command.add_argument('--pilot')
            command.add_argument('--max-train-batches', type=int)
            command.add_argument('--budget-seconds', type=float, default=600)
            command.add_argument('--budget-reason')
            command.add_argument('--synthetic', action='store_true', help='Software checks only, not medical evidence')
    for name in ('train', 'benchmark'):
        command = commands.add_parser(name)
        command.add_argument('--protocol', required=True)
        command.add_argument('--device', default=preferred_device())
    diagnostic = commands.add_parser('diagnostics')
    diagnostic.add_argument('--manifest', required=True)
    diagnostic.add_argument('--config', required=True)
    diagnostic.add_argument('--output', required=True)
    diagnostic.add_argument('--subject')
    diagnostic.add_argument('--sigma', type=float, choices=(0.03, 0.06, 0.10), default=0.06)
    diagnostic.add_argument('--checkpoint')
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare_verified(args.manifest, args.verification, args.output_dir)
    elif args.command == 'pilot':
        result = pilot(args.manifest, args.config, args.output_dir, args.device)
    elif args.command == 'freeze':
        result = freeze(args.manifest, args.config, args.output_dir, args.pilot,
                        args.max_train_batches, args.budget_seconds, args.budget_reason, args.synthetic)
    elif args.command == 'train':
        result = train_matrix(args.protocol, args.device)
    elif args.command == 'benchmark':
        result = benchmark(args.protocol, args.device)
    else:
        result = export_diagnostics(args.manifest, args.config, args.output, args.subject, args.sigma, args.checkpoint)
    print(result)


if __name__ == '__main__':
    main()
