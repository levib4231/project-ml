from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from datasets.manifest import write_json
from datasets.noise import RicianNoiseConfig, derive_subject_seed, simulate_rician
from datasets.volume_folder import load_volume


def _label(sigma: float, reference: str) -> str:
    suffix = "range" if reference == "data_range_fraction" else "absolute"
    return f"sigma_{sigma:g}_{suffix}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic Rician observations from evaluation-only clean sources.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--data-root", default=None, help="Override the manifest path_root.")
    parser.add_argument("--sigma", type=float, action="append", required=True)
    parser.add_argument("--sigma-reference", choices=("absolute", "data_range_fraction"), required=True)
    parser.add_argument("--data-min", type=float, default=0.0)
    parser.add_argument("--data-max", type=float, default=1.0)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--clip-output", action="store_true", help="Explicitly clip noisy magnitudes to data-min/data-max.")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_root = Path(args.data_root) if args.data_root else Path(source_manifest["path_root"])
    output_root = Path(args.output_root)

    for sigma in args.sigma:
        config = RicianNoiseConfig(
            sigma=sigma,
            sigma_reference=args.sigma_reference,
            data_min=args.data_min,
            data_max=args.data_max,
            clip_output=args.clip_output,
        )
        label = _label(sigma, args.sigma_reference)
        records = []
        for subject in source_manifest["subjects"]:
            subject_id = str(subject["subject_id"])
            split = subject.get("split")
            if split not in {"train", "val", "test"}:
                raise ValueError(f"Subject {subject_id} has no train/val/test split assignment.")
            clean_path = data_root / subject["volume_path"]
            clean = load_volume(clean_path)
            seed = derive_subject_seed(args.base_seed, subject_id, label)
            noisy = simulate_rician(clean, config, seed=seed)
            destination = output_root / label / split / "noisy" / f"{subject_id}.npy"
            destination.parent.mkdir(parents=True, exist_ok=True)
            np.save(destination, noisy)
            records.append(
                {
                    "subject_id": subject_id,
                    "split": split,
                    "noisy_path": destination.as_posix(),
                    "clean_reference_path": clean_path.as_posix(),
                    "seed": seed,
                }
            )
        corruption_manifest = {
            "schema_version": 1,
            "source_manifest": str(manifest_path),
            "dataset_identity_status": source_manifest.get("identity_status", "unverified"),
            "corruption": {**asdict(config), "effective_sigma": config.effective_sigma, "equation": "sqrt((x+n_real)^2+n_imag^2)"},
            "base_seed": args.base_seed,
            "noise_label": label,
            "subjects": records,
        }
        output_manifest = output_root / label / "corruption_manifest.json"
        write_json(output_manifest, corruption_manifest)
        print(f"Wrote {len(records)} noisy volumes and {output_manifest}")


if __name__ == "__main__":
    main()
