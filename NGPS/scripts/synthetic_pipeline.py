"""Small manifest -> preparation -> train -> infer -> held-out evaluation check."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from datasets.manifest import assign_splits, load_manifest, resolve_path
from datasets.preparation import prepare
from datasets.synthetic import _one_volume
from datasets.volume_folder import load_volume
from ngps.evaluation import evaluate
from ngps.inference import infer_volume, load_model, timed_inference
from ngps.training import train


def make_sources(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    subjects = [f"synthetic_{i:02d}" for i in range(10)]
    splits = assign_splits(subjects)
    records = []
    for i, subject in enumerate(subjects):
        _, clean = _one_volume(i, 4, 16, 0.0)
        clean = np.roll(clean, i, axis=2) * np.float32(0.75 + 0.02 * i)
        np.save(root / f"{subject}.npy", clean)
        source_path = root / f"{subject}.npy"
        records.append({"subject_id": subject, "split": splits[subject], "source": source_path.name,
                        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                        "source_dtype": str(clean.dtype), "source_shape": list(clean.shape),
                        "source_range": [float(clean.min()), float(clean.max())]})
    path = root / "subjects.json"
    path.write_text(json.dumps({"schema_version": 1, "kind": "source", "seed": 42,
                               "subjects": records}, indent=2) + "\n")
    return path


def run(root):
    root = Path(root)
    source = make_sources(root / "sources")
    manifest = prepare(source, root / "prepared", "already-unit")
    checkpoint = train(manifest, ROOT / "configs/synthetic_cpu.yaml", root / "run")
    model, _ = load_model(checkpoint, "cpu")
    test = next(r for r in load_manifest(manifest)["subjects"] if r["split"] == "test")
    noisy = load_volume(resolve_path(manifest, test["noisy"]["0.06"]["path"]))
    prediction, seconds = timed_inference(lambda v: infer_volume(model, v, "cpu"), noisy)
    np.save(root / "prediction.npy", prediction)
    report = evaluate(manifest, root / "model_metrics", "model", checkpoint, 0.06)
    evaluate(manifest, root / "nlm_metrics", "nlm")
    print(f"Synthetic train/infer/evaluate PASS: {len(report['subjects'])} held-out subjects, "
          f"standalone volume inference {seconds:.6f}s; artifacts: {root}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", required=True, help="Use a fresh directory outside the repository")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.work_dir)


if __name__ == "__main__":
    main()
