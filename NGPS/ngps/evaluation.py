from __future__ import annotations

import csv
import math
from pathlib import Path

from datasets.manifest import SIGMAS, load_manifest, manifest_hash, resolve_path
from datasets.preparation import file_hash
from datasets.volume_folder import load_volume
from .baselines import nlm_volume
from .inference import infer_volume, load_model, timed_inference
from .metrics import aggregate_subjects, volume_metrics
from .provenance import provenance, write_json


def json_metrics(value):
    """Strict JSON: exact-match infinite PSNR is the string 'Infinity'."""
    if isinstance(value, dict):
        return {k: json_metrics(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_metrics(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            raise ValueError("Undefined metric")
        return "Infinity" if value > 0 else "-Infinity"
    return value


def evaluate(manifest_path, output_prefix, method="nlm", checkpoint=None, sigma=None, device="cpu"):
    if sigma is not None and sigma not in SIGMAS:
        raise ValueError(f"Unsupported sigma: {sigma}; choose one of {SIGMAS}")
    manifest = load_manifest(manifest_path, prepared=True)
    expected_test = {row["subject_id"] for row in manifest["subjects"] if row["split"] == "test"}
    if not expected_test:
        raise ValueError("Manifest has no held-out test subjects")
    digest = manifest_hash(manifest)
    model = None
    training = None
    if method == "model":
        if checkpoint is None:
            raise ValueError("Model evaluation requires --checkpoint")
        model, training = load_model(checkpoint, device, require_selected=True)
        if training["manifest_hash"] != digest:
            raise ValueError("Checkpoint/manifest mismatch; refusing unverified test split")
        if sigma is None or sigma != training["sigma"]:
            raise ValueError("Evaluate a trained model at its recorded sigma")
    elif method != "nlm":
        raise ValueError("Unknown evaluation method")
    elif checkpoint is not None:
        raise ValueError("NLM does not use a checkpoint")
    rows = []
    aggregates = {}
    for level in SIGMAS if sigma is None else (sigma,):
        level_rows = []
        for subject in manifest["subjects"]:
            if subject["split"] != "test":
                continue
            entry = subject["noisy"][f"{level:.2f}"]
            noisy_path = resolve_path(manifest_path, entry["path"])
            clean_path = resolve_path(manifest_path, subject["clean"])
            if file_hash(noisy_path) != entry["sha256"] or file_hash(clean_path) != subject["clean_sha256"]:
                raise ValueError("Prepared data hash mismatch")
            noisy = load_volume(noisy_path)
            function = (lambda v: infer_volume(model, v, device)) if model is not None else (lambda v: nlm_volume(v, level))
            pred, seconds = timed_inference(function, noisy, device if model is not None else "cpu")
            # This is the ONLY experiment path that opens held-out clean arrays.
            clean = load_volume(clean_path, unit_range=True)
            row = {"subject_id": subject["subject_id"], "sigma": level,
                   "method": training["method"] if training else "nlm",
                   "slices": int(noisy.shape[0]), "inference_seconds": seconds,
                   **volume_metrics(clean, pred)}
            level_rows.append(row)
        if {row["subject_id"] for row in level_rows} != expected_test:
            raise ValueError("Evaluation did not produce exactly one complete row per test subject")
        rows.extend(level_rows)
        aggregates[f"{level:.2f}"] = aggregate_subjects(level_rows)
    report = {"schema_version": 1, **provenance(device if model is not None else "cpu"),
              "manifest_hash": digest, "checkpoint_sha256": file_hash(checkpoint) if checkpoint else None,
              "training": training, "split": "test", "clipping": False,
              "metric_protocol": "slice means within subject; equal subject weights per sigma; data_range=1",
              "timing_protocol": "complete volume including transfers/assembly, no warmup, excludes I/O/model load",
              "baseline_config": {"h_sigma_multiplier": 0.8, "patch_size": 5, "patch_distance": 6,
                                  "fast_mode": True, "dimensionality": "2D slice-wise"} if method == "nlm" else None,
              "subjects": rows, "aggregate_by_sigma": aggregates}
    prefix = Path(output_prefix)
    write_json(str(prefix) + ".json", json_metrics(report))
    with Path(str(prefix) + ".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate_rows = [{"sigma": level, "metric": name, **stats}
                      for level, metrics in aggregates.items() for name, stats in metrics.items()]
    with Path(str(prefix) + "_aggregate.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    return report
