"""Bounded exploratory comparison. It never scores or loads clean or test arrays."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
import yaml

from datasets.manifest import load_manifest, manifest_hash, noisy_items
from datasets.triplet_dataset import VolumeTripletDataset
from .checkpoint2 import read_json
from .config import load_config
from .device import require_device, synchronize
from .guide import make_misalignment_mask
from .provenance import provenance, source_record, write_json
from .sampler import NeighborGuidedPatchSampler
from .training import train

VARIANTS = (("ngps", None), ("blindspot", None), ("adjacent", None),
            ("ngps", "no_rc"), ("ngps", "no_dynamic"))


def retrieval_diagnostics(manifest_path, config_path, subject="subject04", sigma=0.06):
    """Use one fixed training crop. Coordinates include replicated-edge clamping."""
    cfg, training, _ = load_config(config_path)
    manifest = load_manifest(manifest_path, prepared=True)
    items = [item for item in noisy_items(manifest, manifest_path, "train", sigma)
             if item.case_id == subject]
    if not items:
        raise ValueError("Diagnostics require a training subject")
    dataset = VolumeTripletDataset(items, cfg, crop_size=training.crop_size,
                                   use_guides=True, seed=training.seed)
    index = len(dataset) // 2
    _, prev, next_, gc, gp, gn = [v[None] for v in dataset[index]]
    sampler = NeighborGuidedPatchSampler(cfg)
    result = {"subject": subject, "sample_index": index, "sigma": sigma,
              "clean_references_accessed": False, "directions": {}}
    for name, guide, raw in (("prev", gp, prev), ("next", gn, next_)):
        detail = sampler.match_details(gc, guide, raw)
        mask = make_misalignment_mask(gc, guide, cfg.mask_threshold).bool()
        height, width = raw.shape[-2:]
        yy = torch.arange(height)[None, None, :, None]
        xx = torch.arange(width)[None, None, None, :]
        distance = ((detail["y"] - yy).float().square()
                    + (detail["x"] - xx).float().square()).sqrt().mean(1, keepdim=True)
        spread = (detail["y"].float().var(1, unbiased=False, keepdim=True)
                  + detail["x"].float().var(1, unbiased=False, keepdim=True)).sqrt()
        stats = {"dynamic_fraction": float(mask.float().mean())}
        for region, selected in (("all", torch.ones_like(mask)), ("dynamic", mask)):
            values = detail["cost"][selected].numpy()
            stats[region] = {"pixels": int(values.size),
                             "cost_quantiles_0_50_90_99_100": np.quantile(values, [0, .5, .9, .99, 1]).tolist() if values.size else None,
                             "mean_offset_pixels": float(distance[selected].mean()) if values.size else None,
                             "mean_topk_spread_pixels": float(spread[selected].mean()) if values.size else None}
        result["directions"][name] = stats
    return result


def comparison_pilot(manifest_path, config_path, output_dir, device="mps",
                     steps_per_epoch=32, epochs=(1, 2), budget_seconds=600):
    """Matched 32/64-update default study, not the medical benchmark or tuning."""
    require_device(device)
    cfg, training, raw = load_config(config_path)
    if (type(steps_per_epoch) is not int or steps_per_epoch < 1
            or not epochs or any(type(e) is not int or e < 1 for e in epochs)
            or len(set(epochs)) != len(epochs)
            or not np.isfinite(budget_seconds) or budget_seconds <= 0):
        raise ValueError("Invalid pilot workload/budget")
    if training.batch_size != 1 or training.seed != 42:
        raise ValueError("Comparison requires batch1 and seed42")
    manifest = load_manifest(manifest_path, prepared=True)
    subject = next((r for r in manifest["subjects"]
                    if r["subject_id"] == "subject04" and r["split"] == "train"), None)
    if subject is None or subject["shape"][0] - 2 < steps_per_epoch:
        raise ValueError("Subject04 must have enough interior slices")
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    protocol = {"kind": "exploratory_noisy_only_comparison_pilot",
                "manifest_path": str(Path(manifest_path).resolve()), "manifest_hash": manifest_hash(manifest),
                "source_tree_sha256": source_record()["tree_sha256"], "hardware": provenance(device),
                "base_config": raw, "device": str(device), "sigma": 0.06,
                "training_subjects": ["subject04"], "center_slices": steps_per_epoch,
                "epochs": list(epochs), "updates": [steps_per_epoch * e for e in epochs],
                "variants": [list(v) for v in VARIANTS], "seed": 42,
                "validation_batches": 16, "budget_seconds_per_run": budget_seconds,
                "budget_scope": "Independent local pilot safety guard, not a transferred T4 runtime claim",
                "selection": "fixed_final_epoch; no ranking or selection from pilot scores",
                "normalization": "Original active-pixel normalization; omit disabled terms without rescaling",
                "limitations": "Training subset; sigma0.06 only; noisy monitoring is not clean-risk or cross-objective ranking. No test scoring.",
                "schedule": "Same shuffled center slices and deterministic crops at each epoch across variants; longer runs restart from identical initialization."}
    protocol["protocol_hash"] = manifest_hash(protocol)
    write_json(out / "protocol.json", protocol)
    write_json(out / "retrieval_diagnostics.json", retrieval_diagnostics(manifest_path, config_path))
    rows = []
    for count in epochs:
        config = {**raw, "training": {**raw.get("training", {}), "epochs": count}}
        config_path_run = out / f"epochs_{count}.yaml"
        config_path_run.write_text(yaml.safe_dump(config))
        for method, ablation in VARIANTS:
            if source_record()["tree_sha256"] != protocol["source_tree_sha256"]:
                raise ValueError("Runtime changed after pilot freeze")
            name = f"{method}{'_' + ablation if ablation else ''}_steps{steps_per_epoch * count}"
            synchronize(device)
            start = time.perf_counter()
            checkpoint = train(manifest_path, config_path_run, out / "runs" / name,
                               device=device, method=method, sigma=0.06,
                               center_slices=steps_per_epoch, train_subjects=["subject04"],
                               max_train_batches=steps_per_epoch, max_val_batches=16,
                               experiment_hash=protocol["protocol_hash"], ablation=ablation,
                               gradient_diagnostics=True)
            synchronize(device)
            elapsed = time.perf_counter() - start
            meta = read_json(checkpoint.parent / "metadata.json")
            row = {"run": name, "optimizer_steps": meta["optimizer_steps"],
                   "wall_seconds": elapsed, "within_cap": elapsed <= budget_seconds,
                   "training_history": meta["training_history"],
                   "validation_history": meta["validation_history"], "device_memory": meta["device_memory"]}
            rows.append(row)
            write_json(out / "summary.partial.json", rows)
            write_json(checkpoint.parent / "pilot_budget.json", {"wall_seconds": elapsed,
                       "cap_seconds": budget_seconds, "within_cap": elapsed <= budget_seconds})
            if meta["optimizer_steps"] != steps_per_epoch * count or elapsed > budget_seconds:
                raise RuntimeError("Pilot schedule/budget failed; preserve partial evidence")
    write_json(out / "summary.json", {"protocol_hash": protocol["protocol_hash"], "runs": rows,
               "clean_references_accessed": False, "test_scoring_performed": False})
    return out / "summary.json"
