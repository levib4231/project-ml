from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .manifest import SIGMAS, load_manifest, manifest_hash, resolve_path, stable_seed, validate_manifest
from .volume_folder import validate_volume


def normalize(volume, policy, lower=None, upper=None):
    """Explicit conversion to float32; no inferred scale and no clipping."""
    if not (np.issubdtype(volume.dtype, np.integer) or np.issubdtype(volume.dtype, np.floating)):
        raise ValueError("Source must have real integer or floating dtype")
    if volume.ndim != 3 or min(volume.shape) < 1 or not np.isfinite(volume).all():
        raise ValueError("Source must be a finite nonempty 3D numeric array")
    if policy == "already-unit":
        result = volume.astype(np.float32)
    elif policy == "fixed-range":
        if lower is None or upper is None or not np.isfinite([lower, upper]).all() or upper <= lower:
            raise ValueError("fixed-range needs finite lower < upper")
        if volume.min() < lower or volume.max() > upper:
            raise ValueError("Source outside declared normalization range")
        result = ((volume.astype(np.float64) - lower) / (upper - lower)).astype(np.float32)
    else:
        raise ValueError("Choose already-unit or fixed-range explicitly")
    # Check before float32 rounding too.
    if policy == "already-unit" and (volume.min() < 0 or volume.max() > 1):
        raise ValueError("Source is not in [0,1]")
    return validate_volume(result, unit_range=True)


def gaussian_noise(clean, sigma, seed):
    validate_volume(clean, unit_range=True)
    if sigma not in SIGMAS:
        raise ValueError(f"Supported sigma values: {SIGMAS}")
    noise = np.random.default_rng(seed).normal(0, sigma, clean.shape).astype(np.float32)
    return validate_volume(clean + noise)  # Not clipped: retain additive Gaussian model.


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(source_manifest, output_root, policy, lower=None, upper=None):
    source = load_manifest(source_manifest)
    if source.get("kind") != "source":
        raise ValueError("Expected a source manifest")
    out = Path(output_root).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite preparation directory: {out}")
    out.mkdir(parents=True)
    path = out / "manifest.json"
    result = {"schema_version": 1, "kind": "prepared", "seed": source["seed"],
              "source_manifest_hash": manifest_hash(source),
              "normalization": {"policy": policy, "lower": lower, "upper": upper},
              "noise": "unclipped independent additive Gaussian; numpy default_rng PCG64",
              "numpy_version": np.__version__, "subjects": []}
    for i, row in enumerate(source["subjects"]):
        src = resolve_path(source_manifest, row["source"])
        array = np.load(src, allow_pickle=False)
        if (str(array.dtype) != row.get("source_dtype")
                or list(array.shape) != row.get("source_shape")
                or [float(array.min()), float(array.max())] != row.get("source_range")):
            raise ValueError(f"Source metadata mismatch for {row['subject_id']}: {src}")
        clean = normalize(array, policy, lower, upper)
        if clean.shape[0] < 3:
            raise ValueError("At least three slices are needed for adjacent targets")
        folder = out / f"subject_{i:03d}"
        folder.mkdir()
        clean_path = folder / "clean.npy"
        np.save(clean_path, clean)
        record = {**row, "source": os.path.relpath(src, out), "source_sha256": file_hash(src),
                  "source_dtype": str(array.dtype), "source_range": [float(array.min()), float(array.max())],
                  "shape": list(clean.shape), "clean_dtype": str(clean.dtype),
                  "clean": str(clean_path.relative_to(out)),
                  "clean_sha256": file_hash(clean_path), "noisy": {}}
        for sigma in SIGMAS:
            key = f"{sigma:.2f}"
            seed = stable_seed(source["seed"], row["subject_id"], f"noise:{key}")
            noisy = gaussian_noise(clean, sigma, seed)
            dest = folder / f"noisy_{key}.npy"
            np.save(dest, noisy)
            record["noisy"][key] = {"path": str(dest.relative_to(out)), "seed": seed,
                                     "sha256": file_hash(dest),
                                     "range": [float(noisy.min()), float(noisy.max())]}
        result["subjects"].append(record)
    validate_manifest(result, path)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return path
