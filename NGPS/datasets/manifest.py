"""Versioned subject splits and file/path leakage checks."""
from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral
from pathlib import Path

import numpy as np

SIGMAS = (0.03, 0.06, 0.10)
_SPLITS = {"train", "val", "test"}
_HEX64 = set("0123456789abcdef")


def stable_seed(seed: int, subject: str, purpose: str) -> int:
    payload = json.dumps([seed, subject, purpose], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def assign_splits(subjects, seed=42):
    subjects = list(subjects)
    if (isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0
            or len(subjects) < 5 or len(set(subjects)) != len(subjects)):
        raise ValueError("Need a nonnegative integer seed and at least five unique subjects.")
    ordered = sorted(subjects, key=lambda s: (stable_seed(seed, s, "split"), s))
    n_holdout = max(1, int(len(ordered) * 0.2 + 0.5))
    n_train = len(ordered) - 2 * n_holdout
    if n_train < 1:
        raise ValueError("The deterministic 20/20 holdout leaves no training subjects.")
    return {s: ("train" if i < n_train else "val" if i < n_train + n_holdout else "test")
            for i, s in enumerate(ordered)}


def manifest_hash(manifest):
    return hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def resolve_path(manifest_path, relative):
    return (Path(manifest_path).resolve().parent / relative).resolve()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_hash(value, name):
    if (not isinstance(value, str) or len(value) != 64 or value != value.lower()
            or set(value) - _HEX64):
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")


def _check_shape(value, name):
    if (not isinstance(value, list) or len(value) != 3
            or any(isinstance(v, bool) or not isinstance(v, Integral) or v < 1 for v in value)):
        raise ValueError(f"{name} must be a positive [D,H,W] integer list")


def _check_range(value, name):
    if (not isinstance(value, list) or len(value) != 2
            or any(isinstance(v, bool) or not isinstance(v, (int, float))
                   or not math.isfinite(v) for v in value)
            or value[0] > value[1]):
        raise ValueError(f"{name} must be a finite ordered [min,max] list")


def validate_manifest(manifest, path):
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or manifest.get("kind") not in {"source", "prepared"}:
        raise ValueError("Unsupported manifest schema_version or kind")
    if isinstance(manifest.get("seed"), bool) or not isinstance(manifest.get("seed"), Integral) or manifest["seed"] < 0:
        raise ValueError("Manifest seed must be a nonnegative integer")
    records = manifest.get("subjects")
    if not isinstance(records, list) or not records:
        raise ValueError("Manifest subjects must be a nonempty list")
    try:
        splits = assign_splits([r["subject_id"] for r in records], manifest["seed"])
    except (KeyError, TypeError) as exc:
        raise ValueError("Each subject record needs a subject_id") from exc
    prepared = manifest.get("kind") == "prepared"
    owners, inodes, ids = {}, {}, set()
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("Subject records must be mappings")
        subject = row.get("subject_id")
        if not isinstance(subject, str) or not subject or subject in ids:
            raise ValueError("Subject IDs must be nonempty and unique")
        if row.get("split") not in _SPLITS or row["split"] != splits[subject]:
            raise ValueError("Duplicate subject or non-deterministic split")
        source = row.get("source")
        if not isinstance(source, str) or not source or Path(source).is_absolute():
            raise ValueError("Subject source must be a nonempty relative path")
        ids.add(subject)
        paths = [source]
        if not prepared:
            _check_hash(row.get("source_sha256"), f"{subject}.source_sha256")
            source_path = resolve_path(path, source)
            if not source_path.is_file() or _file_hash(source_path) != row["source_sha256"]:
                raise ValueError(f"Source file hash mismatch: {source_path}")
            _check_shape(row.get("source_shape"), f"{subject}.source_shape")
            if not isinstance(row.get("source_dtype"), str) or not row["source_dtype"]:
                raise ValueError(f"{subject}.source_dtype must record the source dtype")
            _check_range(row.get("source_range"), f"{subject}.source_range")
            try:
                source_array = np.load(source_path, mmap_mode="r", allow_pickle=False)
                actual_range = [float(np.min(source_array)), float(np.max(source_array))]
                valid_array = (source_array.ndim == 3 and np.issubdtype(source_array.dtype, np.number)
                               and np.isfinite(source_array).all())
            except (OSError, ValueError, TypeError) as exc:
                raise ValueError(f"Unable to inspect source array: {source_path}") from exc
            if (not valid_array or str(source_array.dtype) != row["source_dtype"]
                    or list(source_array.shape) != row["source_shape"]
                    or actual_range != row["source_range"]):
                raise ValueError(f"Source metadata mismatch: {source_path}")
        else:
            for field in ("source_sha256", "clean", "clean_sha256", "shape"):
                if field not in row:
                    raise ValueError(f"Prepared subject {subject} is missing {field}")
            _check_hash(row["source_sha256"], f"{subject}.source_sha256")
            _check_hash(row["clean_sha256"], f"{subject}.clean_sha256")
            _check_shape(row["shape"], f"{subject}.shape")
            if row.get("clean_dtype") != "float32":
                raise ValueError(f"Prepared subject {subject} clean_dtype must be float32")
            paths.append(row["clean"])
            noisy = row.get("noisy")
            if not isinstance(noisy, dict) or set(noisy) != {f"{s:.2f}" for s in SIGMAS}:
                raise ValueError("Expected all three noise levels")
            for sigma in SIGMAS:
                key = f"{sigma:.2f}"
                entry = noisy[key]
                if (not isinstance(entry, dict) or not isinstance(entry.get("path"), str)
                        or not isinstance(entry.get("seed"), int) or isinstance(entry["seed"], bool)):
                    raise ValueError(f"Malformed noisy entry for {subject} sigma {key}")
                if entry["seed"] != stable_seed(manifest["seed"], subject, f"noise:{key}"):
                    raise ValueError("Incorrect noise seed")
                _check_hash(entry.get("sha256"), f"{subject}.noisy[{key}].sha256")
                _check_range(entry.get("range"), f"{subject}.noisy[{key}].range")
                paths.append(entry["path"])
        local = set()
        for name in paths:
            if not isinstance(name, str) or not name or Path(name).is_absolute():
                raise ValueError("Manifest paths must be nonempty and relative")
            resolved = resolve_path(path, name)
            if prepared and not resolved.is_file():
                raise ValueError(f"Prepared manifest path does not exist: {resolved}")
            if resolved in local or resolved in owners:
                raise ValueError(f"Path leakage/alias: {resolved}")
            local.add(resolved)
            owners[resolved] = subject
            if resolved.exists():
                stat = resolved.stat()
                inode = (stat.st_dev, stat.st_ino)
                if inode in inodes:
                    raise ValueError(f"Hard-link leakage: {resolved}")
                inodes[inode] = subject
    return manifest


def load_manifest(path, prepared=False):
    with Path(path).open(encoding="utf-8") as f:
        manifest = validate_manifest(json.load(f), path)
    if prepared and manifest.get("kind") != "prepared":
        raise ValueError("Run prepare_data.py first; a prepared manifest is required")
    return manifest


def noisy_items(manifest, path, split, sigma):
    from .volume_folder import VolumeItem
    from .preparation import file_hash
    if split not in _SPLITS:
        raise ValueError(f"Unknown split: {split}")
    if sigma not in SIGMAS:
        raise ValueError(f"sigma must be one of {SIGMAS}")
    key = f"{sigma:.2f}"
    items = []
    for row in manifest["subjects"]:
        if row["split"] != split:
            continue
        entry = row["noisy"][key]
        noisy_path = resolve_path(path, entry["path"])
        if file_hash(noisy_path) != entry["sha256"]:
            raise ValueError(f"Noisy file hash mismatch: {noisy_path}")
        items.append(VolumeItem(row["subject_id"], noisy_path))
    if not items:
        raise ValueError(f"Manifest has no subjects in split {split!r}")
    return items
