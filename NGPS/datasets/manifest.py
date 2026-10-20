from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

import numpy as np


SCHEMA_VERSION = 1
SUBJECT_PATTERN = re.compile(r"^(subject(?P<number>\d+))_clean\.npy$")


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _volume_record(
    volume_path: Path,
    source_path: Path,
    root: Path,
    preprocessing_version: str,
    provenance_note: str,
    foreground_threshold: float,
) -> dict[str, Any]:
    volume = np.load(volume_path, mmap_mode="r", allow_pickle=False)
    if volume.ndim != 3:
        raise ValueError(f"Unexpected dimensions for {volume_path}: expected (D, H, W), got {volume.shape}")
    if not np.issubdtype(volume.dtype, np.floating):
        raise ValueError(f"Expected floating-point data in {volume_path}, got {volume.dtype}")
    if not np.isfinite(volume).all():
        raise ValueError(f"Non-finite values found in {volume_path}")

    foreground = volume > foreground_threshold
    foreground_count = int(foreground.sum())
    match = SUBJECT_PATTERN.match(volume_path.name)
    if match is None:
        raise ValueError(f"Cannot derive subject identifier from {volume_path.name}")
    subject_id = match.group(1)
    return {
        "subject_id": subject_id,
        "source_path": _relative(source_path, root),
        "source_type": "minc_gzip" if source_path.name.endswith(".mnc.gz") else source_path.suffix.lstrip("."),
        "volume_path": _relative(volume_path, root),
        "shape": [int(value) for value in volume.shape],
        "dtype": str(volume.dtype),
        "minimum": float(volume.min()),
        "maximum": float(volume.max()),
        "mean": float(volume.mean()),
        "foreground": {
            "threshold": float(foreground_threshold),
            "voxel_count": foreground_count,
            "fraction": float(foreground_count / volume.size),
            "mean": float(volume[foreground].mean()) if foreground_count else None,
        },
        "preprocessing_version": preprocessing_version,
        "sha256": sha256_file(volume_path),
        "source_sha256": sha256_file(source_path),
        "split": None,
        "provenance_note": provenance_note,
    }


def build_dataset_manifest(
    data_root: str | Path,
    *,
    dataset_name: str,
    preprocessing_version: str = "unknown",
    provenance_note: str = "unverified",
    foreground_threshold: float = 0.0,
    recursive: bool = False,
    require_source: bool = True,
) -> dict[str, Any]:
    """Inspect local files without inferring their dataset identity."""

    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {root}")
    iterator: Iterable[Path] = root.rglob("*_clean.npy") if recursive else root.glob("*_clean.npy")
    volume_paths = sorted(iterator, key=lambda path: path.as_posix())
    if not volume_paths:
        raise RuntimeError(f"No *_clean.npy volumes found in {root}")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    expected_shape: tuple[int, ...] | None = None
    for volume_path in volume_paths:
        match = SUBJECT_PATTERN.match(volume_path.name)
        if match is None:
            raise ValueError(f"Unexpected clean-volume filename: {volume_path.name}")
        subject_id = match.group(1)
        if subject_id in seen:
            raise ValueError(f"Duplicate subject identifier: {subject_id}")
        seen.add(subject_id)
        source_path = volume_path.with_name(f"{subject_id}_t1w_p4.mnc.gz")
        if require_source and not source_path.is_file():
            raise FileNotFoundError(f"Missing source file for {subject_id}: {source_path}")
        if not source_path.is_file():
            source_path = volume_path
        record = _volume_record(
            volume_path,
            source_path,
            root,
            preprocessing_version,
            provenance_note,
            foreground_threshold,
        )
        shape = tuple(record["shape"])
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError(f"Volume shape mismatch: expected {expected_shape}, got {shape} for {volume_path}")
        records.append(record)

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "identity_status": "unverified",
        "path_root": str(root),
        "preprocessing_version": preprocessing_version,
        "provenance_note": provenance_note,
        "subject_count": len(records),
        "subjects": records,
    }


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
