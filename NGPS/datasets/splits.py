from __future__ import annotations

import copy
import random
from typing import Any, Mapping


SPLIT_NAMES = ("train", "val", "test")


def _subject_ids(manifest: Mapping[str, Any]) -> list[str]:
    ids = [str(subject["subject_id"]) for subject in manifest.get("subjects", [])]
    if not ids:
        raise ValueError("Manifest contains no subjects.")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate subject identifiers in manifest.")
    return ids


def counts_from_ratios(subject_count: int, ratios: Mapping[str, float]) -> dict[str, int]:
    if set(ratios) != set(SPLIT_NAMES):
        raise ValueError(f"Ratios must define exactly {SPLIT_NAMES}.")
    if any(value < 0 for value in ratios.values()) or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("Split ratios must be non-negative and sum to 1.")
    exact = {name: subject_count * ratios[name] for name in SPLIT_NAMES}
    counts = {name: int(exact[name]) for name in SPLIT_NAMES}
    remainder = subject_count - sum(counts.values())
    order = sorted(SPLIT_NAMES, key=lambda name: (-(exact[name] - counts[name]), SPLIT_NAMES.index(name)))
    for name in order[:remainder]:
        counts[name] += 1
    return counts


def assign_subject_splits(
    manifest: Mapping[str, Any],
    *,
    seed: int,
    counts: Mapping[str, int] | None = None,
    ratios: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    subject_ids = _subject_ids(manifest)
    if (counts is None) == (ratios is None):
        raise ValueError("Provide exactly one of counts or ratios.")
    resolved_counts = dict(counts) if counts is not None else counts_from_ratios(len(subject_ids), ratios or {})
    if set(resolved_counts) != set(SPLIT_NAMES):
        raise ValueError(f"Counts must define exactly {SPLIT_NAMES}.")
    if any(not isinstance(value, int) or value < 0 for value in resolved_counts.values()):
        raise ValueError("Split counts must be non-negative integers.")
    if sum(resolved_counts.values()) != len(subject_ids):
        raise ValueError("Split counts must sum to the number of subjects.")

    shuffled = sorted(subject_ids)
    random.Random(seed).shuffle(shuffled)
    assigned: dict[str, list[str]] = {}
    offset = 0
    for name in SPLIT_NAMES:
        assigned[name] = sorted(shuffled[offset : offset + resolved_counts[name]])
        offset += resolved_counts[name]

    result = copy.deepcopy(dict(manifest))
    lookup = {subject_id: name for name, ids in assigned.items() for subject_id in ids}
    for subject in result["subjects"]:
        subject["split"] = lookup[str(subject["subject_id"])]
    result["split"] = {
        "seed": int(seed),
        "strategy": "explicit_counts" if counts is not None else "largest_remainder_ratios",
        "counts": {name: resolved_counts[name] for name in SPLIT_NAMES},
        "ratios": None if ratios is None else {name: float(ratios[name]) for name in SPLIT_NAMES},
        "subject_ids": assigned,
    }
    validate_split_manifest(result)
    return result


def validate_split_manifest(manifest: Mapping[str, Any]) -> None:
    subject_ids = set(_subject_ids(manifest))
    split = manifest.get("split")
    if not isinstance(split, Mapping):
        raise ValueError("Manifest has no split metadata.")
    groups = split.get("subject_ids")
    if not isinstance(groups, Mapping) or set(groups) != set(SPLIT_NAMES):
        raise ValueError("Split metadata must contain train, val, and test subject lists.")
    flattened = [str(subject) for name in SPLIT_NAMES for subject in groups[name]]
    if len(flattened) != len(set(flattened)):
        raise ValueError("A subject appears in more than one split.")
    if set(flattened) != subject_ids:
        raise ValueError("Split subject lists do not match the manifest inventory.")
    assignments = {str(subject["subject_id"]): subject.get("split") for subject in manifest["subjects"]}
    for name in SPLIT_NAMES:
        for subject_id in groups[name]:
            if assignments[str(subject_id)] != name:
                raise ValueError(f"Subject {subject_id} has inconsistent split metadata.")
