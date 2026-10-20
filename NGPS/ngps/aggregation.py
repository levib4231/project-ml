from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import t

from .metrics import METRIC_NAMES


def aggregate_subject_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    metric_names: Sequence[str] = METRIC_NAMES,
) -> list[dict[str, Any]]:
    """Aggregate subject means across subjects, never across pooled slices."""

    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["noise_level"]))].append(row)
    output: list[dict[str, Any]] = []
    for (method, noise_level), group in sorted(grouped.items()):
        subject_ids = [str(row["subject_id"]) for row in group]
        if len(subject_ids) != len(set(subject_ids)):
            raise ValueError(f"Duplicate subject rows for {method}/{noise_level}.")
        aggregate: dict[str, Any] = {
            "row_type": "aggregate",
            "method": method,
            "noise_level": noise_level,
            "subject_count": len(group),
        }
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            half_width = float(t.ppf(0.975, len(values) - 1) * std / np.sqrt(len(values))) if len(values) > 1 else 0.0
            aggregate[f"{metric}_mean"] = mean
            aggregate[f"{metric}_std"] = std
            aggregate[f"{metric}_ci95_low"] = mean - half_width
            aggregate[f"{metric}_ci95_high"] = mean + half_width
        output.append(aggregate)
    return output
