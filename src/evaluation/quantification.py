"""Numerical quantification of EA and barcoding detector labels."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def extract_label_intervals(
    labels: Sequence[str],
    target_label: str,
) -> list[dict[str, int]]:
    """Extract inclusive contiguous intervals for one column label."""
    label_array = np.asarray(labels, dtype=str)
    if label_array.ndim != 1:
        raise ValueError("labels must be one-dimensional.")

    mask = np.char.lower(label_array) == target_label.lower()
    padded = np.pad(mask.astype(np.int8), (1, 1), constant_values=0)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1

    return [
        {
            "start": int(start),
            "end": int(end),
            "width_pixels": int(end - start + 1),
        }
        for start, end in zip(starts, ends)
    ]


def quantify_detection_labels(labels: Sequence[str]) -> dict[str, Any]:
    """Summarize interval counts and widths for EA and barcoding labels."""
    label_array = np.asarray(labels, dtype=str)
    if label_array.ndim != 1 or label_array.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional sequence.")

    result: dict[str, Any] = {"number_of_columns": int(label_array.size)}
    for label in ("barcoding", "ea"):
        intervals = extract_label_intervals(label_array, label)
        widths = [interval["width_pixels"] for interval in intervals]
        result[label] = {
            "number_of_intervals": len(intervals),
            "interval_widths_pixels": widths,
            "total_width_pixels": int(sum(widths)),
            "mean_width_pixels": float(np.mean(widths)) if widths else 0.0,
            "median_width_pixels": float(np.median(widths)) if widths else 0.0,
            "minimum_width_pixels": int(min(widths)) if widths else 0,
            "maximum_width_pixels": int(max(widths)) if widths else 0,
            "intervals": intervals,
        }
    return result
