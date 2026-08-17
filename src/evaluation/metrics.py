"""Metrics for comparing column-wise detector output with interval annotations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def intervals_to_mask(
    intervals: Iterable[dict[str, Any]],
    width: int,
    *,
    labels: Iterable[str] | None = None,
) -> np.ndarray:
    """Rasterize annotation intervals into a Boolean column mask."""
    accepted = None if labels is None else {label.lower() for label in labels}
    mask = np.zeros(width, dtype=bool)
    for interval in intervals:
        if accepted is not None and str(interval.get("label", "")).lower() not in accepted:
            continue
        start = max(0, int(np.floor(float(interval.get("x_start", interval.get("start", 0))))))
        end = min(width - 1, int(np.ceil(float(interval.get("x_end", interval.get("end", -1))))))
        if end >= start:
            mask[start:end + 1] = True
    return mask


def load_ground_truth_mask(
    json_path: str | Path,
    *,
    width: int | None = None,
    labels: Iterable[str] = ("Barcoding",),
) -> np.ndarray:
    """Load manual JSON annotations as a column mask."""
    with Path(json_path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if width is None:
        width = int(payload["image_shape"][1])
    return intervals_to_mask(payload.get("annotations", []), width, labels=labels)


def load_ground_truth_masks(
    json_path: str | Path,
    *,
    width: int | None = None,
    target_labels: Iterable[str] = ("Barcoding",),
    ignored_labels: Iterable[str] = ("Uncertain", "Vessel / Structural"),
) -> tuple[np.ndarray, np.ndarray]:
    """Load a target mask and valid-scoring mask from manual annotations."""
    with Path(json_path).open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if width is None:
        width = int(payload["image_shape"][1])
    annotations = payload.get("annotations", [])
    target = intervals_to_mask(annotations, width, labels=target_labels)
    ignored = intervals_to_mask(annotations, width, labels=ignored_labels)
    return target, ~ignored


def evaluate_detection(
    predicted: np.ndarray,
    target: np.ndarray,
    *,
    valid_mask: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Calculate confusion counts, overlap, and classification metrics."""
    predicted = np.asarray(predicted, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if predicted.ndim != 1 or target.ndim != 1 or predicted.shape != target.shape:
        raise ValueError("predicted and target must be same-length 1D masks.")

    if valid_mask is None:
        valid = np.ones(predicted.shape, dtype=bool)
    else:
        valid = np.asarray(valid_mask, dtype=bool)
        if valid.shape != predicted.shape:
            raise ValueError("valid_mask must have the same shape as predicted.")
    predicted = predicted[valid]
    target = target[valid]
    if predicted.size == 0:
        raise ValueError("No valid columns remain after applying valid_mask.")

    tp = int(np.count_nonzero(predicted & target))
    fp = int(np.count_nonzero(predicted & ~target))
    fn = int(np.count_nonzero(~predicted & target))
    tn = int(np.count_nonzero(~predicted & ~target))

    def ratio(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": precision,
        "recall_sensitivity": recall,
        "specificity": ratio(tn, tn + fp),
        "accuracy": ratio(tp + tn, predicted.size),
        "f1_dice": ratio(2 * tp, 2 * tp + fp + fn),
        "intersection_over_union": ratio(tp, tp + fp + fn),
        "predicted_fraction": float(predicted.mean()),
        "target_fraction": float(target.mean()),
        "number_of_scored_columns": int(predicted.size),
        "number_of_ignored_columns": int(valid.size - predicted.size),
    }
