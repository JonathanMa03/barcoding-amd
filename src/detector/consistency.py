"""Interpretable interval filtering using detections in adjacent B-scans."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def _label_value(value: int | Mapping[str, int], label: str) -> int:
    if isinstance(value, Mapping):
        return int(value.get(label, value.get("default", 0)))
    return int(value)


def best_interval_match(
    target: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    *,
    minimum_overlap_fraction: float,
    maximum_center_shift_columns: float,
    require_same_label: bool = True,
) -> dict[str, Any] | None:
    """Return the strongest compatible candidate interval, if one exists.

    Overlap is divided by the narrower interval. This permits a lesion to grow
    or shrink between slices while still requiring a shared spatial location.
    """
    target_start, target_end = int(target["start"]), int(target["end"])
    target_center = 0.5 * (target_start + target_end)
    target_width = target_end - target_start + 1
    best: tuple[tuple[float, float], dict[str, Any]] | None = None
    for candidate in candidates:
        if require_same_label and candidate["label"] != target["label"]:
            continue
        start, end = int(candidate["start"]), int(candidate["end"])
        intersection = max(0, min(target_end, end) - max(target_start, start) + 1)
        candidate_width = end - start + 1
        overlap = intersection / max(1, min(target_width, candidate_width))
        center_shift = abs(target_center - 0.5 * (start + end))
        if (
            overlap < float(minimum_overlap_fraction)
            or center_shift > float(maximum_center_shift_columns)
        ):
            continue
        match = {
            **candidate,
            "overlap_fraction": float(overlap),
            "center_shift_columns": float(center_shift),
        }
        score = (overlap, -center_shift)
        if best is None or score > best[0]:
            best = (score, match)
    return None if best is None else best[1]


def apply_adjacent_consistency(
    target_labels: np.ndarray,
    target_intervals: Sequence[Mapping[str, Any]],
    neighbor_intervals: Mapping[int, Sequence[Mapping[str, Any]]],
    *,
    target_bscan_index: int,
    minimum_supporting_neighbors: int | Mapping[str, int] = 1,
    minimum_support_before: int | Mapping[str, int] = 0,
    minimum_support_after: int | Mapping[str, int] = 0,
    minimum_overlap_fraction: float = 0.25,
    maximum_center_shift_columns: float = 30.0,
    require_same_label: bool = True,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Filter target intervals and return labels plus auditable evidence rows."""
    labels = np.full(np.asarray(target_labels).shape, "normal", dtype="<U10")
    evidence: list[dict[str, Any]] = []
    for interval in target_intervals:
        matches = []
        for neighbor_index, candidates in sorted(neighbor_intervals.items()):
            match = best_interval_match(
                interval,
                candidates,
                minimum_overlap_fraction=minimum_overlap_fraction,
                maximum_center_shift_columns=maximum_center_shift_columns,
                require_same_label=require_same_label,
            )
            if match is not None:
                matches.append({"bscan_index": int(neighbor_index), "match": match})

        label = str(interval["label"])
        before = sum(row["bscan_index"] < target_bscan_index for row in matches)
        after = sum(row["bscan_index"] > target_bscan_index for row in matches)
        required_total = _label_value(minimum_supporting_neighbors, label)
        required_before = _label_value(minimum_support_before, label)
        required_after = _label_value(minimum_support_after, label)
        retained = (
            len(matches) >= required_total
            and before >= required_before
            and after >= required_after
        )
        row = {
            **interval,
            "supporting_neighbor_count": len(matches),
            "support_before_count": before,
            "support_after_count": after,
            "required_supporting_neighbors": required_total,
            "required_support_before": required_before,
            "required_support_after": required_after,
            "supporting_matches": matches,
            "retained": retained,
            "rejection_reason": None if retained else "insufficient_adjacent_support",
        }
        evidence.append(row)
        if retained:
            labels[int(interval["start"]):int(interval["end"]) + 1] = label
    return labels, evidence
