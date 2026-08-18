"""Interpretable interval filtering using detections in adjacent B-scans."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def _label_value(value: int | Mapping[str, int], label: str) -> int:
    if isinstance(value, Mapping):
        return int(value.get(label, value.get("default", 0)))
    return int(value)


def _nested_value(row: Mapping[str, Any], path: str) -> float:
    """Read a dot-separated numeric value from an interval evidence row."""
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(f"Interval evidence does not contain '{path}'.")
        value = value[part]
    return float(value)


def _hybrid_decision(
    interval: Mapping[str, Any],
    supporting_neighbor_count: int,
    config: Mapping[str, Any] | None,
) -> tuple[bool, dict[str, Any]]:
    """Return whether a candidate passes an auditable hybrid rejection rule."""
    if not config or not bool(config.get("enabled", False)):
        return True, {"enabled": False, "rejected": False}
    labels = {str(value) for value in config.get("labels", ("barcoding",))}
    label_applies = str(interval["label"]) in labels
    maximum_support = int(config.get("maximum_supporting_neighbors", 0))
    unsupported = supporting_neighbor_count <= maximum_support
    width = int(interval.get(
        "width_pixels", int(interval["end"]) - int(interval["start"]) + 1
    ))
    short = width <= int(config["maximum_width_pixels"])
    rule_results = []
    for path, maximum in dict(config.get("weak_evidence_maximums", {})).items():
        value = _nested_value(interval, path)
        rule_results.append({
            "feature": path,
            "value": value,
            "weak_when_at_or_below": float(maximum),
            "is_weak": value <= float(maximum),
        })
    weak_count = sum(row["is_weak"] for row in rule_results)
    required_weak = int(config.get("minimum_weak_evidence_failures", 1))
    weak = weak_count >= required_weak
    rejected = label_applies and unsupported and short and weak
    return not rejected, {
        "enabled": True,
        "label_applies": label_applies,
        "unsupported": unsupported,
        "maximum_supporting_neighbors": maximum_support,
        "short": short,
        "maximum_width_pixels": int(config["maximum_width_pixels"]),
        "weak_evidence_count": weak_count,
        "minimum_weak_evidence_failures": required_weak,
        "weak_evidence": weak,
        "evidence_rules": rule_results,
        "rejected": rejected,
    }


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
    hybrid_rejection: Mapping[str, Any] | None = None,
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
        adjacent_retained = (
            len(matches) >= required_total
            and before >= required_before
            and after >= required_after
        )
        hybrid_retained, hybrid_evidence = _hybrid_decision(
            interval, len(matches), hybrid_rejection
        )
        retained = adjacent_retained and hybrid_retained
        if not adjacent_retained:
            rejection_reason = "insufficient_adjacent_support"
        elif not hybrid_retained:
            rejection_reason = "hybrid_unsupported_short_weak"
        else:
            rejection_reason = None
        row = {
            **interval,
            "supporting_neighbor_count": len(matches),
            "support_before_count": before,
            "support_after_count": after,
            "required_supporting_neighbors": required_total,
            "required_support_before": required_before,
            "required_support_after": required_after,
            "supporting_matches": matches,
            "adjacent_rule_retained": adjacent_retained,
            "hybrid_rejection": hybrid_evidence,
            "retained": retained,
            "rejection_reason": rejection_reason,
        }
        evidence.append(row)
        if retained:
            labels[int(interval["start"]):int(interval["end"]) + 1] = label
    return labels, evidence
