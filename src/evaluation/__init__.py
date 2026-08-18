"""Detector metrics, ground truth loading, and tuning support."""

from .metrics import evaluate_detection, load_ground_truth_mask, load_ground_truth_masks
from .manual_annotation import ManualIntervalAnnotator
from .quantification import extract_label_intervals, quantify_detection_labels
from .result_naming import automatic_result_stem, resolve_result_identity

__all__ = [
    "evaluate_detection",
    "load_ground_truth_mask",
    "load_ground_truth_masks",
    "ManualIntervalAnnotator",
    "extract_label_intervals",
    "quantify_detection_labels",
    "automatic_result_stem",
    "resolve_result_identity",
]
