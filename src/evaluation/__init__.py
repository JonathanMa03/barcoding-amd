"""Detector metrics, ground truth loading, and tuning support."""

from .metrics import evaluate_detection, load_ground_truth_mask
from .quantification import extract_label_intervals, quantify_detection_labels

__all__ = [
    "evaluate_detection",
    "load_ground_truth_mask",
    "extract_label_intervals",
    "quantify_detection_labels",
]
