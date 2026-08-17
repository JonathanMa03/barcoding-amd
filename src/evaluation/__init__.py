"""Detector metrics, ground truth loading, and tuning support."""

from .metrics import evaluate_detection, load_ground_truth_mask

__all__ = ["evaluate_detection", "load_ground_truth_mask"]
