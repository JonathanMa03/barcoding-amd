"""Hypertransmission, EA, and barcoding detection."""

from .consistency import apply_adjacent_consistency, best_interval_match
from .detector import (
    DetectorOutput,
    CALIBRATED_PHENOTYPE_V1_CONFIG,
    DETECTOR_CONFIG_0818,
    STRUCTURAL_HYPERTD_V1_CONFIG,
    detect_barcoding,
    detect_ea,
    detect_hypertransmission,
    classify_ea_and_barcoding,
    detect_structural_hypertransmission,
    run_detector,
    save_detector_output,
)

__all__ = [
    "DetectorOutput", "CALIBRATED_PHENOTYPE_V1_CONFIG",
    "DETECTOR_CONFIG_0818",
    "STRUCTURAL_HYPERTD_V1_CONFIG", "detect_barcoding",
    "detect_ea", "detect_hypertransmission",
    "classify_ea_and_barcoding",
    "detect_structural_hypertransmission", "run_detector",
    "save_detector_output", "apply_adjacent_consistency",
    "best_interval_match",
]
