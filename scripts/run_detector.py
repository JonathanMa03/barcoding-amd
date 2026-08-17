"""Run EA/barcoding/normal detector rules using the configuration below."""

from pathlib import Path
from copy import deepcopy
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector.detector import (
    CALIBRATED_PHENOTYPE_V1_CONFIG,
    run_detector,
    save_detector_output,
)
from src.preprocess.preprocessing import load_preprocessed_scan


PIPELINE_CONFIG = {
    "input_path": Path("results/pipeline/preprocessed_scan.npz"),
    "output_path": Path("results/pipeline/detections.json"),
}

# Mirrors DETECTOR_CONFIG in pipeline_validation.ipynb.
DETECTOR_CONFIG = {
    "detector_type": "structural",
    "verticality_smoothing_sigma": 1.0,
    "verticality_threshold": 0.60,
    "gradient_quantile": 0.80,
    "minimum_component_size": 0,
    "column_upper_quantile": 0.90,
    "minimum_valid_pixels": 5,
    "signal_smoothing_sigma": 2.0,
    "median_iqr_multiplier": 1.0,
    "q90_iqr_multiplier": 0.5,
    "continuity_window_width": 15,
    "continuity_depth_lag": 4,
    "continuity_minimum_row_standard_deviation": 1e-6,
    "continuity_quantile": 0.60,
    "vertical_fraction_quantile": 0.70,
    "minimum_positive_run": 5,
    "maximum_negative_gap": 2,
    "edge_margin": 10,
    # Two independent EA/barcoding classifiers calibrated against the manual
    # annotations. Coefficients normally remain fixed. Probability thresholds
    # and interval cleanup settings can be adjusted per class below.
    "phenotype_config": deepcopy(CALIBRATED_PHENOTYPE_V1_CONFIG),
}


def main() -> None:
    processed = load_preprocessed_scan(PIPELINE_CONFIG["input_path"])
    detected = run_detector(processed.image, DETECTOR_CONFIG)
    detected.metadata["source"] = {
        **processed.metadata,
        "bscan_index": processed.bscan_index,
    }
    output = save_detector_output(detected, PIPELINE_CONFIG["output_path"])
    print(f"Detected labels {detected.metadata['label_counts']} -> {output}")


if __name__ == "__main__":
    main()
