"""Run EA/barcoding/normal detector rules using the configuration below."""

from pathlib import Path
from copy import deepcopy
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector.detector import (
    DETECTOR_CONFIG_0818,
    run_detector,
    save_detector_output,
)
from src.preprocess.preprocessing import load_preprocessed_scan


PIPELINE_CONFIG = {
    "input_path": Path("results/pipeline/preprocessed_scan.npz"),
    "output_path": Path("results/pipeline/detections.json"),
}

# Canonical selected configuration. Edit a copied value locally for a new
# experiment; do not mutate the shared module constant.
DETECTOR_CONFIG = deepcopy(DETECTOR_CONFIG_0818)


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
