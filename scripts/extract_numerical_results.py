"""Extract EA and barcoding interval counts and widths from detector output."""

from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.quantification import quantify_detection_labels


CONFIG = {
    "detection_path": Path("results/pipeline/detections.json"),
    "output_path": Path("results/pipeline/numerical_results.json"),
}


def main() -> None:
    detection_path = Path(CONFIG["detection_path"])
    if not detection_path.is_file():
        raise FileNotFoundError(f"Detection file not found: {detection_path}")

    with detection_path.open("r", encoding="utf-8") as stream:
        detection = json.load(stream)
    labels = detection.get("labels")
    if labels is None:
        raise KeyError("Detector output does not contain a 'labels' array.")

    results = quantify_detection_labels(labels)
    results["source_detection_path"] = str(detection_path.resolve())
    results["detector_type"] = detection.get("detector_type")

    output_path = Path(CONFIG["output_path"]).with_suffix(".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)

    print(
        "Barcoding: "
        f"{results['barcoding']['number_of_intervals']} interval(s), "
        f"widths={results['barcoding']['interval_widths_pixels']}"
    )
    print(
        "EA: "
        f"{results['ea']['number_of_intervals']} interval(s), "
        f"widths={results['ea']['interval_widths_pixels']}"
    )
    print(f"Numerical results -> {output_path.resolve()}")


if __name__ == "__main__":
    main()
