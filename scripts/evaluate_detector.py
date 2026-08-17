"""Evaluate saved detector labels against manual JSON annotations."""

import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import evaluate_detection, load_ground_truth_masks


CONFIG = {
    "detection_path": Path("results/pipeline/detections.json"),
    "ground_truth_path": Path(
        "results/manual_ground_truth/fast_08_bscan_048_ground_truth.json"
    ),
    "classes": {
        "barcoding": ("Barcoding",),
        "ea": ("Early Atrophy (EA)",),
    },
    "ignored_ground_truth_labels": ("Uncertain", "Vessel / Structural"),
    "output_path": Path("results/pipeline/evaluation.json"),
}


def main() -> None:
    with CONFIG["detection_path"].open("r", encoding="utf-8") as stream:
        detection = json.load(stream)
    labels = np.asarray(detection["labels"])
    metrics = {}
    for predicted_label, ground_truth_labels in CONFIG["classes"].items():
        predicted = labels == predicted_label
        target, valid = load_ground_truth_masks(
            CONFIG["ground_truth_path"],
            width=predicted.size,
            target_labels=ground_truth_labels,
            ignored_labels=CONFIG["ignored_ground_truth_labels"],
        )
        metrics[predicted_label] = evaluate_detection(
            predicted,
            target,
            valid_mask=valid,
        )
    output_path = CONFIG["output_path"].with_suffix(".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    print(f"Evaluation {metrics} -> {output_path.resolve()}")


if __name__ == "__main__":
    main()
