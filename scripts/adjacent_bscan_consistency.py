"""Filter one B-scan's detections using support from adjacent E2E B-scans.

Edit the configuration dictionaries, then run:

    python scripts/adjacent_bscan_consistency.py

This script loads one E2E volume once, preprocesses the target scan and its
neighbors identically, runs the configured detector, and retains target
intervals supported at a similar horizontal location in enough neighboring
scans. It writes raw and consistency-filtered JSON/PNG outputs.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector.detector import CALIBRATED_PHENOTYPE_V1_CONFIG, run_detector
from src.evaluation.quantification import quantify_detection_labels
from src.loading.data_loading import LoadedScan, load_bscan, load_e2e_volume
from src.preprocess.preprocessing import get_layer_boundary, preprocess_loaded_scan
from src.visualization.pipeline import plot_detection_result


CONSISTENCY_CONFIG = {
    "e2e_path": Path("data/heyex/meta/ea8.E2E"),
    "target_bscan_index": 48,
    "neighbor_radius": 2,
    "minimum_supporting_neighbors": 1,
    "minimum_overlap_fraction": 0.25,
    "maximum_center_shift_columns": 30.0,
    "require_same_label": True,
    "output_directory": Path("results/adjacent_bscan_consistency"),
}


PREPROCESSING_CONFIG = {
    "layer_name": "BM",
    "reference_row": None,
    "flatten_fill_value": 0.0,
    "depth_below_layer": 150,
    "include_boundary": True,
    "require_full_depth": False,
    "crop_fill_value": 0.0,
    "normalization_method": "zscore",
    "lower_percentile": 1.0,
    "upper_percentile": 99.0,
    "denoise_method": "gaussian",
    "gaussian_sigma": (1.0, 0.5),
}


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
    "phenotype_config": deepcopy(CALIBRATED_PHENOTYPE_V1_CONFIG),
}

# Optional experiments. Leave False for the standard contextual-v3 detector.
EXPERIMENT_CONFIG = {
    "enable_gabor_gate": False,
    "gabor_minimum_interval_mean_z": 0.0,
    "gabor_minimum_interval_peak_z": 1.0,
    "enable_barcoding_depth_gate": False,
    "enable_ea_depth_gate": False,
}


def configure_experimental_gates() -> None:
    """Apply the easy-to-edit experiment switches to DETECTOR_CONFIG."""
    phenotype = DETECTOR_CONFIG["phenotype_config"]
    texture = phenotype["barcoding"]["texture_context"]
    texture.update({
        "enabled": bool(EXPERIMENT_CONFIG["enable_gabor_gate"]),
        "minimum_interval_mean_z": float(
            EXPERIMENT_CONFIG["gabor_minimum_interval_mean_z"]
        ),
        "minimum_interval_peak_z": float(
            EXPERIMENT_CONFIG["gabor_minimum_interval_peak_z"]
        ),
    })
    phenotype["barcoding"]["depth_context"]["enabled"] = bool(
        EXPERIMENT_CONFIG["enable_barcoding_depth_gate"]
    )
    phenotype["ea"]["depth_context"]["enabled"] = bool(
        EXPERIMENT_CONFIG["enable_ea_depth_gate"]
    )


def intervals_from_labels(labels: np.ndarray) -> list[dict[str, Any]]:
    """Return all EA/barcoding intervals with inclusive coordinates."""
    quantified = quantify_detection_labels(labels)
    intervals = []
    for label in ("barcoding", "ea"):
        for interval in quantified[label]["intervals"]:
            intervals.append({"label": label, **interval})
    return intervals


def interval_support(
    target: dict[str, Any],
    neighbor_intervals: list[dict[str, Any]],
) -> tuple[bool, dict[str, Any] | None]:
    """Find the strongest spatially compatible interval in one neighbor."""
    target_start, target_end = int(target["start"]), int(target["end"])
    target_center = 0.5 * (target_start + target_end)
    target_width = target_end - target_start + 1
    best = None
    for candidate in neighbor_intervals:
        if (
            CONSISTENCY_CONFIG["require_same_label"]
            and candidate["label"] != target["label"]
        ):
            continue
        start, end = int(candidate["start"]), int(candidate["end"])
        intersection = max(0, min(target_end, end) - max(target_start, start) + 1)
        candidate_width = end - start + 1
        overlap = intersection / max(1, min(target_width, candidate_width))
        center_shift = abs(target_center - 0.5 * (start + end))
        supported = (
            overlap >= float(CONSISTENCY_CONFIG["minimum_overlap_fraction"])
            and center_shift
            <= float(CONSISTENCY_CONFIG["maximum_center_shift_columns"])
        )
        score = (overlap, -center_shift)
        if supported and (best is None or score > best[0]):
            best = (score, {**candidate, "overlap_fraction": overlap,
                            "center_shift_columns": center_shift})
    return best is not None, None if best is None else best[1]


def process_scan(volume: Any, e2e_path: Path, index: int) -> dict[str, Any]:
    """Load, preprocess, and detect one already-open volume scan."""
    boundary = get_layer_boundary(
        volume, index, layer_name=PREPROCESSING_CONFIG["layer_name"]
    )
    loaded = LoadedScan(
        image=load_bscan(volume, index),
        source_type="e2e",
        source_path=e2e_path.resolve(),
        metadata={"bscan_index": index, "number_of_bscans": len(volume)},
        bscan_index=index,
        layer_boundary=boundary,
    )
    processed = preprocess_loaded_scan(loaded, **PREPROCESSING_CONFIG)
    detected = run_detector(processed.image, DETECTOR_CONFIG)
    labels = np.asarray(detected.labels, dtype=str)
    return {
        "index": index,
        "image": processed.image,
        "labels": labels,
        "intervals": intervals_from_labels(labels),
        "metadata": detected.metadata,
    }


def main() -> None:
    configure_experimental_gates()
    e2e_path = Path(CONSISTENCY_CONFIG["e2e_path"])
    if not e2e_path.is_file():
        raise FileNotFoundError(f"E2E file not found: {e2e_path}")
    volume = load_e2e_volume(e2e_path)
    target_index = int(CONSISTENCY_CONFIG["target_bscan_index"])
    if not 0 <= target_index < len(volume):
        raise IndexError(f"target B-scan must be between 0 and {len(volume)-1}")
    radius = int(CONSISTENCY_CONFIG["neighbor_radius"])
    if radius < 1:
        raise ValueError("neighbor_radius must be at least one.")
    indices = range(max(0, target_index - radius),
                    min(len(volume), target_index + radius + 1))
    scans = {index: process_scan(volume, e2e_path, index) for index in indices}
    target = scans[target_index]
    neighbor_indices = [index for index in scans if index != target_index]

    support_rows = []
    retained_intervals = []
    for interval in target["intervals"]:
        matches = []
        for neighbor_index in neighbor_indices:
            supported, match = interval_support(
                interval, scans[neighbor_index]["intervals"]
            )
            if supported:
                matches.append({"bscan_index": neighbor_index, "match": match})
        retained = len(matches) >= int(
            CONSISTENCY_CONFIG["minimum_supporting_neighbors"]
        )
        row = {**interval, "supporting_neighbor_count": len(matches),
               "supporting_matches": matches, "retained": retained}
        support_rows.append(row)
        if retained:
            retained_intervals.append(row)

    consistent_labels = np.full(target["labels"].shape, "normal", dtype="<U10")
    for interval in retained_intervals:
        consistent_labels[int(interval["start"]):int(interval["end"]) + 1] = (
            interval["label"]
        )

    output_directory = Path(CONSISTENCY_CONFIG["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    stem = f"{e2e_path.stem}_bscan_{target_index:03d}_adjacent_consistency"
    raw_png = plot_detection_result(
        target["image"], target["labels"], output_directory / f"{stem}_raw.png",
        title=f"{e2e_path.stem} | B-scan {target_index} | raw detector",
    )
    consistent_png = plot_detection_result(
        target["image"], consistent_labels,
        output_directory / f"{stem}_consistent.png",
        title=f"{e2e_path.stem} | B-scan {target_index} | adjacent-scan support",
    )
    payload = {
        "e2e_path": str(e2e_path.resolve()),
        "target_bscan_index": target_index,
        "neighbor_indices": neighbor_indices,
        "consistency_config": CONSISTENCY_CONFIG,
        "preprocessing_config": PREPROCESSING_CONFIG,
        "detector_config": DETECTOR_CONFIG,
        "experiment_config": EXPERIMENT_CONFIG,
        "raw": quantify_detection_labels(target["labels"]),
        "consistent": quantify_detection_labels(consistent_labels),
        "interval_support": support_rows,
        "target_detector_metadata": target["metadata"],
        "neighbor_interval_summaries": {
            str(index): scans[index]["intervals"] for index in neighbor_indices
        },
        "raw_labels": target["labels"].tolist(),
        "consistent_labels": consistent_labels.tolist(),
        "raw_png": str(raw_png),
        "consistent_png": str(consistent_png),
    }
    json_path = output_directory / f"{stem}.json"
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, default=str)
    print(f"Processed B-scans {list(indices)}")
    print(f"Raw intervals: {len(target['intervals'])}")
    print(f"Consistency-retained intervals: {len(retained_intervals)}")
    print(f"Saved {json_path.resolve()}")


if __name__ == "__main__":
    main()
