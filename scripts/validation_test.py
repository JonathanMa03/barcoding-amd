"""Run the complete detector workflow on all manually selected validation scans.

The ten cases and their patient-specific B-scan indices are discovered from
``results/manual_ground_truth/*_ground_truth.json``. Paired automatic JSON and
PNG outputs are written to ``results/automatic_detector/``.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector.detector import CALIBRATED_PHENOTYPE_V1_CONFIG, run_detector
from src.evaluation.metrics import evaluate_detection, load_ground_truth_masks
from src.evaluation.quantification import quantify_detection_labels
from src.evaluation.result_naming import automatic_result_stem
from src.loading.data_loading import load_e2e_with_metadata
from src.preprocess.preprocessing import preprocess_loaded_scan
from src.visualization.pipeline import plot_detection_result


VALIDATION_CONFIG = {
    "e2e_directory": Path("data/heyex/meta"),
    "manual_ground_truth_directory": Path("results/manual_ground_truth"),
    "output_directory": Path("results/automatic_detector_gabor_depth"),
    "overwrite": True,
    "continue_on_error": True,
    "colors": {"ea": "tab:orange", "barcoding": "tab:red"},
    "figure_options": {"figsize": (12, 4), "dpi": 150},
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

# Optional classical-feature experiments. Keep all False for contextual v3.
# Use a separate VALIDATION_CONFIG["output_directory"] when comparing runs.
EXPERIMENT_CONFIG = {
    "enable_gabor_gate": True,
    "gabor_minimum_interval_mean_z": 0.4,
    "gabor_minimum_interval_peak_z": 0.5,
    "enable_barcoding_depth_gate": True,
    "enable_ea_depth_gate": False,
}


def configure_experimental_gates() -> None:
    """Apply experiment switches to the copied phenotype configuration."""
    phenotype = DETECTOR_CONFIG["phenotype_config"]
    phenotype["barcoding"]["texture_context"].update({
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


def discover_validation_cases(directory: Path) -> list[dict[str, Any]]:
    """Read validation identities and scan indices from manual JSON files."""
    cases = []
    for json_path in sorted(directory.glob("*_ground_truth.json")):
        with json_path.open("r", encoding="utf-8") as stream:
            annotation = json.load(stream)
        cases.append({
            "progression_group": str(annotation["progression_group"]).lower(),
            "subject_id": int(annotation["subject_id"]),
            "bscan_index": int(annotation["bscan_index"]),
            "ground_truth_path": json_path,
        })
    if not cases:
        raise FileNotFoundError(f"No manual ground-truth JSON files found in {directory}")
    return cases


def index_e2e_files(directory: Path) -> dict[int, Path]:
    """Index case-insensitive ea<number>.E2E filenames by subject number."""
    if not directory.is_dir():
        raise NotADirectoryError(f"E2E directory not found: {directory}")
    indexed: dict[int, Path] = {}
    for path in directory.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".e2e":
            continue
        match = re.search(r"ea[_-]?(\d+)", path.stem, flags=re.IGNORECASE)
        if match:
            indexed[int(match.group(1))] = path
    return indexed


def evaluate_case(labels: np.ndarray, ground_truth_path: Path) -> dict[str, Any]:
    """Score EA/barcoding while excluding uncertain and structural columns."""
    metrics = {}
    for predicted_label, target_labels in {
        "barcoding": ("Barcoding",),
        "ea": ("Early Atrophy (EA)",),
    }.items():
        target, valid = load_ground_truth_masks(
            ground_truth_path,
            width=labels.size,
            target_labels=target_labels,
            ignored_labels=("Uncertain", "Vessel / Structural"),
        )
        metrics[predicted_label] = evaluate_detection(
            labels == predicted_label,
            target,
            valid_mask=valid,
        )
    return metrics


def run_case(case: dict[str, Any], e2e_path: Path, output_directory: Path) -> dict[str, Any]:
    identity = {
        "progression_group": case["progression_group"],
        "subject_id": case["subject_id"],
        "bscan_index": case["bscan_index"],
    }
    stem = automatic_result_stem(identity)
    json_path = output_directory / f"{stem}.json"
    png_path = output_directory / f"{stem}.png"

    if (
        not VALIDATION_CONFIG["overwrite"]
        and json_path.exists()
        and png_path.exists()
    ):
        return {**identity, "status": "skipped_existing",
                "json_path": str(json_path.resolve()),
                "png_path": str(png_path.resolve())}

    loaded = load_e2e_with_metadata(
        e2e_path,
        selection="index",
        bscan_index=case["bscan_index"],
        layer_name=PREPROCESSING_CONFIG["layer_name"],
        metadata=identity,
    )
    processed = preprocess_loaded_scan(loaded, **PREPROCESSING_CONFIG)
    detected = run_detector(processed.image, DETECTOR_CONFIG)
    labels = np.asarray(detected.labels, dtype=str)
    numerical = quantify_detection_labels(labels)
    validation_metrics = evaluate_case(labels, case["ground_truth_path"])

    payload = {
        **identity,
        "source_e2e_path": str(e2e_path.resolve()),
        "ground_truth_path": str(case["ground_truth_path"].resolve()),
        "image_shape": list(processed.image.shape),
        "labels": labels.tolist(),
        "barcoding": numerical["barcoding"],
        "ea": numerical["ea"],
        "intervals_by_class": {
            "barcoding": numerical["barcoding"]["intervals"],
            "ea": numerical["ea"]["intervals"],
        },
        "validation_metrics": validation_metrics,
        "thresholds": getattr(detected.result, "thresholds", {}),
        "detector_config": DETECTOR_CONFIG,
        "preprocessing_config": PREPROCESSING_CONFIG,
        "detector_metadata": detected.metadata,
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, default=_json_value)

    plot_detection_result(
        processed.image,
        labels,
        png_path,
        title=(
            f"Subject {case['subject_id']} | "
            f"{case['progression_group'].title()} | "
            f"B-scan {case['bscan_index']}"
        ),
        colors=VALIDATION_CONFIG["colors"],
        figure_options=VALIDATION_CONFIG["figure_options"],
    )
    return {
        **identity,
        "status": "processed",
        "json_path": str(json_path.resolve()),
        "png_path": str(png_path.resolve()),
        "barcoding_dice": validation_metrics["barcoding"]["f1_dice"],
        "ea_dice": validation_metrics["ea"]["f1_dice"],
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def main() -> None:
    configure_experimental_gates()
    ground_truth_directory = Path(VALIDATION_CONFIG["manual_ground_truth_directory"])
    output_directory = Path(VALIDATION_CONFIG["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)
    cases = discover_validation_cases(ground_truth_directory)
    e2e_files = index_e2e_files(Path(VALIDATION_CONFIG["e2e_directory"]))

    rows = []
    for position, case in enumerate(cases, start=1):
        subject_id = case["subject_id"]
        print(
            f"[{position}/{len(cases)}] {case['progression_group']} subject "
            f"{subject_id}, B-scan {case['bscan_index']}"
        )
        try:
            if subject_id not in e2e_files:
                raise FileNotFoundError(
                    f"No ea{subject_id}.E2E file found under "
                    f"{VALIDATION_CONFIG['e2e_directory']}"
                )
            row = run_case(case, e2e_files[subject_id], output_directory)
            print(f"  {row['status']}: {Path(row['json_path']).name}")
        except Exception as exc:
            if not VALIDATION_CONFIG["continue_on_error"]:
                raise
            row = {
                "progression_group": case["progression_group"],
                "subject_id": subject_id,
                "bscan_index": case["bscan_index"],
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            print(f"  failed: {row['error']}")
        rows.append(row)

    manifest = {
        "number_of_cases": len(cases),
        "counts": {
            status: sum(row["status"] == status for row in rows)
            for status in ("processed", "skipped_existing", "failed")
        },
        "cases": rows,
    }
    manifest_path = output_directory / "validation_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
    print(f"Finished: {manifest['counts']}")
    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
