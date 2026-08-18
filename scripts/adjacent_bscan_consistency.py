"""Run interpretable adjacent-B-scan experiments on all validation cases.

Edit the dictionaries below, then run this file from the repository root. The
E2E volume is opened once per patient; the combined Gabor + depth detector is
run on the annotated scan and its neighbors. Every experiment writes paired
automatic JSON/PNG outputs plus an aggregate ``experiment_summary.json``.
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

from src.detector.consistency import apply_adjacent_consistency
from src.detector.detector import CALIBRATED_PHENOTYPE_V1_CONFIG, run_detector
from src.evaluation.metrics import evaluate_detection, load_ground_truth_masks
from src.evaluation.quantification import quantify_detection_labels
from src.evaluation.result_naming import automatic_result_stem
from src.loading.data_loading import LoadedScan, load_bscan, load_e2e_volume
from src.preprocess.preprocessing import get_layer_boundary, preprocess_loaded_scan
from src.visualization.pipeline import plot_detection_result


RUN_CONFIG = {
    "e2e_directory": Path("data/heyex/meta"),
    "manual_ground_truth_directory": Path("results/manual_ground_truth"),
    "output_root": Path("results/adjacent_bscan_experiments"),
    "neighbor_radius": 2,
    "overwrite": True,
    "continue_on_error": True,
}

# These variants test one question at a time. Keep their names stable so runs
# can be compared directly after pulling the result folders to another machine.
CONSISTENCY_EXPERIMENTS = {
    "single_scan_control": {
        # No filtering: confirms the batch run matches the combined baseline.
        "minimum_supporting_neighbors": {"barcoding": 0, "ea": 0},
        "minimum_overlap_fraction": 0.25,
        "maximum_center_shift_columns": 30.0,
    },
    "adjacent_any": {
        "minimum_supporting_neighbors": {"barcoding": 1, "ea": 1},
        "minimum_overlap_fraction": 0.25,
        "maximum_center_shift_columns": 30.0,
    },
    "adjacent_class_specific": {
        # EA has more scan-level false positives, so require stronger support.
        "minimum_supporting_neighbors": {"barcoding": 1, "ea": 2},
        "minimum_overlap_fraction": 0.25,
        "maximum_center_shift_columns": 30.0,
    },
    "adjacent_bilateral": {
        # Strongest anatomical persistence test: evidence on both sides.
        "minimum_supporting_neighbors": {"barcoding": 2, "ea": 2},
        "minimum_support_before": {"barcoding": 1, "ea": 1},
        "minimum_support_after": {"barcoding": 1, "ea": 1},
        "minimum_overlap_fraction": 0.25,
        "maximum_center_shift_columns": 30.0,
    },
    "adjacent_spatial_strict": {
        # Tests whether tighter alignment rejects vessel/structural detections.
        "minimum_supporting_neighbors": {"barcoding": 1, "ea": 1},
        "minimum_overlap_fraction": 0.50,
        "maximum_center_shift_columns": 20.0,
    },
    "hybrid_conservative": {
        # Reject only unsupported, short barcoding intervals with at least
        # three weak signals. EA is deliberately unaffected.
        "minimum_supporting_neighbors": {"barcoding": 0, "ea": 0},
        "minimum_overlap_fraction": 0.25,
        "maximum_center_shift_columns": 30.0,
        "hybrid_rejection": {
            "enabled": True,
            "labels": ("barcoding",),
            "maximum_supporting_neighbors": 0,
            "maximum_width_pixels": 12,
            "minimum_weak_evidence_failures": 3,
            "weak_evidence_maximums": {
                "texture_mean_z": 0.60,
                "texture_peak_z": 0.80,
                "depth_band_mean_z.near": 0.25,
                "probability_mean": 0.65,
            },
        },
    },
    "hybrid_balanced": {
        # Wider exploratory rule; still needs three independent weak signals.
        "minimum_supporting_neighbors": {"barcoding": 0, "ea": 0},
        "minimum_overlap_fraction": 0.25,
        "maximum_center_shift_columns": 30.0,
        "hybrid_rejection": {
            "enabled": True,
            "labels": ("barcoding",),
            "maximum_supporting_neighbors": 0,
            "maximum_width_pixels": 16,
            "minimum_weak_evidence_failures": 3,
            "weak_evidence_maximums": {
                "texture_mean_z": 0.70,
                "texture_peak_z": 0.90,
                "depth_band_mean_z.near": 0.35,
                "probability_mean": 0.70,
            },
        },
    },
}

PREPROCESSING_CONFIG = {
    "layer_name": "BM", "reference_row": None, "flatten_fill_value": 0.0,
    "depth_below_layer": 150, "include_boundary": True,
    "require_full_depth": False, "crop_fill_value": 0.0,
    "normalization_method": "zscore", "lower_percentile": 1.0,
    "upper_percentile": 99.0, "denoise_method": "gaussian",
    "gaussian_sigma": (1.0, 0.5),
}

DETECTOR_CONFIG = {
    "detector_type": "structural", "verticality_smoothing_sigma": 1.0,
    "verticality_threshold": 0.60, "gradient_quantile": 0.80,
    "minimum_component_size": 0, "column_upper_quantile": 0.90,
    "minimum_valid_pixels": 5, "signal_smoothing_sigma": 2.0,
    "median_iqr_multiplier": 1.0, "q90_iqr_multiplier": 0.5,
    "continuity_window_width": 15, "continuity_depth_lag": 4,
    "continuity_minimum_row_standard_deviation": 1e-6,
    "continuity_quantile": 0.60, "vertical_fraction_quantile": 0.70,
    "minimum_positive_run": 5, "maximum_negative_gap": 2,
    "edge_margin": 10,
    "phenotype_config": deepcopy(CALIBRATED_PHENOTYPE_V1_CONFIG),
}


def configure_combined_detector() -> None:
    phenotype = DETECTOR_CONFIG["phenotype_config"]
    phenotype["barcoding"]["texture_context"].update({
        "enabled": True,
        "minimum_interval_mean_z": 0.40,
        "minimum_interval_peak_z": 0.50,
    })
    phenotype["barcoding"]["depth_context"]["enabled"] = True
    phenotype["ea"]["depth_context"]["enabled"] = False


def intervals_from_labels(
    labels: np.ndarray,
    detector_metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    quantified = quantify_detection_labels(labels)
    saved_evidence = detector_metadata["phenotype_classification"][
        "interval_evidence"
    ]
    intervals = []
    for label in ("barcoding", "ea"):
        evidence_by_bounds = {
            (int(row["start"]), int(row["end"])): row
            for row in saved_evidence[label]
        }
        for interval in quantified[label]["intervals"]:
            bounds = (int(interval["start"]), int(interval["end"]))
            # Hybrid evidence is currently used only for barcoding. EA can
            # legitimately be reshaped when the final barcoding mask is
            # applied, so requiring an exact pre-label EA interval match made
            # otherwise valid cases fail unnecessarily.
            evidence = evidence_by_bounds.get(bounds, {})
            if label == "barcoding" and not evidence:
                raise KeyError(f"Missing {label} interval evidence for {bounds}.")
            intervals.append({"label": label, **interval, **evidence})
    return intervals


def discover_cases(directory: Path) -> list[dict[str, Any]]:
    cases = []
    for path in sorted(directory.glob("*_ground_truth.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        cases.append({
            "progression_group": str(payload["progression_group"]).lower(),
            "subject_id": int(payload["subject_id"]),
            "bscan_index": int(payload["bscan_index"]),
            "ground_truth_path": path,
        })
    if not cases:
        raise FileNotFoundError(f"No ground-truth JSON files found in {directory}")
    return cases


def index_e2e_files(directory: Path) -> dict[int, Path]:
    indexed = {}
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".e2e":
            match = re.search(r"ea[_-]?(\d+)", path.stem, re.IGNORECASE)
            if match:
                indexed[int(match.group(1))] = path
    return indexed


def process_scan(volume: Any, e2e_path: Path, index: int) -> dict[str, Any]:
    boundary = get_layer_boundary(volume, index, PREPROCESSING_CONFIG["layer_name"])
    loaded = LoadedScan(
        image=load_bscan(volume, index), source_type="e2e",
        source_path=e2e_path.resolve(),
        metadata={"bscan_index": index, "number_of_bscans": len(volume)},
        bscan_index=index, layer_boundary=boundary,
    )
    processed = preprocess_loaded_scan(loaded, **PREPROCESSING_CONFIG)
    detected = run_detector(processed.image, DETECTOR_CONFIG)
    labels = np.asarray(detected.labels, dtype=str)
    return {"image": processed.image, "labels": labels,
            "intervals": intervals_from_labels(labels, detected.metadata),
            "detector_metadata": detected.metadata}


def evaluate(labels: np.ndarray, ground_truth_path: Path) -> dict[str, Any]:
    metrics = {}
    for label, targets in {"barcoding": ("Barcoding",),
                           "ea": ("Early Atrophy (EA)",)}.items():
        target, valid = load_ground_truth_masks(
            ground_truth_path, width=labels.size, target_labels=targets,
            ignored_labels=("Uncertain", "Vessel / Structural"),
        )
        metrics[label] = evaluate_detection(labels == label, target, valid_mask=valid)
    return metrics


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for label in ("barcoding", "ea"):
        counts = {key: sum(row["validation_metrics"][label][key] for row in rows)
                  for key in ("true_positive", "false_positive",
                              "false_negative", "true_negative")}
        tp, fp, fn, tn = (counts[key] for key in
                          ("true_positive", "false_positive",
                           "false_negative", "true_negative"))
        ratio = lambda a, b: float(a / b) if b else 0.0
        output[label] = {**counts, "precision": ratio(tp, tp + fp),
                         "recall_sensitivity": ratio(tp, tp + fn),
                         "specificity": ratio(tn, tn + fp),
                         "f1_dice": ratio(2 * tp, 2 * tp + fp + fn)}
    return output


def run_case(case: dict[str, Any], e2e_path: Path) -> dict[str, Any]:
    volume = load_e2e_volume(e2e_path)
    target_index = case["bscan_index"]
    radius = int(RUN_CONFIG["neighbor_radius"])
    indices = range(max(0, target_index - radius),
                    min(len(volume), target_index + radius + 1))
    scans = {index: process_scan(volume, e2e_path, index) for index in indices}
    target = scans[target_index]
    neighbors = {index: scan["intervals"] for index, scan in scans.items()
                 if index != target_index}
    outputs = {}
    identity = {key: case[key] for key in
                ("progression_group", "subject_id", "bscan_index")}
    stem = automatic_result_stem(identity)

    for name, settings in CONSISTENCY_EXPERIMENTS.items():
        labels, evidence = apply_adjacent_consistency(
            target["labels"], target["intervals"], neighbors,
            target_bscan_index=target_index, require_same_label=True, **settings,
        )
        metrics = evaluate(labels, case["ground_truth_path"])
        numerical = quantify_detection_labels(labels)
        directory = Path(RUN_CONFIG["output_root"]) / name
        directory.mkdir(parents=True, exist_ok=True)
        png_path = directory / f"{stem}.png"
        json_path = directory / f"{stem}.json"
        if not RUN_CONFIG["overwrite"] and json_path.exists() and png_path.exists():
            outputs[name] = json.loads(json_path.read_text(encoding="utf-8"))
            continue
        plot_detection_result(
            target["image"], labels, png_path,
            title=f"{name} | Subject {case['subject_id']} | B-scan {target_index}",
        )
        payload = {
            **identity, "source_e2e_path": str(e2e_path.resolve()),
            "ground_truth_path": str(case["ground_truth_path"].resolve()),
            "neighbor_indices": list(neighbors), "consistency_experiment": name,
            "consistency_config": settings,
            "raw_intervals": target["intervals"], "interval_support": evidence,
            "neighbor_interval_summaries": neighbors,
            "labels": labels.tolist(), "barcoding": numerical["barcoding"],
            "ea": numerical["ea"], "validation_metrics": metrics,
            "detector_config": DETECTOR_CONFIG,
            "preprocessing_config": PREPROCESSING_CONFIG,
            "target_detector_metadata": target["detector_metadata"],
        }
        json_path.write_text(json.dumps(payload, indent=2, default=_json_value),
                             encoding="utf-8")
        outputs[name] = payload
    return outputs


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def main() -> None:
    configure_combined_detector()
    cases = discover_cases(Path(RUN_CONFIG["manual_ground_truth_directory"]))
    e2e_files = index_e2e_files(Path(RUN_CONFIG["e2e_directory"]))
    experiment_rows = {name: [] for name in CONSISTENCY_EXPERIMENTS}
    failures = []
    for position, case in enumerate(cases, 1):
        print(f"[{position}/{len(cases)}] subject {case['subject_id']}, "
              f"B-scan {case['bscan_index']}")
        try:
            e2e_path = e2e_files[case["subject_id"]]
            outputs = run_case(case, e2e_path)
            for name, payload in outputs.items():
                experiment_rows[name].append(payload)
        except Exception as exc:
            if not RUN_CONFIG["continue_on_error"]:
                raise
            failures.append({**case, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  failed: {failures[-1]['error']}")

    summary = {
        "number_of_cases": len(cases), "number_of_failures": len(failures),
        "failures": failures, "neighbor_radius": RUN_CONFIG["neighbor_radius"],
        "experiments": {
            name: {"config": CONSISTENCY_EXPERIMENTS[name],
                   "number_of_completed_cases": len(rows),
                   "aggregate_metrics": aggregate(rows) if rows else {}}
            for name, rows in experiment_rows.items()
        },
    }
    output = Path(RUN_CONFIG["output_root"])
    output.mkdir(parents=True, exist_ok=True)
    path = output / "experiment_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=_json_value),
                    encoding="utf-8")
    for name, result in summary["experiments"].items():
        print(name, result["aggregate_metrics"])
    print(f"Saved {path.resolve()}")


if __name__ == "__main__":
    main()
