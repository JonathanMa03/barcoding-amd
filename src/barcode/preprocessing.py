# flatten_volume(...)
# extract_below_rpe_roi(...)
# normalize_intensity(...)
# zscore_roi(...)
# crop_volume(...)

# flatten_volume(...)
# extract_choroid(...)
# compute_hypertransmission_mask(...)
# remove_background(...)
# resize_for_resnet(...)
from typing import Literal

import numpy as np


def flatten_volume(
    volume: np.ndarray,
    layer_surface: np.ndarray,
    target_y: int | None = None,
    fill_value: float = 0.0,
) -> tuple[np.ndarray, int]:
    """
    Flatten an OCT volume using a segmentation layer, usually the RPE.

    Parameters
    ----------
    volume:
        OCT volume with shape (n_bscans, height, width).
    layer_surface:
        Segmentation layer array with shape (n_bscans, width).
        Values should be y-coordinates in image pixel space.
    target_y:
        Common row to align the layer to. If None, the median layer position is used.
    fill_value:
        Value used for columns with missing layer values.

    Returns
    -------
    flattened:
        Flattened OCT volume with same shape as input.
    target_y:
        Row used as the common alignment position.
    """
    volume = volume.astype(np.float32)
    layer_surface = np.asarray(layer_surface)

    if volume.ndim != 3:
        raise ValueError(f"Expected volume shape (n_bscans, height, width), got {volume.shape}")

    if layer_surface.ndim != 2:
        raise ValueError(f"Expected layer_surface shape (n_bscans, width), got {layer_surface.shape}")

    n_bscans, height, width = volume.shape

    if layer_surface.shape != (n_bscans, width):
        raise ValueError(
            f"Layer shape {layer_surface.shape} does not match volume slices/width {(n_bscans, width)}"
        )

    if target_y is None:
        target_y = int(np.nanmedian(layer_surface))

    flattened = np.full_like(volume, fill_value, dtype=np.float32)

    for b in range(n_bscans):
        for x in range(width):
            y = layer_surface[b, x]

            if np.isnan(y):
                continue

            shift = int(round(target_y - y))
            flattened[b, :, x] = np.roll(volume[b, :, x], shift)

    return flattened, target_y


def transform_layer_after_flattening(
    layer_surface: np.ndarray,
    reference_surface: np.ndarray,
    target_y: int,
) -> np.ndarray:
    """
    Transform a segmentation layer using the same shifts used for flattening.

    This is useful for checking how BM behaves after RPE-based flattening.

    Parameters
    ----------
    layer_surface:
        Layer to transform, shape (n_bscans, width).
    reference_surface:
        Layer used for flattening, usually RPE, shape (n_bscans, width).
    target_y:
        Common alignment row used during flattening.

    Returns
    -------
    transformed_layer:
        Layer coordinates after applying the flattening shifts.
    """
    layer_surface = np.asarray(layer_surface, dtype=np.float32)
    reference_surface = np.asarray(reference_surface, dtype=np.float32)

    if layer_surface.shape != reference_surface.shape:
        raise ValueError(
            f"layer_surface shape {layer_surface.shape} does not match "
            f"reference_surface shape {reference_surface.shape}"
        )

    shifts = target_y - reference_surface
    transformed = layer_surface + shifts

    return transformed


def extract_below_layer_roi(
    flattened_volume: np.ndarray,
    target_y: int,
    offset_top: int = 5,
    offset_bottom: int = 160,
) -> np.ndarray:
    """
    Extract a fixed ROI below a flattened reference layer.

    Parameters
    ----------
    flattened_volume:
        Flattened OCT volume with shape (n_bscans, height, width).
    target_y:
        Row where the reference layer was aligned.
    offset_top:
        Number of pixels below target_y where ROI begins.
    offset_bottom:
        Number of pixels below target_y where ROI ends.

    Returns
    -------
    roi_volume:
        ROI volume with shape (n_bscans, offset_bottom - offset_top, width).
    """
    if flattened_volume.ndim != 3:
        raise ValueError(
            f"Expected flattened_volume shape (n_bscans, height, width), got {flattened_volume.shape}"
        )

    _, height, _ = flattened_volume.shape

    roi_top = target_y + offset_top
    roi_bottom = target_y + offset_bottom

    if roi_top < 0 or roi_bottom > height:
        raise ValueError(
            f"ROI bounds [{roi_top}, {roi_bottom}) exceed image height {height}."
        )

    return flattened_volume[:, roi_top:roi_bottom, :]


def zscore_volume(
    volume: np.ndarray,
    eps: float = 1e-8,
    mode: Literal["global", "slice"] = "global",
) -> np.ndarray:
    """
    Z-score normalize a volume.

    Parameters
    ----------
    volume:
        Input array.
    eps:
        Small constant to avoid division by zero.
    mode:
        'global' normalizes using the full volume mean/std.
        'slice' normalizes each B-scan independently.

    Returns
    -------
    normalized:
        Z-scored volume.
    """
    volume = volume.astype(np.float32)

    if mode == "global":
        mu = np.nanmean(volume)
        sigma = np.nanstd(volume)
        return (volume - mu) / (sigma + eps)

    if mode == "slice":
        mu = np.nanmean(volume, axis=(1, 2), keepdims=True)
        sigma = np.nanstd(volume, axis=(1, 2), keepdims=True)
        return (volume - mu) / (sigma + eps)

    raise ValueError("mode must be either 'global' or 'slice'")


def minmax_normalize_volume(
    volume: np.ndarray,
    eps: float = 1e-8,
    mode: Literal["global", "slice"] = "global",
) -> np.ndarray:
    """
    Min-max normalize a volume to [0, 1].

    Parameters
    ----------
    volume:
        Input array.
    eps:
        Small constant to avoid division by zero.
    mode:
        'global' normalizes using the full volume min/max.
        'slice' normalizes each B-scan independently.

    Returns
    -------
    normalized:
        Min-max normalized volume.
    """
    volume = volume.astype(np.float32)

    if mode == "global":
        vmin = np.nanmin(volume)
        vmax = np.nanmax(volume)
        return (volume - vmin) / (vmax - vmin + eps)

    if mode == "slice":
        vmin = np.nanmin(volume, axis=(1, 2), keepdims=True)
        vmax = np.nanmax(volume, axis=(1, 2), keepdims=True)
        return (volume - vmin) / (vmax - vmin + eps)

    raise ValueError("mode must be either 'global' or 'slice'")


def preprocess_volume_from_layers(
    volume: np.ndarray,
    rpe: np.ndarray,
    offset_top: int = 5,
    offset_bottom: int = 160,
    normalization: Literal["none", "zscore", "minmax"] = "none",
    normalization_mode: Literal["global", "slice"] = "global",
) -> dict[str, np.ndarray | int]:
    """
    Convenience wrapper for the current volume preprocessing pipeline.

    Pipeline:
    1. RPE flattening.
    2. Fixed below-RPE ROI extraction.
    3. Optional intensity normalization.

    Parameters
    ----------
    volume:
        OCT volume with shape (n_bscans, height, width).
    rpe:
        RPE layer with shape (n_bscans, width).
    offset_top:
        ROI top offset below flattened RPE.
    offset_bottom:
        ROI bottom offset below flattened RPE.
    normalization:
        Intensity normalization method.
    normalization_mode:
        Whether normalization is global or slice-wise.

    Returns
    -------
    dict
        Contains flattened_volume, roi_volume, processed_roi, and target_y.
    """
    flattened_volume, target_y = flatten_volume(volume, rpe)

    roi_volume = extract_below_layer_roi(
        flattened_volume,
        target_y=target_y,
        offset_top=offset_top,
        offset_bottom=offset_bottom,
    )

    if normalization == "none":
        processed_roi = roi_volume
    elif normalization == "zscore":
        processed_roi = zscore_volume(roi_volume, mode=normalization_mode)
    elif normalization == "minmax":
        processed_roi = minmax_normalize_volume(roi_volume, mode=normalization_mode)
    else:
        raise ValueError("normalization must be one of: 'none', 'zscore', 'minmax'")

    return {
        "flattened_volume": flattened_volume,
        "roi_volume": roi_volume,
        "processed_roi": processed_roi,
        "target_y": target_y,
    }

from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd

from barcode.data_loading import load_e2e


def make_processed_dirs(processed_dir: str | Path) -> dict[str, Path]:
    """
    Create standard processed-data directories.

    Returns
    -------
    dict[str, Path]
        Dictionary of key output directories.
    """
    processed_dir = Path(processed_dir)

    dirs = {
        "processed": processed_dir,
        "qc": processed_dir / "qc",
        "roi": processed_dir / "roi",
        "flattened": processed_dir / "flattened",
        "labels": processed_dir / "labels",
        "clinician": processed_dir / "labels" / "clinician",
        "masks": processed_dir / "masks",
        "features": processed_dir / "features",
        "metadata": processed_dir / "metadata",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    readme_path = dirs["clinician"] / "README.txt"
    readme_path.write_text(
        """Place completed clinician label files here.

Expected filename:
barcode_labels.csv

Recommended columns:
patient_id
file_name
barcode_volume_status
first_positive_bscan
last_positive_bscan
label_confidence
clinician_notes
"""
    )

    return dirs


def get_e2e_paths(data_dir: str | Path, n_patients: int | None = None) -> list[dict[str, Any]]:
    """
    Collect E2E file paths from patient subfolders.

    Expected structure:
        data/heyex/001/*.E2E
        data/heyex/002/*.E2E
        ...

    Parameters
    ----------
    data_dir:
        Directory containing patient subfolders.
    n_patients:
        Number of patient folders to process. If None, process all.

    Returns
    -------
    list[dict]
        Rows containing patient_id, file_name, and e2e_path.
    """
    data_dir = Path(data_dir)

    patient_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])

    if n_patients is not None:
        patient_dirs = patient_dirs[:n_patients]

    rows = []

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name

        e2e_files = sorted(
            [
                p
                for p in patient_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".e2e"
            ]
        )

        for e2e_path in e2e_files:
            rows.append(
                {
                    "patient_id": patient_id,
                    "file_name": e2e_path.name,
                    "e2e_path": e2e_path,
                }
            )

    return rows


def _safe_shape(x):
    if x is None:
        return None
    return tuple(x.shape)


def extract_basic_metadata(ev) -> dict[str, Any]:
    """
    Extract basic metadata from an eyepy EyeVolume.
    """
    meta = {
        "shape": _safe_shape(ev.data),
        "n_bscans": ev.data.shape[0],
        "height": ev.data.shape[1],
        "width": ev.data.shape[2],
        "available_layers": list(ev.layers.keys()),
        "has_ilm": "ILM" in ev.layers,
        "has_rpe": "RPE" in ev.layers,
        "has_bm": "BM" in ev.layers,
        "scale": str(getattr(ev, "scale", None)),
        "scale_unit": getattr(ev, "scale_unit", None),
        "scale_x": getattr(ev, "scale_x", None),
        "scale_y": getattr(ev, "scale_y", None),
        "scale_z": getattr(ev, "scale_z", None),
        "laterality": getattr(ev, "laterality", None),
    }

    try:
        bscan_meta = ev.meta.get("bscan_meta", None)
        if bscan_meta is not None and len(bscan_meta) > 0:
            first = bscan_meta[0]
            meta["acquisition_time"] = str(getattr(first, "acquisitionTime", None))
            meta["quality"] = getattr(first, "quality", None)
            meta["n_bscans_meta"] = getattr(first, "n_bscans", None)
            meta["scan_pattern"] = getattr(first, "scan_pattern", None)
            meta["start_x"] = getattr(first, "start_x", None)
            meta["start_y"] = getattr(first, "start_y", None)
            meta["end_x"] = getattr(first, "end_x", None)
            meta["end_y"] = getattr(first, "end_y", None)
    except Exception:
        meta["metadata_parse_warning"] = True

    return meta


def preprocess_one_e2e(
    e2e_path: str | Path,
    patient_id: str,
    processed_dirs: dict[str, Path],
    offset_top: int = 5,
    offset_bottom: int = 160,
    normalization: str = "zscore",
    normalization_mode: str = "global",
    save_flattened: bool = False,
) -> dict[str, Any]:
    """
    Load and preprocess one E2E file.

    Saves ROI volume to:
        data/processed/roi/{patient_id}/{file_stem}_roi.npy

    Optionally saves flattened volume to:
        data/processed/flattened/{patient_id}/{file_stem}_flattened.npy
    """
    e2e_path = Path(e2e_path)
    file_stem = e2e_path.stem

    result = {
        "patient_id": patient_id,
        "file_name": e2e_path.name,
        "e2e_path": str(e2e_path),
        "loaded_ok": False,
        "preprocess_ok": False,
        "roi_saved": False,
        "flattened_saved": False,
        "error": None,
    }

    try:
        ev = load_e2e(e2e_path)
        result["loaded_ok"] = True

        meta = extract_basic_metadata(ev)
        result.update(meta)

        if "RPE" not in ev.layers:
            raise ValueError("Missing RPE layer.")

        if "BM" not in ev.layers:
            result["warning"] = "Missing BM layer."

        out = preprocess_volume_from_layers(
            volume=ev.data,
            rpe=ev.layers["RPE"].data,
            offset_top=offset_top,
            offset_bottom=offset_bottom,
            normalization=normalization,
            normalization_mode=normalization_mode,
        )

        patient_roi_dir = processed_dirs["roi"] / patient_id
        patient_roi_dir.mkdir(parents=True, exist_ok=True)

        roi_path = patient_roi_dir / f"{file_stem}_roi.npy"
        np.save(roi_path, out["processed_roi"])

        result["roi_path"] = str(roi_path)
        result["roi_shape"] = tuple(out["processed_roi"].shape)
        result["target_y"] = out["target_y"]
        result["roi_saved"] = True

        if save_flattened:
            patient_flat_dir = processed_dirs["flattened"] / patient_id
            patient_flat_dir.mkdir(parents=True, exist_ok=True)

            flattened_path = patient_flat_dir / f"{file_stem}_flattened.npy"
            np.save(flattened_path, out["flattened_volume"])

            result["flattened_path"] = str(flattened_path)
            result["flattened_shape"] = tuple(out["flattened_volume"].shape)
            result["flattened_saved"] = True

        result["preprocess_ok"] = True

    except Exception as e:
        result["error"] = repr(e)

    return result


def create_label_template(qc_df: pd.DataFrame, processed_dirs: dict[str, Path]) -> pd.DataFrame:
    """
    Create a volume-level clinician label template from preprocessing QC.
    """
    keep_cols = [
        "patient_id",
        "file_name",
        "n_bscans",
        "height",
        "width",
        "roi_path",
        "roi_shape",
        "has_rpe",
        "has_bm",
        "target_y",
    ]

    existing_cols = [c for c in keep_cols if c in qc_df.columns]

    label_df = qc_df.loc[qc_df["preprocess_ok"] == True, existing_cols].copy()

    label_df["barcode_volume_status"] = ""
    label_df["first_positive_bscan"] = ""
    label_df["last_positive_bscan"] = ""
    label_df["label_confidence"] = ""
    label_df["clinician_notes"] = ""

    template_path = processed_dirs["labels"] / "barcode_label_template.csv"
    label_df.to_csv(template_path, index=False)

    return label_df


def run_preprocessing_pipeline(
    data_dir: str | Path,
    processed_dir: str | Path,
    n_patients: int | None = None,
    offset_top: int = 5,
    offset_bottom: int = 160,
    normalization: str = "zscore",
    normalization_mode: str = "global",
    save_flattened: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run full batch preprocessing pipeline.

    Parameters
    ----------
    data_dir:
        Raw HEYEX directory containing patient folders.
    processed_dir:
        Output directory for processed files.
    n_patients:
        Number of patients to process. Use None for all.
    offset_top:
        ROI top offset below flattened RPE.
    offset_bottom:
        ROI bottom offset below flattened RPE.
    normalization:
        'none', 'zscore', or 'minmax'.
    normalization_mode:
        'global' or 'slice'.
    save_flattened:
        Whether to save full flattened volumes. False saves disk space.

    Returns
    -------
    qc_df:
        Preprocessing QC dataframe.
    label_df:
        Clinician label template dataframe.
    """
    processed_dirs = make_processed_dirs(processed_dir)
    rows = get_e2e_paths(data_dir, n_patients=n_patients)

    print(f"Found {len(rows)} E2E files.")

    results = []

    for i, row in enumerate(rows, start=1):
        print(f"[{i}/{len(rows)}] {row['patient_id']} | {row['file_name']}")

        result = preprocess_one_e2e(
            e2e_path=row["e2e_path"],
            patient_id=row["patient_id"],
            processed_dirs=processed_dirs,
            offset_top=offset_top,
            offset_bottom=offset_bottom,
            normalization=normalization,
            normalization_mode=normalization_mode,
            save_flattened=save_flattened,
        )

        results.append(result)

    qc_df = pd.DataFrame(results)

    qc_path = processed_dirs["qc"] / "preprocessing_qc.csv"
    qc_df.to_csv(qc_path, index=False)

    label_df = create_label_template(qc_df, processed_dirs)

    summary = {
        "n_files": int(len(qc_df)),
        "n_loaded_ok": int(qc_df["loaded_ok"].sum()) if "loaded_ok" in qc_df else 0,
        "n_preprocess_ok": int(qc_df["preprocess_ok"].sum()) if "preprocess_ok" in qc_df else 0,
        "n_patients": int(qc_df["patient_id"].nunique()) if "patient_id" in qc_df else 0,
    }

    summary_path = processed_dirs["qc"] / "preprocessing_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\nSaved:")
    print(f"- QC table: {qc_path}")
    print(f"- Label template: {processed_dirs['labels'] / 'barcode_label_template.csv'}")
    print(f"- Summary: {summary_path}")

    return qc_df, label_df