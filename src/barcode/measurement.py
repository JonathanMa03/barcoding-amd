from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, binary_closing, rectangle


# ---------------------------------------------------------------------
# TODO after clinician labels/model predictions are finalized:
# Measurements should be run only on confirmed or predicted barcode-positive
# volumes/B-scans. For now, functions are generic and can measure any ROI/mask.
# ---------------------------------------------------------------------


def make_measurement_dirs(processed_dir: str | Path) -> dict[str, Path]:
    processed_dir = Path(processed_dir)

    dirs = {
        "processed": processed_dir,
        "features": processed_dir / "features",
        "figures": processed_dir / "figures" / "measurement_examples",
        "predictions": processed_dir / "predictions",
        "labels": processed_dir / "labels",
        "qc": processed_dir / "qc",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return dirs


def load_roi_volume(roi_path: str | Path) -> np.ndarray:
    roi_path = Path(roi_path)

    if not roi_path.exists():
        raise FileNotFoundError(f"ROI volume not found: {roi_path}")

    return np.load(roi_path)


def make_candidate_barcode_mask(
    roi: np.ndarray,
    threshold_method: str = "otsu",
    min_size: int = 50,
) -> np.ndarray:
    """
    Create a simple candidate barcode mask from one ROI B-scan.

    TODO after ResNet/Grad-CAM is finalized:
    Replace or combine this with Grad-CAM-guided masking so measurement focuses
    on model-localized barcode regions rather than simple intensity thresholding.
    """
    roi = roi.astype(float)
    roi = np.nan_to_num(roi)

    if threshold_method == "otsu":
        thresh = threshold_otsu(roi)
        mask = roi > thresh
    else:
        raise ValueError("Currently only threshold_method='otsu' is implemented.")

    mask = remove_small_objects(mask.astype(bool), min_size=min_size)
    mask = binary_closing(mask, rectangle(5, 3))

    return mask


def horizontal_runs(binary_profile: np.ndarray) -> list[tuple[int, int]]:
    """
    Convert a 1D binary profile into contiguous positive x-runs.

    Returns list of (start, end) intervals, where end is exclusive.
    """
    binary_profile = np.asarray(binary_profile).astype(bool)

    runs = []
    in_run = False
    start = None

    for i, val in enumerate(binary_profile):
        if val and not in_run:
            start = i
            in_run = True
        elif not val and in_run:
            runs.append((start, i))
            in_run = False

    if in_run:
        runs.append((start, len(binary_profile)))

    return runs


def measure_mask_bscan(mask: np.ndarray) -> dict[str, Any]:
    """
    Measure barcode properties from one binary B-scan mask.
    """
    mask = mask.astype(bool)

    height, width = mask.shape
    area_pixels = int(mask.sum())
    area_fraction = float(mask.mean())

    x_profile = mask.any(axis=0)
    runs = horizontal_runs(x_profile)

    if len(runs) == 0:
        total_width_px = 0
        min_x = np.nan
        max_x = np.nan
        bar_widths = []
    else:
        min_x = min(s for s, _ in runs)
        max_x = max(e for _, e in runs)
        total_width_px = int(max_x - min_x)
        bar_widths = [int(e - s) for s, e in runs]

    labeled = label(mask)
    components = regionprops(labeled)

    return {
        "area_pixels": area_pixels,
        "area_fraction": area_fraction,
        "n_bars": int(len(runs)),
        "total_width_px": total_width_px,
        "min_x": min_x,
        "max_x": max_x,
        "mean_bar_width_px": float(np.mean(bar_widths)) if bar_widths else 0.0,
        "median_bar_width_px": float(np.median(bar_widths)) if bar_widths else 0.0,
        "max_bar_width_px": float(np.max(bar_widths)) if bar_widths else 0.0,
        "bar_density": float(len(runs) / width),
        "n_components": int(len(components)),
    }


def measure_roi_bscan(
    roi: np.ndarray,
    threshold_method: str = "otsu",
    min_size: int = 50,
) -> dict[str, Any]:
    """
    Create candidate mask and measure one ROI B-scan.
    """
    mask = make_candidate_barcode_mask(
        roi,
        threshold_method=threshold_method,
        min_size=min_size,
    )

    measurements = measure_mask_bscan(mask)

    return {
        **measurements,
        "mask": mask,
    }


def measure_roi_volume(
    roi_volume: np.ndarray,
    threshold_method: str = "otsu",
    min_size: int = 50,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Measure barcode features for every B-scan in a volume.

    Returns
    -------
    bscan_df:
        One row per B-scan.
    mask_volume:
        Candidate mask volume.
    """
    rows = []
    masks = []

    for bidx in range(roi_volume.shape[0]):
        out = measure_roi_bscan(
            roi_volume[bidx],
            threshold_method=threshold_method,
            min_size=min_size,
        )

        mask = out.pop("mask")
        masks.append(mask)

        row = {
            "bscan_index": bidx,
            **out,
        }

        rows.append(row)

    bscan_df = pd.DataFrame(rows)
    mask_volume = np.stack(masks, axis=0)

    return bscan_df, mask_volume


def summarize_volume_measurements(bscan_df: pd.DataFrame) -> dict[str, Any]:
    """
    Aggregate B-scan-level measurements into one volume-level summary.
    """
    positive = bscan_df["area_pixels"] > 0

    return {
        "n_bscans": int(len(bscan_df)),
        "positive_bscan_count": int(positive.sum()),
        "positive_bscan_fraction": float(positive.mean()),
        "mean_area_fraction": float(bscan_df["area_fraction"].mean()),
        "max_area_fraction": float(bscan_df["area_fraction"].max()),
        "mean_total_width_px": float(bscan_df["total_width_px"].mean()),
        "max_total_width_px": float(bscan_df["total_width_px"].max()),
        "mean_n_bars": float(bscan_df["n_bars"].mean()),
        "max_n_bars": int(bscan_df["n_bars"].max()),
        "mean_bar_width_px": float(bscan_df["mean_bar_width_px"].mean()),
        "max_bar_width_px": float(bscan_df["max_bar_width_px"].max()),
        "mean_bar_density": float(bscan_df["bar_density"].mean()),
    }


def load_positive_scan_table(
    processed_dir: str | Path,
    predictions_file: str | Path | None = None,
    label_file: str | Path | None = None,
) -> pd.DataFrame:
    """
    Load scans to measure.

    Priority:
    1. If predictions_file is provided, measure predicted positives.
    2. Else if label_file is provided, measure clinician-labeled positives.

    TODO after ResNet predictions are finalized:
    Confirm prediction column names. Current expected volume-level prediction:
    barcode_volume_pred, where 1 means predicted barcode-positive.

    TODO after clinician labels arrive:
    Confirm label column name. Current expected clinician label:
    barcode_volume_status, with 'present' meaning positive.
    """
    if predictions_file is not None:
        df = pd.read_csv(predictions_file)

        if "barcode_volume_pred" not in df.columns:
            raise KeyError("Expected prediction column `barcode_volume_pred`.")

        return df[df["barcode_volume_pred"] == 1].copy()

    if label_file is not None:
        df = pd.read_csv(label_file)

        if "barcode_volume_status" not in df.columns:
            raise KeyError("Expected label column `barcode_volume_status`.")

        status = df["barcode_volume_status"].astype(str).str.lower().str.strip()
        return df[status.isin(["present", "positive", "yes", "1"])].copy()

    raise ValueError("Provide either predictions_file or label_file.")


def merge_with_preprocessing_qc(
    scan_df: pd.DataFrame,
    processed_dir: str | Path,
) -> pd.DataFrame:
    """
    Add ROI paths and QC metadata to scan table.
    """
    qc_path = Path(processed_dir) / "qc" / "preprocessing_qc.csv"

    if not qc_path.exists():
        raise FileNotFoundError(f"Preprocessing QC file not found: {qc_path}")

    qc_df = pd.read_csv(qc_path)

    merged = scan_df.merge(
        qc_df,
        on=["patient_id", "file_name"],
        how="left",
        suffixes=("", "_qc"),
    )

    return merged


def run_measurement_one_volume(
    row: pd.Series,
    processed_dirs: dict[str, Path],
    threshold_method: str = "otsu",
    min_size: int = 50,
    save_masks: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Measure one barcode-positive volume.
    """
    patient_id = str(row["patient_id"])
    file_name = row["file_name"]
    file_stem = Path(file_name).stem

    roi_volume = load_roi_volume(row["roi_path"])

    bscan_df, mask_volume = measure_roi_volume(
        roi_volume,
        threshold_method=threshold_method,
        min_size=min_size,
    )

    bscan_df.insert(0, "patient_id", patient_id)
    bscan_df.insert(1, "file_name", file_name)

    volume_summary = summarize_volume_measurements(bscan_df)
    volume_summary.update(
        {
            "patient_id": patient_id,
            "file_name": file_name,
            "measurement_ok": True,
            "error": None,
        }
    )

    if save_masks:
        mask_dir = processed_dirs["features"] / "candidate_masks" / patient_id
        mask_dir.mkdir(parents=True, exist_ok=True)

        mask_path = mask_dir / f"{file_stem}_measurement_mask.npy"
        np.save(mask_path, mask_volume)

        volume_summary["measurement_mask_path"] = str(mask_path)

    return bscan_df, volume_summary


def run_measurement_pipeline(
    processed_dir: str | Path,
    predictions_file: str | Path | None = None,
    label_file: str | Path | None = None,
    threshold_method: str = "otsu",
    min_size: int = 50,
    max_volumes: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run barcode measurements for predicted or clinician-confirmed positives.

    TODO after classifier is finalized:
    In normal use, predictions_file should be:
    data/processed/predictions/barcode_volume_predictions.csv

    TODO after clinician labels arrive:
    Can alternatively use clinician positives directly for measurement development.
    """
    processed_dirs = make_measurement_dirs(processed_dir)

    scan_df = load_positive_scan_table(
        processed_dir=processed_dir,
        predictions_file=predictions_file,
        label_file=label_file,
    )

    scan_df = merge_with_preprocessing_qc(scan_df, processed_dir)
    scan_df = scan_df[scan_df["preprocess_ok"] == True].copy()

    if max_volumes is not None:
        scan_df = scan_df.head(max_volumes)

    all_bscan_rows = []
    volume_rows = []

    print(f"Running measurements on {len(scan_df)} positive volumes.")

    for i, (_, row) in enumerate(scan_df.iterrows(), start=1):
        print(f"[{i}/{len(scan_df)}] {row['patient_id']} | {row['file_name']}")

        try:
            bscan_df, volume_summary = run_measurement_one_volume(
                row=row,
                processed_dirs=processed_dirs,
                threshold_method=threshold_method,
                min_size=min_size,
            )

            all_bscan_rows.append(bscan_df)
            volume_rows.append(volume_summary)

        except Exception as e:
            volume_rows.append(
                {
                    "patient_id": row.get("patient_id"),
                    "file_name": row.get("file_name"),
                    "measurement_ok": False,
                    "error": repr(e),
                }
            )

    if all_bscan_rows:
        bscan_measure_df = pd.concat(all_bscan_rows, ignore_index=True)
    else:
        bscan_measure_df = pd.DataFrame()

    volume_measure_df = pd.DataFrame(volume_rows)

    bscan_path = processed_dirs["features"] / "barcode_bscan_measurements.csv"
    volume_path = processed_dirs["features"] / "barcode_volume_measurements.csv"

    bscan_measure_df.to_csv(bscan_path, index=False)
    volume_measure_df.to_csv(volume_path, index=False)

    print("\nSaved:")
    print(f"- B-scan measurements: {bscan_path}")
    print(f"- Volume measurements: {volume_path}")

    return bscan_measure_df, volume_measure_df