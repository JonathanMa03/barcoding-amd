from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


# ---------------------------------------------------------------------
# TODO after clinician labels arrive:
# Confirm final label file path, label column name, prediction column name,
# and whether validation is volume-level, B-scan-level, or both.
# ---------------------------------------------------------------------


def make_validation_dirs(processed_dir: str | Path) -> dict[str, Path]:
    processed_dir = Path(processed_dir)

    dirs = {
        "processed": processed_dir,
        "validation": processed_dir / "validation",
        "figures": processed_dir / "figures" / "validation",
        "predictions": processed_dir / "predictions",
        "labels": processed_dir / "labels",
        "features": processed_dir / "features",
        "qc": processed_dir / "qc",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return dirs


def normalize_binary_label(value: Any) -> int | None:
    """
    Convert clinician label to binary.

    Returns:
    - 1 for positive/present
    - 0 for negative/absent
    - None for unsure/missing
    """
    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if value in {"present", "positive", "yes", "1", "true"}:
        return 1

    if value in {"absent", "negative", "no", "0", "false"}:
        return 0

    return None


def load_validation_inputs(
    processed_dir: str | Path,
    label_file: str | Path | None = None,
    prediction_file: str | Path | None = None,
    label_col: str = "barcode_volume_status",
    prob_col: str = "mean_barcode_prob",
) -> pd.DataFrame:
    """
    Load and merge clinician labels with volume-level predictions.

    TODO after labels/predictions are finalized:
    - Update label_col if clinician CSV uses different naming.
    - Update prob_col if prediction file changes.
    - Add B-scan-level validation if needed.
    """
    processed_dir = Path(processed_dir)

    if label_file is None:
        label_file = processed_dir / "labels" / "clinician" / "barcode_labels.csv"

    if prediction_file is None:
        prediction_file = processed_dir / "predictions" / "barcode_volume_predictions.csv"

    label_file = Path(label_file)
    prediction_file = Path(prediction_file)

    if not label_file.exists():
        raise FileNotFoundError(f"Missing clinician label file: {label_file}")

    if not prediction_file.exists():
        raise FileNotFoundError(f"Missing prediction file: {prediction_file}")

    labels = pd.read_csv(label_file)
    preds = pd.read_csv(prediction_file)

    merged = labels.merge(
        preds,
        on=["patient_id", "file_name"],
        how="inner",
        suffixes=("_label", "_pred"),
    )

    if label_col not in merged.columns:
        raise KeyError(f"Label column not found: {label_col}")

    if prob_col not in merged.columns:
        raise KeyError(f"Probability column not found: {prob_col}")

    merged["y_true"] = merged[label_col].apply(normalize_binary_label)
    merged = merged[merged["y_true"].notna()].copy()
    merged["y_true"] = merged["y_true"].astype(int)

    merged["y_score"] = merged[prob_col].astype(float)

    return merged


def binarize_scores(y_score, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(y_score) >= threshold).astype(int)


def confusion_counts(y_true, y_pred) -> dict[str, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def classification_metrics(
    y_true,
    y_score,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute core binary classification metrics.

    TODO after class balance is known:
    Decide whether threshold should be fixed at 0.5 or tuned using validation data.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = binarize_scores(y_score, threshold=threshold)

    counts = confusion_counts(y_true, y_pred)

    tn = counts["tn"]
    fp = counts["fp"]
    fn = counts["fn"]
    tp = counts["tp"]

    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    out = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "precision_ppv": precision_score(y_true, y_pred, zero_division=0),
        "npv": tn / (tn + fn) if (tn + fn) > 0 else np.nan,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }

    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
        out["pr_auc"] = average_precision_score(y_true, y_score)
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan

    return out


def metrics_dataframe(metrics: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"metric": key, "value": value} for key, value in metrics.items()]
    )


def compute_roc_table(y_true, y_score) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    return pd.DataFrame(
        {
            "fpr": fpr,
            "tpr": tpr,
            "threshold": thresholds,
        }
    )


def compute_pr_table(y_true, y_score) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    thresholds = np.append(thresholds, np.nan)

    return pd.DataFrame(
        {
            "precision": precision,
            "recall": recall,
            "threshold": thresholds,
        }
    )


def find_best_threshold(
    y_true,
    y_score,
    method: str = "youden",
) -> dict[str, float]:
    """
    Find threshold using validation data.

    method:
    - 'youden': maximize sensitivity + specificity - 1
    - 'f1': maximize F1

    TODO after validation design is finalized:
    Use only validation set to tune threshold, never test set.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    candidate_thresholds = np.unique(y_score)

    best_threshold = 0.5
    best_value = -np.inf

    for threshold in candidate_thresholds:
        m = classification_metrics(y_true, y_score, threshold=threshold)

        if method == "youden":
            value = m["sensitivity"] + m["specificity"] - 1
        elif method == "f1":
            value = m["f1"]
        else:
            raise ValueError("method must be 'youden' or 'f1'")

        if value > best_value:
            best_value = value
            best_threshold = threshold

    return {
        "method": method,
        "best_threshold": float(best_threshold),
        "best_value": float(best_value),
    }


def bootstrap_metric_ci(
    y_true,
    y_score,
    threshold: float = 0.5,
    n_boot: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Bootstrap confidence intervals for classification metrics.

    TODO after sample size is known:
    Increase n_boot for final report if runtime allows.
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    rng = np.random.default_rng(seed)
    n = len(y_true)

    rows = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)

        y_b = y_true[idx]
        s_b = y_score[idx]

        m = classification_metrics(y_b, s_b, threshold=threshold)
        rows.append(m)

    boot_df = pd.DataFrame(rows)

    summary_rows = []

    for col in boot_df.columns:
        if col in {"tp", "fp", "tn", "fn", "threshold"}:
            continue

        summary_rows.append(
            {
                "metric": col,
                "mean": boot_df[col].mean(),
                "ci_lower": boot_df[col].quantile(0.025),
                "ci_upper": boot_df[col].quantile(0.975),
            }
        )

    return pd.DataFrame(summary_rows)


def merge_measurements_with_labels(
    processed_dir: str | Path,
    label_file: str | Path | None = None,
    measurement_file: str | Path | None = None,
) -> pd.DataFrame:
    """
    Merge volume-level barcode measurements with clinician labels.

    TODO after measurement outputs stabilize:
    Confirm final measurement filename and merge keys.
    """
    processed_dir = Path(processed_dir)

    if label_file is None:
        label_file = processed_dir / "labels" / "clinician" / "barcode_labels.csv"

    if measurement_file is None:
        measurement_file = processed_dir / "features" / "barcode_volume_measurements.csv"

    labels = pd.read_csv(label_file)
    measurements = pd.read_csv(measurement_file)

    merged = labels.merge(
        measurements,
        on=["patient_id", "file_name"],
        how="inner",
    )

    return merged


def summarize_measurements(measure_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize quantitative barcode measurements.

    TODO after final measurement columns are selected:
    Add/remove variables from measurement_cols.
    """
    measurement_cols = [
        "positive_bscan_count",
        "positive_bscan_fraction",
        "mean_area_fraction",
        "max_area_fraction",
        "mean_total_width_px",
        "max_total_width_px",
        "mean_n_bars",
        "max_n_bars",
        "mean_bar_width_px",
        "max_bar_width_px",
        "mean_bar_density",
    ]

    existing = [c for c in measurement_cols if c in measure_df.columns]

    rows = []

    for col in existing:
        values = pd.to_numeric(measure_df[col], errors="coerce")

        rows.append(
            {
                "measurement": col,
                "n": values.notna().sum(),
                "mean": values.mean(),
                "std": values.std(),
                "median": values.median(),
                "min": values.min(),
                "max": values.max(),
            }
        )

    return pd.DataFrame(rows)


def run_validation_pipeline(
    processed_dir: str | Path,
    label_file: str | Path | None = None,
    prediction_file: str | Path | None = None,
    label_col: str = "barcode_volume_status",
    prob_col: str = "mean_barcode_prob",
    threshold: float = 0.5,
    n_boot: int = 1000,
) -> dict[str, pd.DataFrame]:
    """
    Run full validation and save outputs.

    TODO after labels/predictions are finalized:
    Confirm whether validation should use mean probability, max probability,
    or positive B-scan fraction as y_score.
    """
    dirs = make_validation_dirs(processed_dir)

    df = load_validation_inputs(
        processed_dir=processed_dir,
        label_file=label_file,
        prediction_file=prediction_file,
        label_col=label_col,
        prob_col=prob_col,
    )

    metric_dict = classification_metrics(
        df["y_true"],
        df["y_score"],
        threshold=threshold,
    )

    metrics_df = metrics_dataframe(metric_dict)

    roc_df = compute_roc_table(df["y_true"], df["y_score"])
    pr_df = compute_pr_table(df["y_true"], df["y_score"])

    threshold_df = pd.DataFrame(
        [
            find_best_threshold(df["y_true"], df["y_score"], method="youden"),
            find_best_threshold(df["y_true"], df["y_score"], method="f1"),
        ]
    )

    boot_df = bootstrap_metric_ci(
        df["y_true"],
        df["y_score"],
        threshold=threshold,
        n_boot=n_boot,
    )

    metrics_path = dirs["validation"] / "classification_metrics.csv"
    roc_path = dirs["validation"] / "roc_curve.csv"
    pr_path = dirs["validation"] / "precision_recall_curve.csv"
    threshold_path = dirs["validation"] / "threshold_search.csv"
    boot_path = dirs["validation"] / "bootstrap_metric_ci.csv"

    metrics_df.to_csv(metrics_path, index=False)
    roc_df.to_csv(roc_path, index=False)
    pr_df.to_csv(pr_path, index=False)
    threshold_df.to_csv(threshold_path, index=False)
    boot_df.to_csv(boot_path, index=False)

    print("Saved:")
    print(f"- {metrics_path}")
    print(f"- {roc_path}")
    print(f"- {pr_path}")
    print(f"- {threshold_path}")
    print(f"- {boot_path}")

    return {
        "validation_df": df,
        "metrics_df": metrics_df,
        "roc_df": roc_df,
        "pr_df": pr_df,
        "threshold_df": threshold_df,
        "bootstrap_df": boot_df,
    }