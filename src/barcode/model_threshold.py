from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d


@dataclass
class BarcodeInterval:
    """One contiguous predicted barcoding interval."""

    x_start: int
    x_end: int
    length_px: int
    mean_score: float
    max_score: float


@dataclass
class ThresholdPrediction:
    """Output from the threshold-based barcode detector."""

    raw_signal: np.ndarray
    smoothed_signal: np.ndarray
    threshold: float
    raw_mask: np.ndarray
    cleaned_mask: np.ndarray
    intervals: list[BarcodeInterval]


def longest_true_run(values: np.ndarray) -> int:
    """Return the longest consecutive run of True values."""

    values = np.asarray(values, dtype=bool)

    longest = 0
    current = 0

    for value in values:
        if value:
            current += 1
            longest = max(longest, current)
        else:
            current = 0

    return longest


def compute_persistence_signal(
    roi: np.ndarray,
    bright_quantile: float = 0.75,
) -> np.ndarray:
    """
    Compute a vertical bright-pixel persistence score for every column.

    Parameters
    ----------
    roi:
        Normalized sub-BM image with shape ``(depth, width)``.
    bright_quantile:
        Quantile of the full ROI used to define bright pixels.

    Returns
    -------
    np.ndarray
        Longest bright-run fraction for every image column.
    """
    roi = np.asarray(roi, dtype=np.float32)

    if roi.ndim != 2:
        raise ValueError(
            f"Expected a 2D ROI, received shape {roi.shape}."
        )

    if not 0 < bright_quantile < 1:
        raise ValueError("bright_quantile must lie between 0 and 1.")

    threshold = float(np.quantile(roi, bright_quantile))
    bright_mask = roi >= threshold

    depth = roi.shape[0]

    signal = np.array(
        [
            longest_true_run(bright_mask[:, x]) / depth
            for x in range(roi.shape[1])
        ],
        dtype=np.float32,
    )

    return signal


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """
    Estimate a threshold using Otsu's between-class variance criterion.

    Otsu's method selects the threshold that best separates the one-
    dimensional score distribution into two groups.
    """
    values = np.asarray(values, dtype=np.float32)
    finite_values = values[np.isfinite(values)]

    if finite_values.size == 0:
        raise ValueError("Cannot threshold an empty signal.")

    value_min = float(finite_values.min())
    value_max = float(finite_values.max())

    if value_max <= value_min:
        return value_min

    histogram, edges = np.histogram(
        finite_values,
        bins=bins,
        range=(value_min, value_max),
    )

    histogram = histogram.astype(np.float64)
    probabilities = histogram / histogram.sum()

    centers = (edges[:-1] + edges[1:]) / 2

    cumulative_probability = np.cumsum(probabilities)
    cumulative_mean = np.cumsum(probabilities * centers)
    total_mean = cumulative_mean[-1]

    denominator = (
        cumulative_probability
        * (1.0 - cumulative_probability)
    )

    between_class_variance = np.zeros_like(denominator)

    valid = denominator > 0

    between_class_variance[valid] = (
        (
            total_mean * cumulative_probability[valid]
            - cumulative_mean[valid]
        )
        ** 2
        / denominator[valid]
    )

    best_index = int(np.argmax(between_class_variance))

    return float(centers[best_index])


def remove_short_runs(
    mask: np.ndarray,
    min_length: int,
) -> np.ndarray:
    """Remove positive runs shorter than ``min_length`` columns."""

    mask = np.asarray(mask, dtype=bool)
    cleaned = mask.copy()

    start = None

    for index, value in enumerate(
        np.append(mask, False)
    ):
        if value and start is None:
            start = index

        elif not value and start is not None:
            run_length = index - start

            if run_length < min_length:
                cleaned[start:index] = False

            start = None

    return cleaned


def fill_short_gaps(
    mask: np.ndarray,
    max_gap: int,
) -> np.ndarray:
    """Fill negative gaps between positive runs when sufficiently short."""

    mask = np.asarray(mask, dtype=bool)
    filled = mask.copy()

    start = None

    for index, value in enumerate(
        np.append(mask, True)
    ):
        if not value and start is None:
            start = index

        elif value and start is not None:
            gap_length = index - start

            has_positive_left = start > 0 and mask[start - 1]
            has_positive_right = index < mask.size and mask[index]

            if (
                gap_length <= max_gap
                and has_positive_left
                and has_positive_right
            ):
                filled[start:index] = True

            start = None

    return filled


def extract_intervals(
    mask: np.ndarray,
    scores: np.ndarray,
) -> list[BarcodeInterval]:
    """Convert a one-dimensional Boolean mask into intervals."""

    mask = np.asarray(mask, dtype=bool)
    scores = np.asarray(scores, dtype=np.float32)

    if mask.shape != scores.shape:
        raise ValueError("mask and scores must have identical shapes.")

    intervals: list[BarcodeInterval] = []
    start = None

    for index, value in enumerate(
        np.append(mask, False)
    ):
        if value and start is None:
            start = index

        elif not value and start is not None:
            end = index - 1
            interval_scores = scores[start:index]

            intervals.append(
                BarcodeInterval(
                    x_start=int(start),
                    x_end=int(end),
                    length_px=int(end - start + 1),
                    mean_score=float(interval_scores.mean()),
                    max_score=float(interval_scores.max()),
                )
            )

            start = None

    return intervals


class ThresholdBarcodeDetector:
    """
    Unsupervised threshold detector for horizontal barcoding intervals.
    """

    def __init__(
        self,
        bright_quantile: float = 0.75,
        smooth_sigma: float = 4.0,
        min_interval_length: int = 12,
        max_gap: int = 6,
        threshold_multiplier: float = 1.0,
    ) -> None:
        self.bright_quantile = bright_quantile
        self.smooth_sigma = smooth_sigma
        self.min_interval_length = min_interval_length
        self.max_gap = max_gap
        self.threshold_multiplier = threshold_multiplier

    def predict(
        self,
        roi: np.ndarray,
    ) -> ThresholdPrediction:
        """Detect barcoding intervals in a normalized sub-BM ROI."""

        raw_signal = compute_persistence_signal(
            roi=roi,
            bright_quantile=self.bright_quantile,
        )

        smoothed_signal = gaussian_filter1d(
            raw_signal,
            sigma=self.smooth_sigma,
        )

        base_threshold = otsu_threshold(smoothed_signal)

        threshold = (
            base_threshold * self.threshold_multiplier
        )

        raw_mask = smoothed_signal >= threshold

        cleaned_mask = fill_short_gaps(
            mask=raw_mask,
            max_gap=self.max_gap,
        )

        cleaned_mask = remove_short_runs(
            mask=cleaned_mask,
            min_length=self.min_interval_length,
        )

        intervals = extract_intervals(
            mask=cleaned_mask,
            scores=smoothed_signal,
        )

        return ThresholdPrediction(
            raw_signal=raw_signal,
            smoothed_signal=smoothed_signal,
            threshold=float(threshold),
            raw_mask=raw_mask,
            cleaned_mask=cleaned_mask,
            intervals=intervals,
        )