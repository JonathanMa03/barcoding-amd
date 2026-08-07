# detector.py
#
# Responsibilities:
# - Apply detector rules to mathematical feature signals.
# - Support the original weighted-score detector.
# - Run the structural-hypertransmission detector.
# - Calculate scan-relative robust thresholds.
# - Construct intensity, continuity, and vertical-refinement masks.
# - Fill short gaps and remove short positive runs.
# - Extract contiguous horizontal candidate intervals.
# - Return complete structured detector results and metadata.
#
# Future responsibilities:
# - Tune thresholds against manual annotations.
# - Learn detector parameters using regression/classification.
# - Compare classical and learned detector variants.
#
# Inputs:
# - Preprocessed and denoised two-dimensional OCT image
# - Detector configuration
#
# Outputs:
# - Mathematical feature signals
# - Intermediate detector masks
# - Final barcode candidate mask
# - Contiguous candidate intervals
# - Thresholds and detector metadata
#
# This module must not load E2E data, preprocess scans, or produce figures.


from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from src.detector.features import (
    compute_column_intensity_statistics,
    compute_column_verticality_statistics,
    compute_local_depth_continuity,
    compute_simple_verticality_map,
    create_structural_mask,
    smooth_finite_signal,
)


FEATURE_NAMES = (
    "verticality",
    "persistence",
    "continuity",
    "amplitude",
    "heterogeneity",
)

STRUCTURAL_FEATURES = (
    "verticality",
    "continuity",
    "heterogeneity",
)

HYPERTRANSMISSION_FEATURES = (
    "persistence",
    "amplitude",
)


DEFAULT_INDIVIDUAL_WEIGHTS: dict[str, float] = {
    "verticality": 0.25,
    "persistence": 0.25,
    "continuity": 0.15,
    "amplitude": 0.20,
    "heterogeneity": 0.15,
}


DEFAULT_STRUCTURAL_WEIGHTS: dict[str, float] = {
    "verticality": 0.45,
    "continuity": 0.275,
    "heterogeneity": 0.275,
}


DEFAULT_HYPERTRANSMISSION_WEIGHTS: dict[str, float] = {
    "persistence": 0.55,
    "amplitude": 0.45,
}


DEFAULT_GROUP_WEIGHTS: dict[str, float] = {
    "structural": 0.55,
    "hypertransmission": 0.45,
}


@dataclass
class DetectionInterval:
    """
    One contiguous horizontal barcode detection interval.

    Attributes
    ----------
    x_start:
        Left endpoint in horizontal image coordinates.
    x_end:
        Right endpoint in horizontal image coordinates.
    width_pixels:
        Horizontal width of the interval in pixels.
    start_index:
        Index of the first positive score location.
    end_index:
        Index of the final positive score location.
    number_of_positions:
        Number of evaluated horizontal positions in the interval.
    mean_score:
        Mean barcode score within the interval.
    maximum_score:
        Maximum barcode score within the interval.
    minimum_score:
        Minimum barcode score within the interval.
    """

    x_start: float
    x_end: float
    width_pixels: float

    start_index: int
    end_index: int
    number_of_positions: int

    mean_score: float
    maximum_score: float
    minimum_score: float


@dataclass
class DetectionResult:
    """
    Output from the barcode detector.

    Attributes
    ----------
    x_positions:
        Horizontal positions corresponding to the score and masks.
    score:
        Final combined barcode score.
    raw_mask:
        Boolean mask produced directly from thresholding.
    cleaned_mask:
        Boolean mask after spatial cleanup.
    intervals:
        Contiguous barcode-positive intervals from the cleaned mask.
    scoring_mode:
        Either ``"individual"`` or ``"grouped"``.
    threshold:
        Score threshold used to create the raw mask.
    feature_contributions:
        Weighted contribution from each individual feature.
    group_scores:
        Group-level scores. Empty when individual scoring is used.
    metadata:
        Detector configuration and diagnostic information.
    """

    x_positions: np.ndarray

    score: np.ndarray
    raw_mask: np.ndarray
    cleaned_mask: np.ndarray

    intervals: list[DetectionInterval]

    scoring_mode: str
    threshold: float

    feature_contributions: dict[str, np.ndarray]
    group_scores: dict[str, np.ndarray]

    metadata: dict[str, Any]


def _validate_one_dimensional_signal(
    values: np.ndarray,
    *,
    name: str,
    expected_length: int | None = None,
) -> np.ndarray:
    """
    Validate one numerical feature or score signal.
    """
    array = np.asarray(
        values,
        dtype=np.float32,
    )

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional; "
            f"received shape {array.shape}."
        )

    if array.size == 0:
        raise ValueError(
            f"{name} must not be empty."
        )

    if expected_length is not None and array.size != expected_length:
        raise ValueError(
            f"{name} contains {array.size} values, "
            f"but expected {expected_length}."
        )

    if not np.isfinite(array).all():
        raise ValueError(
            f"{name} contains non-finite values."
        )

    return array


def validate_feature_result(
    feature_result: Any,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
]:
    """
    Validate a FeatureSignals-like object.

    The detector uses standardized features because feature magnitudes
    must be comparable before weighting.
    """
    required_attributes = {
        "x_positions",
        "standardized_features",
    }

    missing = [
        attribute
        for attribute in required_attributes
        if not hasattr(
            feature_result,
            attribute,
        )
    ]

    if missing:
        raise TypeError(
            "feature_result is missing required attributes: "
            f"{missing}"
        )

    x_positions = _validate_one_dimensional_signal(
        feature_result.x_positions,
        name="x_positions",
    )

    standardized_features = (
        feature_result.standardized_features
    )

    if not isinstance(
        standardized_features,
        Mapping,
    ):
        raise TypeError(
            "feature_result.standardized_features must be a mapping."
        )

    validated_features: dict[
        str,
        np.ndarray,
    ] = {}

    for feature_name in FEATURE_NAMES:
        if feature_name not in standardized_features:
            raise KeyError(
                f"Missing standardized feature '{feature_name}'."
            )

        validated_features[
            feature_name
        ] = _validate_one_dimensional_signal(
            standardized_features[
                feature_name
            ],
            name=(
                f"standardized feature "
                f"'{feature_name}'"
            ),
            expected_length=x_positions.size,
        )

    return (
        x_positions.astype(np.float32),
        validated_features,
    )


def validate_weights(
    weights: Mapping[str, float],
    required_names: tuple[str, ...],
    *,
    normalize: bool = True,
    require_nonnegative: bool = True,
    tolerance: float = 1e-8,
) -> dict[str, float]:
    """
    Validate and optionally normalize a weight dictionary.

    Parameters
    ----------
    weights:
        Mapping from feature or group name to numerical weight.
    required_names:
        Exact names required in the mapping.
    normalize:
        Whether weights should be divided by their total so they sum to
        one.
    require_nonnegative:
        Whether negative weights are prohibited.
    tolerance:
        Small numerical tolerance used when checking the weight sum.

    Returns
    -------
    dict[str, float]
        Validated weight dictionary.
    """
    if not isinstance(
        weights,
        Mapping,
    ):
        raise TypeError(
            "weights must be a mapping."
        )

    missing = (
        set(required_names)
        - set(weights)
    )

    extra = (
        set(weights)
        - set(required_names)
    )

    if missing:
        raise KeyError(
            f"Missing weights: {sorted(missing)}"
        )

    if extra:
        raise KeyError(
            f"Unexpected weights: {sorted(extra)}"
        )

    validated = {
        name: float(
            weights[name]
        )
        for name in required_names
    }

    if not all(
        np.isfinite(value)
        for value in validated.values()
    ):
        raise ValueError(
            "All weights must be finite."
        )

    if (
        require_nonnegative
        and any(
            value < 0
            for value in validated.values()
        )
    ):
        raise ValueError(
            "Weights must be nonnegative."
        )

    total_weight = float(
        sum(
            validated.values()
        )
    )

    if total_weight <= tolerance:
        raise ValueError(
            "The total weight must be greater than zero."
        )

    if normalize:
        validated = {
            name: value / total_weight
            for name, value in validated.items()
        }

    return validated


def compute_weighted_composite(
    signals: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
    names: tuple[str, ...],
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
]:
    """
    Compute a weighted sum and its individual contributions.

    Parameters
    ----------
    signals:
        Mapping from signal name to one-dimensional numerical array.
    weights:
        Validated weight mapping.
    names:
        Ordered signal names included in the composite.

    Returns
    -------
    composite:
        Weighted sum across signals.
    contributions:
        Weighted contribution from each signal.
    """
    first_name = names[0]

    expected_length = np.asarray(
        signals[first_name]
    ).size

    contributions: dict[
        str,
        np.ndarray,
    ] = {}

    composite = np.zeros(
        expected_length,
        dtype=np.float32,
    )

    for name in names:
        signal = _validate_one_dimensional_signal(
            signals[name],
            name=name,
            expected_length=expected_length,
        )

        contribution = (
            float(weights[name])
            * signal
        ).astype(np.float32)

        contributions[name] = contribution

        composite += contribution

    return (
        composite.astype(np.float32),
        contributions,
    )


def compute_individual_score(
    standardized_features: Mapping[
        str,
        np.ndarray,
    ],
    *,
    feature_weights: Mapping[
        str,
        float,
    ] | None = None,
    normalize_weights: bool = True,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, float],
]:
    """
    Compute a direct weighted score from all five standardized features.

    The score is

        score =
            w_verticality * verticality
            + w_persistence * persistence
            + w_continuity * continuity
            + w_amplitude * amplitude
            + w_heterogeneity * heterogeneity

    This mode treats each feature as a separate contribution.
    """
    resolved_weights = validate_weights(
        (
            DEFAULT_INDIVIDUAL_WEIGHTS
            if feature_weights is None
            else feature_weights
        ),
        FEATURE_NAMES,
        normalize=normalize_weights,
    )

    score, contributions = (
        compute_weighted_composite(
            signals=standardized_features,
            weights=resolved_weights,
            names=FEATURE_NAMES,
        )
    )

    return (
        score,
        contributions,
        resolved_weights,
    )


def compute_grouped_score(
    standardized_features: Mapping[
        str,
        np.ndarray,
    ],
    *,
    structural_weights: Mapping[
        str,
        float,
    ] | None = None,
    hypertransmission_weights: Mapping[
        str,
        float,
    ] | None = None,
    group_weights: Mapping[
        str,
        float,
    ] | None = None,
    normalize_weights: bool = True,
) -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, Any],
]:
    """
    Compute a grouped barcode score.

    Correlated features are first combined into two interpretable groups.

    Structural score
    ----------------
    Combines:

    - verticality;
    - continuity;
    - heterogeneity.

    These features describe organized vertical texture and variation.

    Hypertransmission score
    -----------------------
    Combines:

    - persistence;
    - amplitude.

    These features describe bright signal extending through depth.

    The final score is

        score =
            w_structural * structural_score
            + w_hypertransmission * hypertransmission_score

    Grouping helps avoid unintentionally counting several highly
    correlated structural features as fully independent evidence.
    """
    resolved_structural_weights = (
        validate_weights(
            (
                DEFAULT_STRUCTURAL_WEIGHTS
                if structural_weights is None
                else structural_weights
            ),
            STRUCTURAL_FEATURES,
            normalize=normalize_weights,
        )
    )

    resolved_hypertransmission_weights = (
        validate_weights(
            (
                DEFAULT_HYPERTRANSMISSION_WEIGHTS
                if hypertransmission_weights is None
                else hypertransmission_weights
            ),
            HYPERTRANSMISSION_FEATURES,
            normalize=normalize_weights,
        )
    )

    resolved_group_weights = (
        validate_weights(
            (
                DEFAULT_GROUP_WEIGHTS
                if group_weights is None
                else group_weights
            ),
            (
                "structural",
                "hypertransmission",
            ),
            normalize=normalize_weights,
        )
    )

    structural_score, structural_contributions = (
        compute_weighted_composite(
            signals=standardized_features,
            weights=resolved_structural_weights,
            names=STRUCTURAL_FEATURES,
        )
    )

    (
        hypertransmission_score,
        hypertransmission_contributions,
    ) = compute_weighted_composite(
        signals=standardized_features,
        weights=(
            resolved_hypertransmission_weights
        ),
        names=(
            HYPERTRANSMISSION_FEATURES
        ),
    )

    group_scores = {
        "structural": structural_score,
        "hypertransmission": (
            hypertransmission_score
        ),
    }

    final_score, group_contributions = (
        compute_weighted_composite(
            signals=group_scores,
            weights=resolved_group_weights,
            names=(
                "structural",
                "hypertransmission",
            ),
        )
    )

    feature_contributions: dict[
        str,
        np.ndarray,
    ] = {}

    for feature_name, contribution in (
        structural_contributions.items()
    ):
        feature_contributions[
            feature_name
        ] = (
            resolved_group_weights[
                "structural"
            ]
            * contribution
        ).astype(np.float32)

    for feature_name, contribution in (
        hypertransmission_contributions.items()
    ):
        feature_contributions[
            feature_name
        ] = (
            resolved_group_weights[
                "hypertransmission"
            ]
            * contribution
        ).astype(np.float32)

    resolved_weights = {
        "structural_features": (
            resolved_structural_weights
        ),
        "hypertransmission_features": (
            resolved_hypertransmission_weights
        ),
        "groups": resolved_group_weights,
        "group_contributions": (
            group_contributions
        ),
    }

    return (
        final_score,
        feature_contributions,
        group_scores,
        resolved_weights,
    )


def threshold_score(
    score: np.ndarray,
    threshold: float,
    *,
    structural_score: np.ndarray | None = None,
    structural_threshold: float | None = None,
    hypertransmission_score: np.ndarray | None = None,
    hypertransmission_threshold: float | None = None,
) -> np.ndarray:
    """
    Convert a continuous barcode score into a boolean candidate mask.

    The combined score threshold is always applied:

        score > threshold

    Optional group-level gates may also be applied:

        structural_score > structural_threshold

        hypertransmission_score > hypertransmission_threshold

    When both gates are supplied, a position must satisfy all three
    requirements to be marked as a raw candidate.

    Parameters
    ----------
    score:
        Final combined barcode score.
    threshold:
        Minimum required combined score.
    structural_score:
        Optional structural group score.
    structural_threshold:
        Optional minimum structural score.
    hypertransmission_score:
        Optional hypertransmission group score.
    hypertransmission_threshold:
        Optional minimum hypertransmission score.

    Returns
    -------
    np.ndarray
        Boolean raw detection mask.
    """
    score = _validate_one_dimensional_signal(
        score,
        name="score",
    )

    if not np.isfinite(threshold):
        raise ValueError(
            "threshold must be finite."
        )

    raw_mask = (
        score > float(threshold)
    )

    if structural_threshold is not None:
        if structural_score is None:
            raise ValueError(
                "structural_score must be supplied when "
                "structural_threshold is used."
            )

        if not np.isfinite(
            structural_threshold
        ):
            raise ValueError(
                "structural_threshold must be finite."
            )

        validated_structural_score = (
            _validate_one_dimensional_signal(
                structural_score,
                name="structural_score",
                expected_length=score.size,
            )
        )

        raw_mask &= (
            validated_structural_score
            > float(structural_threshold)
        )

    elif structural_score is not None:
        raise ValueError(
            "structural_threshold must be supplied when "
            "structural_score is supplied."
        )

    if hypertransmission_threshold is not None:
        if hypertransmission_score is None:
            raise ValueError(
                "hypertransmission_score must be supplied when "
                "hypertransmission_threshold is used."
            )

        if not np.isfinite(
            hypertransmission_threshold
        ):
            raise ValueError(
                "hypertransmission_threshold must be finite."
            )

        validated_hypertransmission_score = (
            _validate_one_dimensional_signal(
                hypertransmission_score,
                name="hypertransmission_score",
                expected_length=score.size,
            )
        )

        raw_mask &= (
            validated_hypertransmission_score
            > float(
                hypertransmission_threshold
            )
        )

    elif hypertransmission_score is not None:
        raise ValueError(
            "hypertransmission_threshold must be supplied when "
            "hypertransmission_score is supplied."
        )

    return raw_mask.astype(bool)


def _find_boolean_runs(
    mask: np.ndarray,
    value: bool,
) -> list[tuple[int, int]]:
    """
    Return inclusive index ranges for consecutive occurrences of a value.
    """
    mask = np.asarray(
        mask,
        dtype=bool,
    )

    runs: list[
        tuple[int, int]
    ] = []

    start_index: int | None = None

    for index, current_value in enumerate(
        mask
    ):
        if bool(current_value) == value:
            if start_index is None:
                start_index = index

        elif start_index is not None:
            runs.append(
                (
                    start_index,
                    index - 1,
                )
            )

            start_index = None

    if start_index is not None:
        runs.append(
            (
                start_index,
                mask.size - 1,
            )
        )

    return runs


def remove_short_positive_runs(
    mask: np.ndarray,
    minimum_run_length: int,
) -> np.ndarray:
    """
    Remove positive runs shorter than the selected number of positions.
    """
    mask = np.asarray(
        mask,
        dtype=bool,
    ).copy()

    if minimum_run_length <= 0:
        raise ValueError(
            "minimum_run_length must be positive."
        )

    for start_index, end_index in _find_boolean_runs(
        mask,
        True,
    ):
        run_length = (
            end_index
            - start_index
            + 1
        )

        if run_length < minimum_run_length:
            mask[
                start_index:
                end_index + 1
            ] = False

    return mask


def fill_short_negative_gaps(
    mask: np.ndarray,
    maximum_gap_length: int,
) -> np.ndarray:
    """
    Fill short internal negative gaps between positive detections.

    Gaps touching the left or right edge are not filled because they are
    not bounded by positive detections on both sides.
    """
    mask = np.asarray(
        mask,
        dtype=bool,
    ).copy()

    if maximum_gap_length < 0:
        raise ValueError(
            "maximum_gap_length must be nonnegative."
        )

    if maximum_gap_length == 0:
        return mask

    for start_index, end_index in _find_boolean_runs(
        mask,
        False,
    ):
        touches_left_edge = (
            start_index == 0
        )

        touches_right_edge = (
            end_index == mask.size - 1
        )

        if (
            touches_left_edge
            or touches_right_edge
        ):
            continue

        gap_length = (
            end_index
            - start_index
            + 1
        )

        if gap_length <= maximum_gap_length:
            mask[
                start_index:
                end_index + 1
            ] = True

    return mask


def exclude_edge_positions(
    mask: np.ndarray,
    edge_margin: int,
) -> np.ndarray:
    """
    Prevent detections inside a selected number of edge positions.
    """
    mask = np.asarray(
        mask,
        dtype=bool,
    ).copy()

    if edge_margin < 0:
        raise ValueError(
            "edge_margin must be nonnegative."
        )

    if edge_margin == 0:
        return mask

    if 2 * edge_margin >= mask.size:
        raise ValueError(
            "edge_margin removes the entire score signal."
        )

    mask[
        :edge_margin
    ] = False

    mask[
        -edge_margin:
    ] = False

    return mask


def clean_detection_mask(
    raw_mask: np.ndarray,
    *,
    minimum_interval_length: int = 8,
    maximum_gap_length: int = 4,
    edge_margin: int = 0,
    cleanup_order: str = "fill_then_remove",
) -> np.ndarray:
    """
    Clean a one-dimensional detection mask.

    Parameters
    ----------
    raw_mask:
        Boolean mask created by thresholding.
    minimum_interval_length:
        Positive runs shorter than this number of evaluated positions are
        removed.
    maximum_gap_length:
        Internal negative gaps of this length or smaller are filled.
    edge_margin:
        Number of evaluated positions excluded from each horizontal edge.
    cleanup_order:
        Either:

        ``"fill_then_remove"``
            Merge nearby candidate regions before removing short runs.

        ``"remove_then_fill"``
            Remove short runs before filling internal gaps.
    """
    cleanup_order = (
        cleanup_order.lower()
    )

    if cleanup_order not in {
        "fill_then_remove",
        "remove_then_fill",
    }:
        raise ValueError(
            "cleanup_order must be 'fill_then_remove' "
            "or 'remove_then_fill'."
        )

    cleaned = exclude_edge_positions(
        raw_mask,
        edge_margin=edge_margin,
    )

    if cleanup_order == "fill_then_remove":
        cleaned = fill_short_negative_gaps(
            cleaned,
            maximum_gap_length=(
                maximum_gap_length
            ),
        )

        cleaned = remove_short_positive_runs(
            cleaned,
            minimum_run_length=(
                minimum_interval_length
            ),
        )

    else:
        cleaned = remove_short_positive_runs(
            cleaned,
            minimum_run_length=(
                minimum_interval_length
            ),
        )

        cleaned = fill_short_negative_gaps(
            cleaned,
            maximum_gap_length=(
                maximum_gap_length
            ),
        )

    return cleaned


def extract_detection_intervals(
    x_positions: np.ndarray,
    cleaned_mask: np.ndarray,
    score: np.ndarray,
) -> list[DetectionInterval]:
    """
    Convert a cleaned boolean mask into contiguous horizontal intervals.
    """
    x_positions = _validate_one_dimensional_signal(
        x_positions,
        name="x_positions",
    )

    score = _validate_one_dimensional_signal(
        score,
        name="score",
        expected_length=x_positions.size,
    )

    cleaned_mask = np.asarray(
        cleaned_mask,
        dtype=bool,
    )

    if cleaned_mask.ndim != 1:
        raise ValueError(
            "cleaned_mask must be one-dimensional."
        )

    if cleaned_mask.size != x_positions.size:
        raise ValueError(
            "cleaned_mask must have the same length as x_positions."
        )

    intervals: list[
        DetectionInterval
    ] = []

    if x_positions.size > 1:
        typical_spacing = float(
            np.median(
                np.diff(
                    x_positions
                )
            )
        )
    else:
        typical_spacing = 1.0

    for start_index, end_index in _find_boolean_runs(
        cleaned_mask,
        True,
    ):
        interval_scores = score[
            start_index:
            end_index + 1
        ]

        x_start = float(
            x_positions[
                start_index
            ]
        )

        x_end = float(
            x_positions[
                end_index
            ]
        )

        width_pixels = float(
            x_end
            - x_start
            + typical_spacing
        )

        intervals.append(
            DetectionInterval(
                x_start=x_start,
                x_end=x_end,
                width_pixels=width_pixels,
                start_index=int(
                    start_index
                ),
                end_index=int(
                    end_index
                ),
                number_of_positions=int(
                    end_index
                    - start_index
                    + 1
                ),
                mean_score=float(
                    interval_scores.mean()
                ),
                maximum_score=float(
                    interval_scores.max()
                ),
                minimum_score=float(
                    interval_scores.min()
                ),
            )
        )

    return intervals


def detect_barcoding(
    feature_result: Any,
    *,
    scoring_mode: str = "grouped",
    threshold: float = 1.0,
    structural_threshold: float | None = None,
    hypertransmission_threshold: float | None = None,
    feature_weights: Mapping[
        str,
        float,
    ] | None = None,
    structural_weights: Mapping[
        str,
        float,
    ] | None = None,
    hypertransmission_weights: Mapping[
        str,
        float,
    ] | None = None,
    group_weights: Mapping[
        str,
        float,
    ] | None = None,
    normalize_weights: bool = True,
    minimum_interval_length: int = 8,
    maximum_gap_length: int = 4,
    edge_margin: int = 10,
    cleanup_order: str = "fill_then_remove",
) -> DetectionResult:
    """
    Run the complete one-dimensional barcode detector.

    Parameters
    ----------
    feature_result:
        FeatureSignals object containing standardized feature signals.
    scoring_mode:
        Scoring approach:

        ``"individual"``
            Combine all five features directly using individual weights.

        ``"grouped"``
            First combine verticality, continuity, and heterogeneity into
            a structural score; combine persistence and amplitude into a
            hypertransmission score; then combine the two group scores.

    threshold:
        Barcode-score threshold. Positions satisfying
        ``score > threshold`` are initially marked positive.
    structural_threshold:
        Optional minimum structural group score. When supplied in grouped
        mode, a position must exceed this value in addition to the
        combined threshold.

    hypertransmission_threshold:
        Optional minimum hypertransmission group score. When supplied in
        grouped mode, a position must exceed this value in addition to
        the combined threshold.
    feature_weights:
        Individual feature weights used only when
        ``scoring_mode="individual"``.
    structural_weights:
        Within-group weights for verticality, continuity, and
        heterogeneity.
    hypertransmission_weights:
        Within-group weights for persistence and amplitude.
    group_weights:
        Final weights assigned to structural and hypertransmission
        group scores.
    normalize_weights:
        Whether each supplied weight set should be normalized to sum to
        one.
    minimum_interval_length:
        Minimum retained positive run length in evaluated positions.
    maximum_gap_length:
        Maximum internal negative gap that may be filled.
    edge_margin:
        Number of evaluated positions excluded from each image edge.
        With a 21-pixel window and stride 1, a value of 10 removes the
        half-window edge region.
    cleanup_order:
        Either ``"fill_then_remove"`` or ``"remove_then_fill"``.
    Notes
    -----
    Group thresholds are available only when ``scoring_mode="grouped"``.
    They prevent one feature group from compensating completely for weak
    evidence in the other group.

    Returns
    -------
    DetectionResult
        Score, masks, intervals, contributions, and metadata.

    """
    (
        x_positions,
        standardized_features,
    ) = validate_feature_result(
        feature_result
    )

    scoring_mode = (
        scoring_mode.lower()
    )
    group_thresholding_requested = (
        structural_threshold is not None
        or hypertransmission_threshold is not None
    )

    if (
        scoring_mode == "individual"
        and group_thresholding_requested
    ):
        raise ValueError(
            "structural_threshold and "
            "hypertransmission_threshold may only be used when "
            "scoring_mode='grouped'."
        )

    if scoring_mode == "individual":
        (
            score,
            feature_contributions,
            resolved_feature_weights,
        ) = compute_individual_score(
            standardized_features,
            feature_weights=feature_weights,
            normalize_weights=(
                normalize_weights
            ),
        )

        group_scores: dict[
            str,
            np.ndarray,
        ] = {}

        resolved_weights: dict[
            str,
            Any,
        ] = {
            "features": (
                resolved_feature_weights
            ),
        }

    elif scoring_mode == "grouped":
        (
            score,
            feature_contributions,
            group_scores,
            resolved_weights,
        ) = compute_grouped_score(
            standardized_features,
            structural_weights=(
                structural_weights
            ),
            hypertransmission_weights=(
                hypertransmission_weights
            ),
            group_weights=group_weights,
            normalize_weights=(
                normalize_weights
            ),
        )

    else:
        raise ValueError(
            "scoring_mode must be either "
            "'individual' or 'grouped'."
        )

    if scoring_mode == "grouped":
        structural_score_for_gate = (
            group_scores["structural"]
            if structural_threshold is not None
            else None
        )

        hypertransmission_score_for_gate = (
            group_scores["hypertransmission"]
            if hypertransmission_threshold is not None
            else None
        )

    else:
        structural_score_for_gate = None
        hypertransmission_score_for_gate = None

    raw_mask = threshold_score(
        score=score,
        threshold=threshold,
        structural_score=(
            structural_score_for_gate
        ),
        structural_threshold=(
            structural_threshold
        ),
        hypertransmission_score=(
            hypertransmission_score_for_gate
        ),
        hypertransmission_threshold=(
            hypertransmission_threshold
        ),
    )

    cleaned_mask = clean_detection_mask(
        raw_mask=raw_mask,
        minimum_interval_length=(
            minimum_interval_length
        ),
        maximum_gap_length=(
            maximum_gap_length
        ),
        edge_margin=edge_margin,
        cleanup_order=cleanup_order,
    )

    intervals = extract_detection_intervals(
        x_positions=x_positions,
        cleaned_mask=cleaned_mask,
        score=score,
    )

    metadata = {
        "scoring_mode": scoring_mode,
        "threshold": float(
            threshold
        ),
        "group_thresholding": {
            "enabled": bool(
                group_thresholding_requested
            ),
            "structural_threshold": (
                float(
                    structural_threshold
                )
                if structural_threshold
                is not None
                else None
            ),
            "hypertransmission_threshold": (
                float(
                    hypertransmission_threshold
                )
                if hypertransmission_threshold
                is not None
                else None
            ),
        },
        "normalize_weights": bool(
            normalize_weights
        ),
        "weights": resolved_weights,
        "cleanup": {
            "minimum_interval_length": int(
                minimum_interval_length
            ),
            "maximum_gap_length": int(
                maximum_gap_length
            ),
            "edge_margin": int(
                edge_margin
            ),
            "cleanup_order": (
                cleanup_order.lower()
            ),
        },
        "number_of_positions": int(
            x_positions.size
        ),
        "raw_positive_positions": int(
            raw_mask.sum()
        ),
        "cleaned_positive_positions": int(
            cleaned_mask.sum()
        ),
        "number_of_intervals": int(
            len(
                intervals
            )
        ),
        "score_minimum": float(
            score.min()
        ),
        "score_maximum": float(
            score.max()
        ),
        "score_mean": float(
            score.mean()
        ),
        "score_standard_deviation": float(
            score.std()
        ),
    }

    if hasattr(
        feature_result,
        "metadata",
    ):
        feature_metadata = getattr(
            feature_result,
            "metadata",
        )

        if isinstance(
            feature_metadata,
            Mapping,
        ):
            window_metadata = (
                feature_metadata.get(
                    "window",
                    {},
                )
            )

            metadata[
                "feature_window"
            ] = dict(
                window_metadata
            )

    return DetectionResult(
        x_positions=x_positions,
        score=score.astype(
            np.float32
        ),
        raw_mask=raw_mask.astype(
            bool
        ),
        cleaned_mask=cleaned_mask.astype(
            bool
        ),
        intervals=intervals,
        scoring_mode=scoring_mode,
        threshold=float(
            threshold
        ),
        feature_contributions=(
            feature_contributions
        ),
        group_scores=group_scores,
        metadata=metadata,
    )

@dataclass(frozen=True)
class CandidateInterval:
    """
    One contiguous horizontal candidate barcode interval.

    Attributes
    ----------
    start:
        Inclusive leftmost image column.
    end:
        Inclusive rightmost image column.
    width:
        Number of detected image columns in the interval.
    """

    start: int
    end: int
    width: int

@dataclass
class StructuralHyperTDDetectorResult:
    """
    Complete output from the structural-hypertransmission detector.

    The result contains all intermediate feature representations so that
    individual detector stages can later be inspected without rerunning
    the pipeline.
    """

    image_shape: tuple[int, int]

    verticality_map: np.ndarray
    gradient_magnitude: np.ndarray
    structural_mask: np.ndarray

    column_median: np.ndarray
    column_q90: np.ndarray

    continuity: np.ndarray
    strong_vertical_fraction: np.ndarray

    intensity_mask: np.ndarray
    continuity_mask: np.ndarray
    barcode_mask: np.ndarray

    intervals: list[CandidateInterval]

    thresholds: dict[str, float]

    metadata: dict[str, Any]

def find_boolean_runs(
    mask: np.ndarray,
    *,
    target_value: bool = True,
) -> list[tuple[int, int]]:
    """
    Return inclusive start/end coordinates for contiguous Boolean runs.
    """
    mask = np.asarray(
        mask,
        dtype=bool,
    )

    if mask.ndim != 1:
        raise ValueError(
            "mask must be one-dimensional."
        )

    runs: list[
        tuple[int, int]
    ] = []

    run_start: int | None = None

    for position, value in enumerate(
        mask
    ):
        if bool(value) == bool(
            target_value
        ):
            if run_start is None:
                run_start = position

        elif run_start is not None:
            runs.append(
                (
                    run_start,
                    position - 1,
                )
            )

            run_start = None

    if run_start is not None:
        runs.append(
            (
                run_start,
                mask.size - 1,
            )
        )

    return runs

def clean_column_mask(
    mask: np.ndarray,
    *,
    minimum_positive_run: int = 5,
    maximum_negative_gap: int = 2,
) -> np.ndarray:
    """
    Fill short gaps and remove short positive detector runs.

    Gap filling occurs before removal of short positive runs.
    """
    cleaned = np.asarray(
        mask,
        dtype=bool,
    ).copy()

    if cleaned.ndim != 1:
        raise ValueError(
            "mask must be one-dimensional."
        )

    if minimum_positive_run < 1:
        raise ValueError(
            "minimum_positive_run must be positive."
        )

    if maximum_negative_gap < 0:
        raise ValueError(
            "maximum_negative_gap must be nonnegative."
        )

    for start, end in find_boolean_runs(
        cleaned,
        target_value=False,
    ):
        gap_length = (
            end - start + 1
        )

        touches_edge = (
            start == 0
            or end == cleaned.size - 1
        )

        if (
            not touches_edge
            and gap_length
            <= maximum_negative_gap
        ):
            cleaned[
                start:
                end + 1
            ] = True

    for start, end in find_boolean_runs(
        cleaned,
        target_value=True,
    ):
        run_length = (
            end - start + 1
        )

        if (
            run_length
            < minimum_positive_run
        ):
            cleaned[
                start:
                end + 1
            ] = False

    return cleaned

def extract_candidate_intervals(
    mask: np.ndarray,
) -> list[CandidateInterval]:
    """
    Convert a final detector mask into contiguous candidate intervals.
    """
    intervals: list[
        CandidateInterval
    ] = []

    for start, end in find_boolean_runs(
        mask,
        target_value=True,
    ):
        intervals.append(
            CandidateInterval(
                start=int(start),
                end=int(end),
                width=int(
                    end - start + 1
                ),
            )
        )

    return intervals

def detect_structural_hypertransmission(
    image: np.ndarray,
    *,
    config: dict[str, Any] | None = None,
) -> StructuralHyperTDDetectorResult:
    """
    Detect candidate barcoding regions from a preprocessed OCT scan.

    The detector follows four stages:

    1. Construct a gradient-gated structural exclusion mask.
    2. Identify intensity-based hypertransmission candidates.
    3. Refine candidates using depth continuity.
    4. Refine persistent hypertransmission using vertical organization.

    Parameters
    ----------
    image:
        Two-dimensional normalized and denoised sub-layer OCT image.
    config:
        Optional detector configuration. If omitted, the frozen
        ``STRUCTURAL_HYPERTD_V1_CONFIG`` configuration is used.

    Returns
    -------
    StructuralHyperTDDetectorResult
        Complete detector output including feature maps, masks,
        thresholds, intervals, and metadata.
    """
    image = np.asarray(
        image,
        dtype=np.float32,
    )

    if image.ndim != 2:
        raise ValueError(
            "image must be two-dimensional."
        )

    if image.size == 0:
        raise ValueError(
            "image must not be empty."
        )

    if not np.isfinite(
        image
    ).all():
        raise ValueError(
            "image contains non-finite values."
        )

    resolved_config = dict(
        STRUCTURAL_HYPERTD_V1_CONFIG
    )

    if config is not None:
        resolved_config.update(
            config
        )

    image_height, image_width = (
        image.shape
    )

    edge_margin = int(
        resolved_config[
            "edge_margin"
        ]
    )

    if (
        edge_margin < 0
        or 2 * edge_margin >= image_width
    ):
        raise ValueError(
            "edge_margin is invalid for the supplied image width."
        )

    # ==========================================================
    # 1. Pixel-level verticality and structural exclusion
    # ==========================================================

    (
        verticality_map,
        verticality_diagnostics,
    ) = compute_simple_verticality_map(
        image=image,
        smoothing_sigma=float(
            resolved_config[
                "verticality_smoothing_sigma"
            ]
        ),
    )

    gradient_magnitude = np.hypot(
        verticality_diagnostics[
            "gradient_x"
        ],
        verticality_diagnostics[
            "gradient_z"
        ],
    ).astype(np.float32)

    finite_gradient_values = (
        gradient_magnitude[
            np.isfinite(
                gradient_magnitude
            )
        ]
    )

    gradient_threshold = float(
        np.quantile(
            finite_gradient_values,
            float(
                resolved_config[
                    "gradient_quantile"
                ]
            ),
        )
    )

    (
        structural_mask,
        structural_metadata,
    ) = create_structural_mask(
        verticality_map=(
            verticality_map
        ),
        gradient_magnitude=(
            gradient_magnitude
        ),
        verticality_threshold=float(
            resolved_config[
                "verticality_threshold"
            ]
        ),
        minimum_gradient_magnitude=(
            gradient_threshold
        ),
        minimum_component_size=int(
            resolved_config[
                "minimum_component_size"
            ]
        ),
    )

    # ==========================================================
    # 2. Structurally cleaned column intensity
    # ==========================================================

    (
        column_statistics,
        column_metadata,
    ) = compute_column_intensity_statistics(
        image=image,
        exclusion_mask=(
            structural_mask
        ),
        upper_quantile=float(
            resolved_config[
                "column_upper_quantile"
            ]
        ),
        minimum_valid_pixels=int(
            resolved_config[
                "minimum_valid_pixels"
            ]
        ),
    )

    smoothing_sigma = float(
        resolved_config[
            "signal_smoothing_sigma"
        ]
    )

    column_median = smooth_finite_signal(
        column_statistics[
            "median"
        ],
        sigma=smoothing_sigma,
    )

    column_q90 = smooth_finite_signal(
        column_statistics[
            "upper_quantile"
        ],
        sigma=smoothing_sigma,
    )

    reference_columns = np.ones(
        image_width,
        dtype=bool,
    )

    if edge_margin > 0:
        reference_columns[
            :edge_margin
        ] = False

        reference_columns[
            -edge_margin:
        ] = False

    median_reference = (
        column_median[
            reference_columns
        ]
    )

    q90_reference = (
        column_q90[
            reference_columns
        ]
    )

    median_reference_center = float(
        np.median(
            median_reference
        )
    )

    median_reference_iqr = float(
        np.quantile(
            median_reference,
            0.75,
        )
        - np.quantile(
            median_reference,
            0.25,
        )
    )

    q90_reference_center = float(
        np.median(
            q90_reference
        )
    )

    q90_reference_iqr = float(
        np.quantile(
            q90_reference,
            0.75,
        )
        - np.quantile(
            q90_reference,
            0.25,
        )
    )

    median_threshold = (
        median_reference_center
        + float(
            resolved_config[
                "median_iqr_multiplier"
            ]
        )
        * median_reference_iqr
    )

    q90_threshold = (
        q90_reference_center
        + float(
            resolved_config[
                "q90_iqr_multiplier"
            ]
        )
        * q90_reference_iqr
    )

    intensity_mask = (
        (
            column_median
            > median_threshold
        )
        &
        (
            column_q90
            > q90_threshold
        )
    )

    if edge_margin > 0:
        intensity_mask[
            :edge_margin
        ] = False

        intensity_mask[
            -edge_margin:
        ] = False

    intensity_mask = clean_column_mask(
        intensity_mask,
        minimum_positive_run=int(
            resolved_config[
                "minimum_positive_run"
            ]
        ),
        maximum_negative_gap=int(
            resolved_config[
                "maximum_negative_gap"
            ]
        ),
    )

    # ==========================================================
    # 3. Depth continuity
    # ==========================================================

    continuity_raw = (
        compute_local_depth_continuity(
            image=image,
            window_width=int(
                resolved_config[
                    "continuity_window_width"
                ]
            ),
            depth_lag=int(
                resolved_config[
                    "continuity_depth_lag"
                ]
            ),
            minimum_row_standard_deviation=float(
                resolved_config[
                    "continuity_minimum_row_standard_deviation"
                ]
            ),
        )
    )

    continuity = smooth_finite_signal(
        continuity_raw,
        sigma=smoothing_sigma,
    )

    continuity_threshold = float(
        np.quantile(
            continuity[
                reference_columns
            ],
            float(
                resolved_config[
                    "continuity_quantile"
                ]
            ),
        )
    )

    continuity_mask = (
        intensity_mask
        &
        (
            continuity
            > continuity_threshold
        )
    )

    if edge_margin > 0:
        continuity_mask[
            :edge_margin
        ] = False

        continuity_mask[
            -edge_margin:
        ] = False

    continuity_mask = clean_column_mask(
        continuity_mask,
        minimum_positive_run=int(
            resolved_config[
                "minimum_positive_run"
            ]
        ),
        maximum_negative_gap=int(
            resolved_config[
                "maximum_negative_gap"
            ]
        ),
    )

    # ==========================================================
    # 4. Vertical-organization refinement
    # ==========================================================

    verticality_statistics = (
        compute_column_verticality_statistics(
            verticality_map=(
                verticality_map
            ),
            structural_mask=(
                structural_mask
            ),
            upper_quantile=0.90,
        )
    )

    strong_vertical_fraction = (
        smooth_finite_signal(
            verticality_statistics[
                "strong_vertical_fraction"
            ],
            sigma=smoothing_sigma,
        )
    )

    vertical_fraction_threshold = float(
        np.quantile(
            strong_vertical_fraction[
                reference_columns
            ],
            float(
                resolved_config[
                    "vertical_fraction_quantile"
                ]
            ),
        )
    )

    barcode_mask = (
        continuity_mask
        &
        (
            strong_vertical_fraction
            > vertical_fraction_threshold
        )
    )

    if edge_margin > 0:
        barcode_mask[
            :edge_margin
        ] = False

        barcode_mask[
            -edge_margin:
        ] = False

    barcode_mask = clean_column_mask(
        barcode_mask,
        minimum_positive_run=int(
            resolved_config[
                "minimum_positive_run"
            ]
        ),
        maximum_negative_gap=int(
            resolved_config[
                "maximum_negative_gap"
            ]
        ),
    )

    intervals = (
        extract_candidate_intervals(
            barcode_mask
        )
    )

    thresholds = {
        "gradient": (
            gradient_threshold
        ),
        "column_median": (
            median_threshold
        ),
        "column_q90": (
            q90_threshold
        ),
        "continuity": (
            continuity_threshold
        ),
        "vertical_fraction": (
            vertical_fraction_threshold
        ),
    }

    metadata = {
        "detector_version": (
            "structural_hypertd_v1"
        ),
        "image_shape": tuple(
            int(value)
            for value in image.shape
        ),
        "config": dict(
            resolved_config
        ),
        "number_of_intervals": int(
            len(intervals)
        ),
        "total_detected_columns": int(
            barcode_mask.sum()
        ),
        "structural_pixel_fraction": float(
            structural_mask.mean()
        ),
        "intensity_candidate_columns": int(
            intensity_mask.sum()
        ),
        "continuity_candidate_columns": int(
            continuity_mask.sum()
        ),
        "final_candidate_columns": int(
            barcode_mask.sum()
        ),
        "structural_metadata": (
            structural_metadata
        ),
        "column_metadata": (
            column_metadata
        ),
    }

    return StructuralHyperTDDetectorResult(
        image_shape=(
            image_height,
            image_width,
        ),
        verticality_map=(
            verticality_map
        ),
        gradient_magnitude=(
            gradient_magnitude
        ),
        structural_mask=(
            structural_mask
        ),
        column_median=(
            column_median
        ),
        column_q90=(
            column_q90
        ),
        continuity=(
            continuity
        ),
        strong_vertical_fraction=(
            strong_vertical_fraction
        ),
        intensity_mask=(
            intensity_mask
        ),
        continuity_mask=(
            continuity_mask
        ),
        barcode_mask=(
            barcode_mask
        ),
        intervals=(
            intervals
        ),
        thresholds=(
            thresholds
        ),
        metadata=(
            metadata
        ),
    )