# features.py
#
# Responsibilities:
# - Accept a preprocessed two-dimensional image.
# - Construct overlapping horizontal sliding windows.
# - Support configurable window width, stride, and padding.
# - Calculate interpretable mathematical feature signals,
#   including verticality, depth persistence, periodicity,
#   amplitude, and spatial heterogeneity.
# - Standardize feature signals before combination.
#
# Inputs:
# - Preprocessed image or path to a preprocessed image.
# - Sliding-window configuration.
# - Feature-specific parameters.
#
# Outputs:
# - One feature value per horizontal location for each feature.
# - Raw and standardized feature signals.
# - Feature-extraction metadata.
# - Measurement DataFrame or CSV.
#
# This module should describe the image mathematically but should
# not apply feature weights, thresholds, or class labels.


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.ndimage import gaussian_filter


FEATURE_NAMES = (
    "verticality",
    "persistence",
    "periodicity",
    "amplitude",
    "heterogeneity",
)


@dataclass
class FeatureSignals:
    """
    Mathematical feature signals extracted from one preprocessed scan.

    Attributes
    ----------
    x_positions:
        Horizontal image positions associated with the feature values.
        With ``stride=1``, this generally contains one value per image
        column.
    raw_features:
        Dictionary containing the original feature signals.
    standardized_features:
        Dictionary containing robustly standardized feature signals.
    verticality_map:
        Two-dimensional map describing locally oriented vertical
        structure.
    persistence_signal:
        One-dimensional column-level depth-persistence signal before
        sliding-window aggregation.
    metadata:
        Feature-extraction configuration and diagnostic information.
    """

    x_positions: np.ndarray

    raw_features: dict[str, np.ndarray]
    standardized_features: dict[str, np.ndarray]

    verticality_map: np.ndarray
    persistence_signal: np.ndarray

    metadata: dict[str, Any]

    def to_dataframe(
        self,
        *,
        include_standardized: bool = True,
    ):
        """
        Convert the feature signals to a pandas DataFrame.

        Parameters
        ----------
        include_standardized:
            Whether standardized feature columns are included.

        Returns
        -------
        pandas.DataFrame
            One row per evaluated horizontal position.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required to create a feature DataFrame."
            ) from exc

        data: dict[str, np.ndarray] = {
            "x_position": self.x_positions,
        }

        for feature_name in FEATURE_NAMES:
            data[feature_name] = self.raw_features[
                feature_name
            ]

        if include_standardized:
            for feature_name in FEATURE_NAMES:
                data[
                    f"{feature_name}_standardized"
                ] = self.standardized_features[
                    feature_name
                ]

        return pd.DataFrame(data)

    def save_csv(
        self,
        output_path: str | Path,
        *,
        include_standardized: bool = True,
        index: bool = False,
    ) -> Path:
        """
        Save the feature signals to a CSV file.
        """
        output_path = Path(output_path)

        if output_path.suffix.lower() != ".csv":
            output_path = output_path.with_suffix(
                ".csv"
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        dataframe = self.to_dataframe(
            include_standardized=include_standardized,
        )

        dataframe.to_csv(
            output_path,
            index=index,
        )

        return output_path.resolve()


def validate_feature_image(
    image: np.ndarray,
) -> np.ndarray:
    """
    Validate a preprocessed image and convert it to float32.
    """
    array = np.asarray(
        image,
        dtype=np.float32,
    )

    if array.ndim != 2:
        raise ValueError(
            "Feature extraction requires a two-dimensional image; "
            f"received shape {array.shape}."
        )

    if array.size == 0:
        raise ValueError(
            "The supplied image is empty."
        )

    if not np.isfinite(array).any():
        raise ValueError(
            "The supplied image contains no finite values."
        )

    if not np.isfinite(array).all():
        finite_values = array[
            np.isfinite(array)
        ]

        replacement_value = float(
            np.median(finite_values)
        )

        array = np.where(
            np.isfinite(array),
            array,
            replacement_value,
        ).astype(np.float32)

    return array


def validate_window_configuration(
    image_width: int,
    window_width: int,
    stride: int,
) -> None:
    """
    Validate sliding-window parameters.
    """
    if image_width <= 0:
        raise ValueError(
            "image_width must be positive."
        )

    if window_width <= 0:
        raise ValueError(
            "window_width must be positive."
        )

    if window_width % 2 == 0:
        raise ValueError(
            "window_width must be odd so every window has "
            "a well-defined center."
        )

    if stride <= 0:
        raise ValueError(
            "stride must be positive."
        )


def pad_image_horizontally(
    image: np.ndarray,
    window_width: int,
    padding_mode: str = "reflect",
    constant_value: float = 0.0,
) -> tuple[np.ndarray, int]:
    """
    Pad an image horizontally so edge columns receive full windows.

    Parameters
    ----------
    image:
        Two-dimensional preprocessed image.
    window_width:
        Odd horizontal window width.
    padding_mode:
        NumPy padding mode, such as ``"reflect"``, ``"edge"``, or
        ``"constant"``.
    constant_value:
        Value used when ``padding_mode="constant"``.

    Returns
    -------
    padded_image:
        Horizontally padded image.
    radius:
        Number of padded columns added to each side.
    """
    radius = window_width // 2

    if radius == 0:
        return image.copy(), radius

    supported_modes = {
        "reflect",
        "symmetric",
        "edge",
        "constant",
        "wrap",
    }

    if padding_mode not in supported_modes:
        raise ValueError(
            "padding_mode must be one of "
            f"{sorted(supported_modes)}."
        )

    if padding_mode == "constant":
        padded = np.pad(
            image,
            pad_width=(
                (0, 0),
                (radius, radius),
            ),
            mode=padding_mode,
            constant_values=constant_value,
        )
    else:
        padded = np.pad(
            image,
            pad_width=(
                (0, 0),
                (radius, radius),
            ),
            mode=padding_mode,
        )

    return (
        padded.astype(np.float32),
        radius,
    )


def get_window_centers(
    image_width: int,
    stride: int,
) -> np.ndarray:
    """
    Return horizontal positions evaluated by the sliding-window scan.
    """
    return np.arange(
        0,
        image_width,
        stride,
        dtype=np.int32,
    )


def extract_window(
    padded_image: np.ndarray,
    center_x: int,
    window_width: int,
) -> np.ndarray:
    """
    Extract one full-depth horizontal window from a padded image.

    The padded image is assumed to contain ``window_width // 2``
    columns of padding on both horizontal sides.
    """
    start = int(center_x)
    end = start + int(window_width)

    window = padded_image[
        :,
        start:end,
    ]

    if window.shape[1] != window_width:
        raise RuntimeError(
            "Sliding-window extraction returned an unexpected width: "
            f"expected {window_width}, received {window.shape[1]}."
        )

    return window


def compute_structure_tensor_verticality(
    image: np.ndarray,
    *,
    gradient_sigma: float = 1.0,
    tensor_sigma: float = 2.0,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Compute a two-dimensional verticality map using a structure tensor.

    The image gradients are

        I_x = horizontal intensity change
        I_z = vertical intensity change

    A vertical line generally produces a strong horizontal gradient,
    because brightness changes when moving across the line but remains
    more stable when moving along its depth.

    The structure tensor contains locally averaged squared gradients:

        J = [[J_xx, J_xz],
             [J_xz, J_zz]]

    Its eigenvalues quantify the strength of directional structure.
    Anisotropy approaches one for strongly oriented patterns and zero
    for isotropic or unstructured regions.

    The verticality map combines:

    - structural anisotropy;
    - agreement with a vertical line orientation.

    Parameters
    ----------
    image:
        Preprocessed two-dimensional scan.
    gradient_sigma:
        Gaussian smoothing applied before calculating gradients.
    tensor_sigma:
        Gaussian smoothing applied to the tensor components.
    epsilon:
        Small positive constant used for numerical stability.

    Returns
    -------
    verticality_map:
        Values generally between 0 and 1. Larger values indicate
        stronger vertically oriented structure.
    components:
        Intermediate tensor and orientation quantities.
    """
    if gradient_sigma < 0:
        raise ValueError(
            "gradient_sigma must be nonnegative."
        )

    if tensor_sigma < 0:
        raise ValueError(
            "tensor_sigma must be nonnegative."
        )

    if epsilon <= 0:
        raise ValueError(
            "epsilon must be positive."
        )

    image = validate_feature_image(
        image
    )

    if gradient_sigma > 0:
        gradient_image = gaussian_filter(
            image,
            sigma=gradient_sigma,
            mode="reflect",
        )
    else:
        gradient_image = image

    gradient_z, gradient_x = np.gradient(
        gradient_image
    )

    tensor_xx = gaussian_filter(
        gradient_x * gradient_x,
        sigma=tensor_sigma,
        mode="reflect",
    )

    tensor_zz = gaussian_filter(
        gradient_z * gradient_z,
        sigma=tensor_sigma,
        mode="reflect",
    )

    tensor_xz = gaussian_filter(
        gradient_x * gradient_z,
        sigma=tensor_sigma,
        mode="reflect",
    )

    trace = tensor_xx + tensor_zz

    discriminant = np.sqrt(
        np.maximum(
            (
                tensor_xx - tensor_zz
            )
            ** 2
            + 4.0
            * tensor_xz
            * tensor_xz,
            0.0,
        )
    )

    lambda_1 = (
        trace + discriminant
    ) / 2.0

    lambda_2 = (
        trace - discriminant
    ) / 2.0

    anisotropy = (
        lambda_1 - lambda_2
    ) / (
        lambda_1 + lambda_2 + epsilon
    )

    # This is the dominant gradient orientation.
    gradient_orientation = (
        0.5
        * np.arctan2(
            2.0 * tensor_xz,
            tensor_xx - tensor_zz,
        )
    )

    # A vertical line has a primarily horizontal gradient.
    # cos(theta)^2 is one for a horizontal gradient and zero for
    # a vertical gradient.
    vertical_orientation_agreement = (
        np.cos(
            gradient_orientation
        )
        ** 2
    )

    verticality_map = (
        anisotropy
        * vertical_orientation_agreement
    )

    verticality_map = np.clip(
        verticality_map,
        0.0,
        1.0,
    ).astype(np.float32)

    components = {
        "gradient_x": np.asarray(
            gradient_x,
            dtype=np.float32,
        ),
        "gradient_z": np.asarray(
            gradient_z,
            dtype=np.float32,
        ),
        "tensor_xx": np.asarray(
            tensor_xx,
            dtype=np.float32,
        ),
        "tensor_zz": np.asarray(
            tensor_zz,
            dtype=np.float32,
        ),
        "tensor_xz": np.asarray(
            tensor_xz,
            dtype=np.float32,
        ),
        "lambda_1": np.asarray(
            lambda_1,
            dtype=np.float32,
        ),
        "lambda_2": np.asarray(
            lambda_2,
            dtype=np.float32,
        ),
        "anisotropy": np.asarray(
            anisotropy,
            dtype=np.float32,
        ),
        "gradient_orientation": np.asarray(
            gradient_orientation,
            dtype=np.float32,
        ),
        "vertical_orientation_agreement": np.asarray(
            vertical_orientation_agreement,
            dtype=np.float32,
        ),
    }

    return verticality_map, components


def longest_true_run(
    values: np.ndarray,
) -> int:
    """
    Return the longest consecutive run of true values.
    """
    values = np.asarray(
        values,
        dtype=bool,
    )

    longest = 0
    current = 0

    for value in values:
        if value:
            current += 1
            longest = max(
                longest,
                current,
            )
        else:
            current = 0

    return int(longest)


def compute_column_persistence(
    image: np.ndarray,
    *,
    bright_percentile: float = 75.0,
    verticality_map: np.ndarray | None = None,
    minimum_verticality: float = 0.0,
    method: str = "longest_run",
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Measure how persistently candidate signal extends through depth.

    A candidate pixel is defined by an intensity threshold and,
    optionally, a minimum verticality requirement.

    Parameters
    ----------
    image:
        Preprocessed two-dimensional scan.
    bright_percentile:
        Whole-image intensity percentile defining candidate bright pixels.
    verticality_map:
        Optional structure-tensor verticality map.
    minimum_verticality:
        Minimum verticality required for a candidate pixel.
    method:
        Persistence definition:

        ``"longest_run"``
            Longest consecutive candidate run divided by image depth.

        ``"depth_fraction"``
            Fraction of the column classified as candidate signal.

    Returns
    -------
    persistence:
        One value per image column.
    metadata:
        Threshold and persistence settings.
    """
    image = validate_feature_image(
        image
    )

    if not (
        0.0
        <= bright_percentile
        <= 100.0
    ):
        raise ValueError(
            "bright_percentile must lie between 0 and 100."
        )

    if not (
        0.0
        <= minimum_verticality
        <= 1.0
    ):
        raise ValueError(
            "minimum_verticality must lie between 0 and 1."
        )

    method = method.lower()

    if method not in {
        "longest_run",
        "depth_fraction",
    }:
        raise ValueError(
            "persistence method must be 'longest_run' "
            "or 'depth_fraction'."
        )

    bright_threshold = float(
        np.percentile(
            image,
            bright_percentile,
        )
    )

    candidate_mask = (
        image >= bright_threshold
    )

    if verticality_map is not None:
        verticality_map = np.asarray(
            verticality_map,
            dtype=np.float32,
        )

        if verticality_map.shape != image.shape:
            raise ValueError(
                "verticality_map must have the same shape as image."
            )

        candidate_mask &= (
            verticality_map
            >= minimum_verticality
        )

    image_depth, image_width = image.shape

    if method == "longest_run":
        persistence = np.array(
            [
                longest_true_run(
                    candidate_mask[:, x]
                )
                / image_depth
                for x in range(
                    image_width
                )
            ],
            dtype=np.float32,
        )

    else:
        persistence = candidate_mask.mean(
            axis=0
        ).astype(np.float32)

    metadata = {
        "method": method,
        "bright_percentile": float(
            bright_percentile
        ),
        "bright_threshold": bright_threshold,
        "minimum_verticality": float(
            minimum_verticality
        ),
        "verticality_required": (
            verticality_map is not None
        ),
    }

    return persistence, metadata


def compute_verticality_feature(
    verticality_window: np.ndarray,
    *,
    summary: str = "mean",
    upper_percentile: float = 90.0,
) -> float:
    """
    Summarize verticality within one sliding window.
    """
    summary = summary.lower()

    if summary == "mean":
        return float(
            np.mean(
                verticality_window
            )
        )

    if summary == "median":
        return float(
            np.median(
                verticality_window
            )
        )

    if summary == "percentile":
        if not (
            0.0
            <= upper_percentile
            <= 100.0
        ):
            raise ValueError(
                "upper_percentile must lie between 0 and 100."
            )

        return float(
            np.percentile(
                verticality_window,
                upper_percentile,
            )
        )

    raise ValueError(
        "verticality summary must be 'mean', 'median', "
        "or 'percentile'."
    )


def compute_persistence_feature(
    persistence_window: np.ndarray,
    *,
    summary: str = "mean",
) -> float:
    """
    Summarize column persistence within one sliding window.
    """
    summary = summary.lower()

    if summary == "mean":
        return float(
            np.mean(
                persistence_window
            )
        )

    if summary == "median":
        return float(
            np.median(
                persistence_window
            )
        )

    if summary == "maximum":
        return float(
            np.max(
                persistence_window
            )
        )

    raise ValueError(
        "persistence summary must be 'mean', 'median', "
        "or 'maximum'."
    )


def compute_periodicity_feature(
    image_window: np.ndarray,
    *,
    minimum_lag: int = 2,
    maximum_lag: int | None = None,
    use_absolute_correlation: bool = False,
    epsilon: float = 1e-8,
) -> tuple[float, int, np.ndarray]:
    """
    Measure repeated horizontal structure using local autocorrelation.

    The two-dimensional window is first collapsed into a column signal
    by averaging intensity over depth. Autocorrelation then compares that
    signal with horizontally shifted copies of itself.

    Parameters
    ----------
    image_window:
        Full-depth local image window.
    minimum_lag:
        Smallest horizontal spacing evaluated.
    maximum_lag:
        Largest horizontal spacing evaluated. If omitted, approximately
        half the window width is used.
    use_absolute_correlation:
        Whether negative and positive correlations are treated equally.
        This may be useful for alternating bright-dark patterns.
    epsilon:
        Numerical-stability constant.

    Returns
    -------
    periodicity_score:
        Strongest autocorrelation over the selected lag range.
    best_lag:
        Lag producing the strongest score.
    correlations:
        Autocorrelation values for all evaluated lags.
    """
    image_window = validate_feature_image(
        image_window
    )

    window_width = image_window.shape[1]

    if maximum_lag is None:
        maximum_lag = max(
            minimum_lag,
            window_width // 2,
        )

    maximum_lag = min(
        int(maximum_lag),
        window_width - 1,
    )

    if minimum_lag <= 0:
        raise ValueError(
            "minimum_lag must be positive."
        )

    if maximum_lag < minimum_lag:
        raise ValueError(
            "maximum_lag must be greater than or equal "
            "to minimum_lag."
        )

    column_signal = image_window.mean(
        axis=0
    ).astype(np.float64)

    centered_signal = (
        column_signal
        - column_signal.mean()
    )

    signal_variance = float(
        np.mean(
            centered_signal
            * centered_signal
        )
    )

    lags = np.arange(
        minimum_lag,
        maximum_lag + 1,
        dtype=np.int32,
    )

    if signal_variance <= epsilon:
        correlations = np.zeros(
            lags.size,
            dtype=np.float32,
        )

        return (
            0.0,
            int(lags[0]),
            correlations,
        )

    correlations = []

    for lag in lags:
        left = centered_signal[
            :-lag
        ]

        right = centered_signal[
            lag:
        ]

        denominator = (
            np.sqrt(
                np.sum(
                    left * left
                )
                * np.sum(
                    right * right
                )
            )
            + epsilon
        )

        correlation = float(
            np.sum(
                left * right
            )
            / denominator
        )

        correlations.append(
            correlation
        )

    correlations_array = np.asarray(
        correlations,
        dtype=np.float32,
    )

    ranking_values = (
        np.abs(
            correlations_array
        )
        if use_absolute_correlation
        else correlations_array
    )

    best_index = int(
        np.argmax(
            ranking_values
        )
    )

    best_score = float(
        ranking_values[
            best_index
        ]
    )

    best_lag = int(
        lags[
            best_index
        ]
    )

    return (
        best_score,
        best_lag,
        correlations_array,
    )


def compute_amplitude_feature(
    image_window: np.ndarray,
    *,
    statistic: str = "mean",
    percentile: float = 90.0,
) -> float:
    """
    Summarize intensity amplitude in one sliding window.

    Supported statistics are ``"mean"``, ``"median"``, and
    ``"percentile"``.
    """
    statistic = statistic.lower()

    if statistic == "mean":
        return float(
            np.mean(
                image_window
            )
        )

    if statistic == "median":
        return float(
            np.median(
                image_window
            )
        )

    if statistic == "percentile":
        if not (
            0.0
            <= percentile
            <= 100.0
        ):
            raise ValueError(
                "amplitude percentile must lie between 0 and 100."
            )

        return float(
            np.percentile(
                image_window,
                percentile,
            )
        )

    raise ValueError(
        "amplitude statistic must be 'mean', 'median', "
        "or 'percentile'."
    )


def compute_heterogeneity_feature(
    image_window: np.ndarray,
    *,
    method: str = "column_std",
    epsilon: float = 1e-8,
) -> float:
    """
    Measure local spatial heterogeneity.

    Supported methods
    -----------------
    ``"column_std"``
        Standard deviation of the mean column intensities. This is useful
        for repeated bright-dark vertical columns.

    ``"pixel_std"``
        Standard deviation across every pixel in the window.

    ``"iqr"``
        Interquartile range of all pixel intensities.

    ``"coefficient_of_variation"``
        Pixel standard deviation divided by absolute pixel mean.
    """
    method = method.lower()

    if method == "column_std":
        column_signal = image_window.mean(
            axis=0
        )

        return float(
            np.std(
                column_signal
            )
        )

    if method == "pixel_std":
        return float(
            np.std(
                image_window
            )
        )

    if method == "iqr":
        first_quartile = float(
            np.percentile(
                image_window,
                25.0,
            )
        )

        third_quartile = float(
            np.percentile(
                image_window,
                75.0,
            )
        )

        return (
            third_quartile
            - first_quartile
        )

    if method == "coefficient_of_variation":
        mean_value = float(
            np.mean(
                image_window
            )
        )

        standard_deviation = float(
            np.std(
                image_window
            )
        )

        return (
            standard_deviation
            / (
                abs(mean_value)
                + epsilon
            )
        )

    raise ValueError(
        "heterogeneity method must be one of: "
        "'column_std', 'pixel_std', 'iqr', or "
        "'coefficient_of_variation'."
    )


def robust_standardize_signal(
    values: np.ndarray,
    *,
    epsilon: float = 1e-8,
    clip_limits: tuple[float, float] | None = None,
) -> tuple[np.ndarray, dict[str, float | None]]:
    """
    Standardize a feature using its median and interquartile range.

    The transformation is

        standardized = (value - median) / (IQR + epsilon)

    where

        IQR = 75th percentile - 25th percentile.
    """
    values = np.asarray(
        values,
        dtype=np.float32,
    )

    if values.ndim != 1:
        raise ValueError(
            "Feature standardization requires a one-dimensional signal."
        )

    if values.size == 0:
        raise ValueError(
            "Cannot standardize an empty signal."
        )

    median_value = float(
        np.median(
            values
        )
    )

    first_quartile = float(
        np.percentile(
            values,
            25.0,
        )
    )

    third_quartile = float(
        np.percentile(
            values,
            75.0,
        )
    )

    interquartile_range = (
        third_quartile
        - first_quartile
    )

    if interquartile_range <= epsilon:
        standardized = np.zeros_like(
            values,
            dtype=np.float32,
        )
    else:
        standardized = (
            values - median_value
        ) / (
            interquartile_range
            + epsilon
        )

    if clip_limits is not None:
        lower_limit, upper_limit = clip_limits

        if lower_limit >= upper_limit:
            raise ValueError(
                "clip_limits must satisfy lower < upper."
            )

        standardized = np.clip(
            standardized,
            lower_limit,
            upper_limit,
        )

    metadata: dict[str, float | None] = {
        "median": median_value,
        "first_quartile": first_quartile,
        "third_quartile": third_quartile,
        "interquartile_range": float(
            interquartile_range
        ),
        "epsilon": float(
            epsilon
        ),
        "clip_lower": (
            float(
                clip_limits[0]
            )
            if clip_limits is not None
            else None
        ),
        "clip_upper": (
            float(
                clip_limits[1]
            )
            if clip_limits is not None
            else None
        ),
    }

    return (
        standardized.astype(np.float32),
        metadata,
    )


def standardize_feature_dictionary(
    raw_features: Mapping[str, np.ndarray],
    *,
    epsilon: float = 1e-8,
    clip_limits: tuple[float, float] | None = (
        -5.0,
        5.0,
    ),
) -> tuple[
    dict[str, np.ndarray],
    dict[str, dict[str, float | None]],
]:
    """
    Robustly standardize every feature signal.
    """
    standardized: dict[
        str,
        np.ndarray,
    ] = {}

    metadata: dict[
        str,
        dict[str, float | None],
    ] = {}

    for feature_name in FEATURE_NAMES:
        if feature_name not in raw_features:
            raise KeyError(
                f"Missing required feature: {feature_name}"
            )

        (
            standardized_signal,
            signal_metadata,
        ) = robust_standardize_signal(
            raw_features[
                feature_name
            ],
            epsilon=epsilon,
            clip_limits=clip_limits,
        )

        standardized[
            feature_name
        ] = standardized_signal

        metadata[
            feature_name
        ] = signal_metadata

    return standardized, metadata


def extract_feature_signals(
    image: np.ndarray,
    *,
    window_width: int = 21,
    stride: int = 1,
    padding_mode: str = "reflect",
    padding_constant_value: float = 0.0,
    verticality_gradient_sigma: float = 1.0,
    verticality_tensor_sigma: float = 2.0,
    verticality_summary: str = "mean",
    verticality_percentile: float = 90.0,
    persistence_bright_percentile: float = 75.0,
    persistence_minimum_verticality: float = 0.0,
    persistence_method: str = "longest_run",
    persistence_summary: str = "mean",
    periodicity_minimum_lag: int = 2,
    periodicity_maximum_lag: int | None = None,
    periodicity_use_absolute_correlation: bool = False,
    amplitude_statistic: str = "mean",
    amplitude_percentile: float = 90.0,
    heterogeneity_method: str = "column_std",
    standardization_epsilon: float = 1e-8,
    standardization_clip_limits: (
        tuple[float, float]
        | None
    ) = (-5.0, 5.0),
) -> FeatureSignals:
    """
    Extract all mathematical feature signals from a preprocessed scan.

    Processing order
    ----------------
    1. Validate the preprocessed image.
    2. Compute a full-image structure-tensor verticality map.
    3. Compute one depth-persistence value per image column.
    4. Construct overlapping horizontal windows.
    5. Calculate one feature vector per window center.
    6. Robustly standardize each feature across the scan.
    """
    image = validate_feature_image(
        image
    )

    image_depth, image_width = (
        image.shape
    )

    validate_window_configuration(
        image_width=image_width,
        window_width=window_width,
        stride=stride,
    )

    verticality_map, verticality_components = (
        compute_structure_tensor_verticality(
            image=image,
            gradient_sigma=(
                verticality_gradient_sigma
            ),
            tensor_sigma=(
                verticality_tensor_sigma
            ),
        )
    )

    (
        persistence_signal,
        persistence_metadata,
    ) = compute_column_persistence(
        image=image,
        bright_percentile=(
            persistence_bright_percentile
        ),
        verticality_map=verticality_map,
        minimum_verticality=(
            persistence_minimum_verticality
        ),
        method=persistence_method,
    )

    padded_image, padding_radius = (
        pad_image_horizontally(
            image=image,
            window_width=window_width,
            padding_mode=padding_mode,
            constant_value=(
                padding_constant_value
            ),
        )
    )

    padded_verticality, _ = (
        pad_image_horizontally(
            image=verticality_map,
            window_width=window_width,
            padding_mode=padding_mode,
            constant_value=0.0,
        )
    )

    padded_persistence = np.pad(
        persistence_signal,
        pad_width=(
            padding_radius,
            padding_radius,
        ),
        mode=padding_mode,
    ).astype(np.float32)

    x_positions = get_window_centers(
        image_width=image_width,
        stride=stride,
    )

    feature_storage: dict[
        str,
        list[float],
    ] = {
        feature_name: []
        for feature_name in FEATURE_NAMES
    }

    periodicity_best_lags: list[int] = []

    for center_x in x_positions:
        image_window = extract_window(
            padded_image=padded_image,
            center_x=int(
                center_x
            ),
            window_width=window_width,
        )

        verticality_window = extract_window(
            padded_image=(
                padded_verticality
            ),
            center_x=int(
                center_x
            ),
            window_width=window_width,
        )

        persistence_window = (
            padded_persistence[
                int(center_x):
                int(center_x)
                + window_width
            ]
        )

        verticality_value = (
            compute_verticality_feature(
                verticality_window,
                summary=verticality_summary,
                upper_percentile=(
                    verticality_percentile
                ),
            )
        )

        persistence_value = (
            compute_persistence_feature(
                persistence_window,
                summary=persistence_summary,
            )
        )

        (
            periodicity_value,
            best_lag,
            _,
        ) = compute_periodicity_feature(
            image_window,
            minimum_lag=(
                periodicity_minimum_lag
            ),
            maximum_lag=(
                periodicity_maximum_lag
            ),
            use_absolute_correlation=(
                periodicity_use_absolute_correlation
            ),
        )

        amplitude_value = (
            compute_amplitude_feature(
                image_window,
                statistic=(
                    amplitude_statistic
                ),
                percentile=(
                    amplitude_percentile
                ),
            )
        )

        heterogeneity_value = (
            compute_heterogeneity_feature(
                image_window,
                method=(
                    heterogeneity_method
                ),
            )
        )

        feature_storage[
            "verticality"
        ].append(
            verticality_value
        )

        feature_storage[
            "persistence"
        ].append(
            persistence_value
        )

        feature_storage[
            "periodicity"
        ].append(
            periodicity_value
        )

        feature_storage[
            "amplitude"
        ].append(
            amplitude_value
        )

        feature_storage[
            "heterogeneity"
        ].append(
            heterogeneity_value
        )

        periodicity_best_lags.append(
            best_lag
        )

    raw_features = {
        feature_name: np.asarray(
            feature_storage[
                feature_name
            ],
            dtype=np.float32,
        )
        for feature_name in FEATURE_NAMES
    }

    (
        standardized_features,
        standardization_metadata,
    ) = standardize_feature_dictionary(
        raw_features=raw_features,
        epsilon=standardization_epsilon,
        clip_limits=(
            standardization_clip_limits
        ),
    )

    metadata = {
        "image_shape": tuple(
            image.shape
        ),
        "image_depth": int(
            image_depth
        ),
        "image_width": int(
            image_width
        ),
        "number_of_windows": int(
            x_positions.size
        ),
        "window": {
            "width": int(
                window_width
            ),
            "radius": int(
                padding_radius
            ),
            "stride": int(
                stride
            ),
            "padding_mode": padding_mode,
            "padding_constant_value": float(
                padding_constant_value
            ),
        },
        "verticality": {
            "gradient_sigma": float(
                verticality_gradient_sigma
            ),
            "tensor_sigma": float(
                verticality_tensor_sigma
            ),
            "summary": (
                verticality_summary.lower()
            ),
            "summary_percentile": float(
                verticality_percentile
            ),
            "map_minimum": float(
                verticality_map.min()
            ),
            "map_maximum": float(
                verticality_map.max()
            ),
            "map_mean": float(
                verticality_map.mean()
            ),
        },
        "persistence": {
            **persistence_metadata,
            "summary": (
                persistence_summary.lower()
            ),
            "signal_minimum": float(
                persistence_signal.min()
            ),
            "signal_maximum": float(
                persistence_signal.max()
            ),
            "signal_mean": float(
                persistence_signal.mean()
            ),
        },
        "periodicity": {
            "minimum_lag": int(
                periodicity_minimum_lag
            ),
            "maximum_lag": (
                int(
                    periodicity_maximum_lag
                )
                if periodicity_maximum_lag
                is not None
                else None
            ),
            "use_absolute_correlation": bool(
                periodicity_use_absolute_correlation
            ),
            "best_lags": [
                int(value)
                for value in periodicity_best_lags
            ],
        },
        "amplitude": {
            "statistic": (
                amplitude_statistic.lower()
            ),
            "percentile": float(
                amplitude_percentile
            ),
        },
        "heterogeneity": {
            "method": (
                heterogeneity_method.lower()
            ),
        },
        "standardization": {
            "method": "median_iqr",
            "epsilon": float(
                standardization_epsilon
            ),
            "clip_limits": (
                [
                    float(
                        standardization_clip_limits[0]
                    ),
                    float(
                        standardization_clip_limits[1]
                    ),
                ]
                if standardization_clip_limits
                is not None
                else None
            ),
            "feature_statistics": (
                standardization_metadata
            ),
        },
        "structure_tensor_diagnostics": {
            "anisotropy_minimum": float(
                verticality_components[
                    "anisotropy"
                ].min()
            ),
            "anisotropy_maximum": float(
                verticality_components[
                    "anisotropy"
                ].max()
            ),
            "anisotropy_mean": float(
                verticality_components[
                    "anisotropy"
                ].mean()
            ),
        },
    }

    return FeatureSignals(
        x_positions=x_positions.astype(
            np.float32
        ),
        raw_features=raw_features,
        standardized_features=(
            standardized_features
        ),
        verticality_map=verticality_map,
        persistence_signal=(
            persistence_signal
        ),
        metadata=metadata,
    )