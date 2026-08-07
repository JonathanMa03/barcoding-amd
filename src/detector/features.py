# features.py
#
# Responsibilities:
# - Accept a preprocessed two-dimensional OCT image.
# - Calculate pixel-level verticality and gradient information.
# - Construct structural-pixel evidence maps.
# - Calculate structurally cleaned column-intensity summaries.
# - Calculate local depth-continuity signals.
# - Calculate column-level vertical-organization summaries.
# - Support configurable signal smoothing.
# - Retain the original sliding-window feature implementation for
#   comparison with the structural-hypertransmission pipeline.
#
# Inputs:
# - Preprocessed two-dimensional image
# - Structural feature configuration
# - Column-statistic configuration
# - Continuity configuration
#
# Outputs:
# - Pixel-level feature maps
# - One-dimensional column-level feature signals
# - Feature metadata and diagnostics
#
# This module describes the image mathematically.
# It must not assign barcode labels, apply detector thresholds,
# clean binary detector masks, or extract final barcode intervals.


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.ndimage import gaussian_filter1d


FEATURE_NAMES = (
    "verticality",
    "persistence",
    "continuity",
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
        Dictionary containing the original feature signals:
        verticality, persistence, continuity, amplitude, and
        heterogeneity.
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

def compute_simple_verticality_map(
    image: np.ndarray,
    *,
    smoothing_sigma: float = 1.0,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, dict[str, np.ndarray | float]]:
    """
    Compute a simple verticality map using gradient-direction dominance.

    A vertically oriented image structure typically produces a strong
    horizontal intensity gradient because intensity changes while moving
    across the structure but changes less while moving along its depth.

    The verticality score is

        verticality =
            |I_x|
            / (|I_x| + |I_z| + epsilon)

    where

        I_x
            Horizontal image gradient.

        I_z
            Vertical or depth-wise image gradient.

    Values approach one when the horizontal gradient dominates and
    approach zero when the depth-wise gradient dominates.

    Parameters
    ----------
    image:
        Two-dimensional preprocessed image.
    smoothing_sigma:
        Gaussian smoothing applied before gradient calculation. A value
        of zero disables smoothing.
    epsilon:
        Small positive constant used for numerical stability.

    Returns
    -------
    verticality_map:
        Two-dimensional float32 map with values between zero and one.
    diagnostics:
        Intermediate smoothed image and gradient arrays.
    """
    image = validate_feature_image(
        image
    )

    if smoothing_sigma < 0:
        raise ValueError(
            "smoothing_sigma must be nonnegative."
        )

    if epsilon <= 0:
        raise ValueError(
            "epsilon must be positive."
        )

    if smoothing_sigma > 0:
        smoothed_image = gaussian_filter(
            image,
            sigma=float(
                smoothing_sigma
            ),
            mode="reflect",
        ).astype(np.float32)

    else:
        smoothed_image = image.copy()

    gradient_z, gradient_x = np.gradient(
        smoothed_image
    )

    absolute_gradient_x = np.abs(
        gradient_x
    )

    absolute_gradient_z = np.abs(
        gradient_z
    )

    verticality_map = (
        absolute_gradient_x
        / (
            absolute_gradient_x
            + absolute_gradient_z
            + epsilon
        )
    )

    verticality_map = np.clip(
        verticality_map,
        0.0,
        1.0,
    ).astype(np.float32)

    diagnostics: dict[
        str,
        np.ndarray | float,
    ] = {
        "smoothed_image": (
            smoothed_image.astype(
                np.float32
            )
        ),
        "gradient_x": np.asarray(
            gradient_x,
            dtype=np.float32,
        ),
        "gradient_z": np.asarray(
            gradient_z,
            dtype=np.float32,
        ),
        "absolute_gradient_x": np.asarray(
            absolute_gradient_x,
            dtype=np.float32,
        ),
        "absolute_gradient_z": np.asarray(
            absolute_gradient_z,
            dtype=np.float32,
        ),
        "smoothing_sigma": float(
            smoothing_sigma
        ),
        "epsilon": float(
            epsilon
        ),
    }

    return (
        verticality_map,
        diagnostics,
    )

def create_structural_mask(
    verticality_map: np.ndarray,
    *,
    verticality_threshold: float = 0.60,
    gradient_magnitude: np.ndarray | None = None,
    minimum_gradient_magnitude: float | None = None,
    minimum_component_size: int = 0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Create a binary mask of strongly vertically organized pixels.

    A pixel is initially classified as structural when

        verticality >= verticality_threshold.

    An optional gradient-magnitude condition can also be imposed:

        gradient_magnitude >= minimum_gradient_magnitude.

    This prevents nearly uniform pixels from being classified as
    structurally meaningful based only on gradient direction.

    Parameters
    ----------
    verticality_map:
        Two-dimensional verticality map with values between zero and one.
    verticality_threshold:
        Minimum verticality required for a structural pixel.
    gradient_magnitude:
        Optional two-dimensional gradient-magnitude map.
    minimum_gradient_magnitude:
        Optional minimum gradient magnitude. This requires
        ``gradient_magnitude``.
    minimum_component_size:
        Optional minimum connected-component size in pixels. Components
        smaller than this value are removed. A value of zero disables
        connected-component cleanup.

    Returns
    -------
    structural_mask:
        Two-dimensional boolean structural-pixel mask.
    metadata:
        Threshold settings and mask summaries.
    """
    verticality = np.asarray(
        verticality_map,
        dtype=np.float32,
    )

    if verticality.ndim != 2:
        raise ValueError(
            "verticality_map must be two-dimensional."
        )

    if verticality.size == 0:
        raise ValueError(
            "verticality_map must not be empty."
        )

    if not np.isfinite(
        verticality
    ).all():
        raise ValueError(
            "verticality_map contains non-finite values."
        )

    if not (
        0.0
        <= verticality_threshold
        <= 1.0
    ):
        raise ValueError(
            "verticality_threshold must lie between 0 and 1."
        )

    if minimum_component_size < 0:
        raise ValueError(
            "minimum_component_size must be nonnegative."
        )

    structural_mask = (
        verticality
        >= float(
            verticality_threshold
        )
    )

    resolved_gradient_threshold = None

    if minimum_gradient_magnitude is not None:
        if gradient_magnitude is None:
            raise ValueError(
                "gradient_magnitude must be supplied when "
                "minimum_gradient_magnitude is used."
            )

        gradient_array = np.asarray(
            gradient_magnitude,
            dtype=np.float32,
        )

        if gradient_array.shape != verticality.shape:
            raise ValueError(
                "gradient_magnitude must have the same shape as "
                "verticality_map."
            )

        if not np.isfinite(
            gradient_array
        ).all():
            raise ValueError(
                "gradient_magnitude contains non-finite values."
            )

        if minimum_gradient_magnitude < 0:
            raise ValueError(
                "minimum_gradient_magnitude must be nonnegative."
            )

        resolved_gradient_threshold = float(
            minimum_gradient_magnitude
        )

        structural_mask &= (
            gradient_array
            >= resolved_gradient_threshold
        )

    if minimum_component_size > 1:
        try:
            from scipy.ndimage import label
        except ImportError as exc:
            raise ImportError(
                "scipy is required for connected-component cleanup."
            ) from exc

        labeled_mask, number_of_components = label(
            structural_mask
        )

        cleaned_mask = np.zeros_like(
            structural_mask,
            dtype=bool,
        )

        retained_components = 0

        for component_id in range(
            1,
            number_of_components + 1,
        ):
            component = (
                labeled_mask
                == component_id
            )

            component_size = int(
                component.sum()
            )

            if (
                component_size
                >= minimum_component_size
            ):
                cleaned_mask |= component
                retained_components += 1

        structural_mask = cleaned_mask

    else:
        number_of_components = None
        retained_components = None

    structural_mask = structural_mask.astype(
        bool
    )

    metadata = {
        "verticality_threshold": float(
            verticality_threshold
        ),
        "minimum_gradient_magnitude": (
            resolved_gradient_threshold
        ),
        "minimum_component_size": int(
            minimum_component_size
        ),
        "structural_pixel_count": int(
            structural_mask.sum()
        ),
        "total_pixel_count": int(
            structural_mask.size
        ),
        "structural_pixel_fraction": float(
            structural_mask.mean()
        ),
        "number_of_components_before_cleanup": (
            int(number_of_components)
            if number_of_components is not None
            else None
        ),
        "number_of_components_retained": (
            int(retained_components)
            if retained_components is not None
            else None
        ),
    }

    return (
        structural_mask,
        metadata,
    )

def compute_column_intensity_statistics(
    image: np.ndarray,
    *,
    exclusion_mask: np.ndarray | None = None,
    upper_quantile: float = 0.90,
    minimum_valid_pixels: int = 5,
    empty_column_value: float = np.nan,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, Any],
]:
    """
    Calculate column-level intensity statistics after optional exclusion.

    When an exclusion mask is supplied, pixels marked ``True`` are
    removed before calculating each column's intensity summaries.

    For each horizontal column, this function calculates:

    - mean intensity;
    - median intensity;
    - selected upper quantile;
    - intensity standard deviation;
    - number and fraction of valid pixels.

    Parameters
    ----------
    image:
        Two-dimensional normalized or denoised image.
    exclusion_mask:
        Optional boolean mask with the same shape as ``image``. Pixels
        marked ``True`` are excluded. This will typically be the
        structural-pixel mask.
    upper_quantile:
        Quantile calculated for each column. The default of 0.90 returns
        the 90th percentile.
    minimum_valid_pixels:
        Minimum number of retained pixels required to summarize a
        column.
    empty_column_value:
        Value assigned to statistics for columns containing fewer than
        ``minimum_valid_pixels`` retained pixels.

    Returns
    -------
    statistics:
        Dictionary containing one-dimensional arrays:

        ``mean``
            Mean retained intensity for each column.

        ``median``
            Median retained intensity for each column.

        ``upper_quantile``
            Requested retained-intensity quantile for each column.

        ``standard_deviation``
            Standard deviation of retained intensities.

        ``valid_pixel_count``
            Number of retained pixels per column.

        ``valid_pixel_fraction``
            Fraction of depth pixels retained per column.

    metadata:
        Image dimensions, exclusion summaries, and quantile settings.
    """
    image_array = validate_feature_image(
        image
    )

    image_depth, image_width = (
        image_array.shape
    )

    if not (
        0.0
        <= upper_quantile
        <= 1.0
    ):
        raise ValueError(
            "upper_quantile must lie between 0 and 1."
        )

    if minimum_valid_pixels <= 0:
        raise ValueError(
            "minimum_valid_pixels must be positive."
        )

    if exclusion_mask is None:
        excluded = np.zeros_like(
            image_array,
            dtype=bool,
        )

    else:
        excluded = np.asarray(
            exclusion_mask,
            dtype=bool,
        )

        if excluded.shape != image_array.shape:
            raise ValueError(
                "exclusion_mask must have the same shape as image."
            )

    column_mean = np.full(
        image_width,
        empty_column_value,
        dtype=np.float32,
    )

    column_median = np.full(
        image_width,
        empty_column_value,
        dtype=np.float32,
    )

    column_upper_quantile = np.full(
        image_width,
        empty_column_value,
        dtype=np.float32,
    )

    column_standard_deviation = np.full(
        image_width,
        empty_column_value,
        dtype=np.float32,
    )

    valid_pixel_count = np.zeros(
        image_width,
        dtype=np.int32,
    )

    for column_index in range(
        image_width
    ):
        valid_mask = ~excluded[
            :,
            column_index,
        ]

        valid_values = image_array[
            valid_mask,
            column_index,
        ]

        valid_pixel_count[
            column_index
        ] = int(
            valid_values.size
        )

        if (
            valid_values.size
            < minimum_valid_pixels
        ):
            continue

        column_mean[
            column_index
        ] = float(
            np.mean(
                valid_values
            )
        )

        column_median[
            column_index
        ] = float(
            np.median(
                valid_values
            )
        )

        column_upper_quantile[
            column_index
        ] = float(
            np.quantile(
                valid_values,
                upper_quantile,
            )
        )

        column_standard_deviation[
            column_index
        ] = float(
            np.std(
                valid_values
            )
        )

    valid_pixel_fraction = (
        valid_pixel_count.astype(
            np.float32
        )
        / float(
            image_depth
        )
    )

    statistics = {
        "mean": column_mean,
        "median": column_median,
        "upper_quantile": (
            column_upper_quantile
        ),
        "standard_deviation": (
            column_standard_deviation
        ),
        "valid_pixel_count": (
            valid_pixel_count
        ),
        "valid_pixel_fraction": (
            valid_pixel_fraction
        ),
    }

    columns_with_sufficient_data = (
        valid_pixel_count
        >= minimum_valid_pixels
    )

    metadata = {
        "image_shape": tuple(
            image_array.shape
        ),
        "image_depth": int(
            image_depth
        ),
        "image_width": int(
            image_width
        ),
        "exclusion_applied": bool(
            exclusion_mask is not None
        ),
        "excluded_pixel_count": int(
            excluded.sum()
        ),
        "excluded_pixel_fraction": float(
            excluded.mean()
        ),
        "upper_quantile": float(
            upper_quantile
        ),
        "upper_percentile": float(
            100.0
            * upper_quantile
        ),
        "minimum_valid_pixels": int(
            minimum_valid_pixels
        ),
        "columns_with_sufficient_data": int(
            columns_with_sufficient_data.sum()
        ),
        "columns_with_insufficient_data": int(
            (
                ~columns_with_sufficient_data
            ).sum()
        ),
        "empty_column_value": float(
            empty_column_value
        ),
    }

    return (
        statistics,
        metadata,
    )



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


def compute_continuity_feature(
    image_window: np.ndarray,
    *,
    depth_lag: int = 1,
    summary: str = "mean",
    minimum_row_standard_deviation: float = 1e-6,
    epsilon: float = 1e-8,
) -> tuple[float, np.ndarray]:
    """
    Measure whether the horizontal intensity pattern persists with depth.

    A barcode-like region contains vertical structures. Therefore, the
    bright-dark pattern observed across the horizontal window should
    remain similar as the detector moves downward through the image.

    For each pair of rows separated by ``depth_lag``, this function
    calculates the correlation between their horizontal intensity
    patterns.

    For example, with ``depth_lag=1``:

        row 0 is compared with row 1
        row 1 is compared with row 2
        row 2 is compared with row 3
        ...

    A high positive correlation indicates that bright and dark
    horizontal positions remain aligned across depth, which is
    consistent with vertically continuous structures.

    Parameters
    ----------
    image_window:
        Full-depth local image window with shape
        ``(image_depth, window_width)``.
    depth_lag:
        Number of depth rows separating the compared horizontal
        intensity patterns.
    summary:
        Method used to summarize the row-pair correlations. Supported
        values are ``"mean"``, ``"median"``, and ``"percentile"``.
    minimum_row_standard_deviation:
        Row pairs with less horizontal variation than this value are
        excluded because correlation is not meaningful for nearly
        constant rows.
    epsilon:
        Small positive value used for numerical stability.

    Returns
    -------
    continuity_score:
        Summary of valid row-to-row correlations. Larger positive values
        indicate stronger persistence of the horizontal pattern through
        depth.
    row_correlations:
        Correlation calculated for each valid pair of depth rows.
    """
    image_window = validate_feature_image(
        image_window
    )

    image_depth = image_window.shape[0]

    if depth_lag <= 0:
        raise ValueError(
            "depth_lag must be positive."
        )

    if depth_lag >= image_depth:
        raise ValueError(
            "depth_lag must be smaller than the image-window depth."
        )

    if minimum_row_standard_deviation < 0:
        raise ValueError(
            "minimum_row_standard_deviation must be nonnegative."
        )

    if epsilon <= 0:
        raise ValueError(
            "epsilon must be positive."
        )

    summary = summary.lower()

    valid_summaries = {
        "mean",
        "median",
        "percentile",
    }

    if summary not in valid_summaries:
        raise ValueError(
            "continuity summary must be one of "
            f"{sorted(valid_summaries)}."
        )

    row_correlations: list[float] = []

    for first_row_index in range(
        image_depth - depth_lag
    ):
        second_row_index = (
            first_row_index
            + depth_lag
        )

        first_row = np.asarray(
            image_window[
                first_row_index,
                :,
            ],
            dtype=np.float64,
        )

        second_row = np.asarray(
            image_window[
                second_row_index,
                :,
            ],
            dtype=np.float64,
        )

        first_centered = (
            first_row
            - first_row.mean()
        )

        second_centered = (
            second_row
            - second_row.mean()
        )

        first_standard_deviation = float(
            first_centered.std()
        )

        second_standard_deviation = float(
            second_centered.std()
        )

        if (
            first_standard_deviation
            < minimum_row_standard_deviation
            or second_standard_deviation
            < minimum_row_standard_deviation
        ):
            continue

        denominator = (
            np.sqrt(
                np.sum(
                    first_centered
                    * first_centered
                )
                * np.sum(
                    second_centered
                    * second_centered
                )
            )
            + epsilon
        )

        correlation = float(
            np.sum(
                first_centered
                * second_centered
            )
            / denominator
        )

        row_correlations.append(
            correlation
        )

    if not row_correlations:
        return (
            0.0,
            np.empty(
                0,
                dtype=np.float32,
            ),
        )

    correlations_array = np.asarray(
        row_correlations,
        dtype=np.float32,
    )

    if summary == "mean":
        continuity_score = float(
            np.mean(
                correlations_array
            )
        )

    elif summary == "median":
        continuity_score = float(
            np.median(
                correlations_array
            )
        )

    else:
        continuity_score = float(
            np.percentile(
                correlations_array,
                75.0,
            )
        )

    return (
        continuity_score,
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
    continuity_depth_lag: int = 1,
    continuity_summary: str = "mean",
    continuity_minimum_row_standard_deviation: float = 1e-6,
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
            continuity_value,
            _,
        ) = compute_continuity_feature(
            image_window,
            depth_lag=(
                continuity_depth_lag
            ),
            summary=(
                continuity_summary
            ),
            minimum_row_standard_deviation=(
                continuity_minimum_row_standard_deviation
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
            "continuity"
        ].append(
            continuity_value
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
        "continuity": {
            "definition": (
                "correlation_between_horizontal_patterns_"
                "at_separated_depth_rows"
            ),
            "depth_lag": int(
                continuity_depth_lag
            ),
            "summary": (
                continuity_summary.lower()
            ),
            "minimum_row_standard_deviation": float(
                continuity_minimum_row_standard_deviation
            ),
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

def smooth_finite_signal(
    values: np.ndarray,
    *,
    sigma: float = 2.0,
) -> np.ndarray:
    """
    Smooth a one-dimensional feature signal while preserving length.

    Non-finite values are first replaced by linear interpolation before
    Gaussian smoothing is applied.

    Parameters
    ----------
    values:
        One-dimensional numerical feature signal.
    sigma:
        Standard deviation of the one-dimensional Gaussian kernel,
        measured in signal samples.

    Returns
    -------
    np.ndarray
        Smoothed float32 signal with the same length as the input.
    """
    values = np.asarray(
        values,
        dtype=np.float32,
    )

    if values.ndim != 1:
        raise ValueError(
            "values must be one-dimensional."
        )

    if values.size == 0:
        raise ValueError(
            "values must not be empty."
        )

    if sigma < 0:
        raise ValueError(
            "sigma must be nonnegative."
        )

    finite = np.isfinite(
        values
    )

    if not finite.any():
        raise ValueError(
            "values contains no finite observations."
        )

    filled = values.copy()

    if not finite.all():
        positions = np.arange(
            values.size
        )

        filled[
            ~finite
        ] = np.interp(
            positions[~finite],
            positions[finite],
            values[finite],
        )

    if sigma == 0:
        return filled.astype(
            np.float32
        )

    return gaussian_filter1d(
        filled,
        sigma=float(sigma),
        mode="nearest",
    ).astype(np.float32)

def compute_local_depth_continuity(
    image: np.ndarray,
    *,
    window_width: int = 15,
    depth_lag: int = 4,
    minimum_row_standard_deviation: float = 1e-6,
) -> np.ndarray:
    """
    Calculate local persistence of horizontal intensity structure through depth.

    For each horizontal position, a local window is extracted. Intensity
    patterns separated by ``depth_lag`` rows are correlated, and the
    median valid correlation across depth is returned.

    Conceptually,

        C(x) = median_z corr(
            I(z, W_x),
            I(z + lag, W_x)
        )

    where

        x:
            Horizontal image position.

        W_x:
            Local horizontal neighborhood centered at x.

        z:
            Depth coordinate.

        lag:
            Number of rows separating the two compared depth profiles.

    Higher values indicate that the same horizontal intensity pattern
    persists through depth.

    Parameters
    ----------
    image:
        Two-dimensional preprocessed OCT image.
    window_width:
        Odd horizontal width of the local comparison window.
    depth_lag:
        Number of rows separating compared depth profiles.
    minimum_row_standard_deviation:
        Rows with less variation than this value are skipped because
        correlation is unstable for nearly constant signals.

    Returns
    -------
    np.ndarray
        One continuity value per horizontal image column.
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

    if (
        window_width <= 1
        or window_width % 2 == 0
    ):
        raise ValueError(
            "window_width must be an odd integer greater than one."
        )

    if not (
        1
        <= depth_lag
        < image.shape[0]
    ):
        raise ValueError(
            "depth_lag must be between 1 and image depth - 1."
        )

    if minimum_row_standard_deviation < 0:
        raise ValueError(
            "minimum_row_standard_deviation must be nonnegative."
        )

    radius = (
        window_width // 2
    )

    padded_image = np.pad(
        image,
        (
            (0, 0),
            (radius, radius),
        ),
        mode="reflect",
    )

    continuity = np.full(
        image.shape[1],
        np.nan,
        dtype=np.float32,
    )

    for horizontal_position in range(
        image.shape[1]
    ):
        local_window = padded_image[
            :,
            horizontal_position:
            horizontal_position
            + window_width,
        ]

        correlations: list[float] = []

        for depth_position in range(
            image.shape[0]
            - depth_lag
        ):
            first_row = local_window[
                depth_position
            ]

            second_row = local_window[
                depth_position
                + depth_lag
            ]

            if (
                np.std(first_row)
                < minimum_row_standard_deviation
                or
                np.std(second_row)
                < minimum_row_standard_deviation
            ):
                continue

            correlation = np.corrcoef(
                first_row,
                second_row,
            )[0, 1]

            if np.isfinite(
                correlation
            ):
                correlations.append(
                    float(correlation)
                )

        if correlations:
            continuity[
                horizontal_position
            ] = float(
                np.median(
                    correlations
                )
            )

    return continuity

def compute_column_verticality_statistics(
    verticality_map: np.ndarray,
    *,
    structural_mask: np.ndarray | None = None,
    upper_quantile: float = 0.90,
) -> dict[str, np.ndarray]:
    """
    Summarize vertical organization through depth for every image column.

    Parameters
    ----------
    verticality_map:
        Two-dimensional verticality map with the same shape as the
        processed image.
    structural_mask:
        Optional Boolean map identifying pixels considered strongly
        vertical and sufficiently high-gradient.
    upper_quantile:
        Verticality quantile calculated through depth for each column.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary containing:

        ``mean``
            Mean verticality through depth.

        ``upper_quantile``
            Requested upper verticality quantile through depth.

        ``strong_vertical_fraction``
            Fraction of depth pixels marked in ``structural_mask``.
            Returned as NaN when no structural mask is supplied.
    """
    verticality = np.asarray(
        verticality_map,
        dtype=np.float32,
    )

    if verticality.ndim != 2:
        raise ValueError(
            "verticality_map must be two-dimensional."
        )

    if not (
        0.0
        <= upper_quantile
        <= 1.0
    ):
        raise ValueError(
            "upper_quantile must lie between 0 and 1."
        )

    mean_verticality = np.mean(
        verticality,
        axis=0,
    ).astype(np.float32)

    upper_verticality = np.quantile(
        verticality,
        upper_quantile,
        axis=0,
    ).astype(np.float32)

    if structural_mask is None:
        strong_vertical_fraction = np.full(
            verticality.shape[1],
            np.nan,
            dtype=np.float32,
        )

    else:
        mask = np.asarray(
            structural_mask,
            dtype=bool,
        )

        if mask.shape != verticality.shape:
            raise ValueError(
                "structural_mask must have the same shape "
                "as verticality_map."
            )

        strong_vertical_fraction = np.mean(
            mask,
            axis=0,
        ).astype(np.float32)

    return {
        "mean": mean_verticality,
        "upper_quantile": upper_verticality,
        "strong_vertical_fraction": (
            strong_vertical_fraction
        ),
    }