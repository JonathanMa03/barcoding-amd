# preprocessing.py
#
# Responsibilities:
# - Retrieve a selected retinal-layer boundary, such as BM.
# - Interpolate missing or invalid boundary coordinates.
# - Flatten a B-scan to the selected boundary.
# - Crop a configurable fixed-depth region below the boundary.
# - Normalize image intensities using configurable methods.
# - Denoise the processed image using configurable methods.
# - Run the complete preprocessing pipeline.
#
# Inputs:
# - EyeVolume object
# - B-scan index
# - Boundary name
# - Crop, normalization, and denoising settings
#
# Outputs:
# - A structured preprocessing result containing the raw scan,
#   boundary, flattened scan, cropped ROI, normalized image,
#   denoised image, and metadata.
#
# This module should not compute barcode features or detections.
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

import numpy as np
from scipy.ndimage import (
    gaussian_filter,
    median_filter,
)

from src.loading.data_loading import (
    inspect_volume_layers,
    load_bscan,
)


@dataclass
class PreprocessedArtifact:
    """Source-independent output passed to the detector stage."""

    image: np.ndarray
    source_image: np.ndarray
    metadata: dict[str, Any]
    bscan_index: int | None = None
    layer_boundary: np.ndarray | None = None
    flattened_image: np.ndarray | None = None
    cropped_image: np.ndarray | None = None


def preprocess_loaded_scan(
    scan,
    *,
    layer_name: str = "BM",
    flatten: bool | None = None,
    reference_row: int | None = None,
    flatten_fill_value: float = 0.0,
    depth_below_layer: int = 150,
    include_boundary: bool = True,
    require_full_depth: bool = False,
    crop_fill_value: float = 0.0,
    normalization_method: str = "percentile",
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    zscore_epsilon: float = 1e-8,
    denoise_method: str = "gaussian",
    gaussian_sigma: float | tuple[float, float] = (1.0, 0.5),
    median_size: int | tuple[int, int] = 3,
    gaussian_mode: str = "reflect",
) -> PreprocessedArtifact:
    """Preprocess either an E2E-backed or PNG-backed :class:`LoadedScan`.

    Flattening defaults to enabled when a layer boundary is present. A PNG
    without a boundary is treated as an already selected image/ROI and still
    receives the same normalization and denoising steps.
    """
    source_image = np.asarray(scan.image, dtype=np.float32)
    if source_image.ndim != 2:
        raise ValueError("Loaded scan image must be two-dimensional.")

    should_flatten = scan.layer_boundary is not None if flatten is None else flatten
    flattened = None
    cropped = None
    resolved_reference_row = None
    working_image = source_image

    if should_flatten:
        if scan.layer_boundary is None:
            raise ValueError("Flattening requires a layer_boundary in the loaded scan.")
        flattened, resolved_reference_row = flatten_to_boundary(
            source_image,
            scan.layer_boundary,
            reference_row=reference_row,
            fill_value=flatten_fill_value,
        )
        cropped = crop_below_boundary(
            flattened,
            resolved_reference_row,
            depth_below_layer=depth_below_layer,
            include_boundary=include_boundary,
            require_full_depth=require_full_depth,
            fill_value=crop_fill_value,
        )
        working_image = cropped

    normalized, normalization_metadata = normalize_image(
        working_image,
        method=normalization_method,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
        zscore_epsilon=zscore_epsilon,
    )
    denoised, denoising_metadata = denoise_image(
        normalized,
        method=denoise_method,
        gaussian_sigma=gaussian_sigma,
        median_size=median_size,
        gaussian_mode=gaussian_mode,
    )
    metadata = dict(scan.metadata)
    metadata.setdefault("source_type", scan.source_type)
    metadata.setdefault("source_path", str(scan.source_path))
    metadata.setdefault("bscan_index", scan.bscan_index)
    metadata["preprocessing"] = {
        "layer_name": layer_name,
        "flattened": bool(should_flatten),
        "reference_row": resolved_reference_row,
        "depth_below_layer": depth_below_layer if should_flatten else None,
        "normalization": normalization_metadata,
        "denoising": denoising_metadata,
        "output_shape": list(denoised.shape),
    }
    return PreprocessedArtifact(
        image=denoised,
        source_image=source_image,
        metadata=metadata,
        bscan_index=scan.bscan_index,
        layer_boundary=scan.layer_boundary,
        flattened_image=flattened,
        cropped_image=cropped,
    )


def save_preprocessed_scan(scan: PreprocessedArtifact, output_path: str | Path) -> Path:
    """Save a preprocessed artifact for the detector script."""
    output_path = Path(output_path).with_suffix(".npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        image=scan.image,
        source_image=scan.source_image,
        metadata=np.asarray(json.dumps(scan.metadata, default=str)),
        bscan_index=np.asarray(-1 if scan.bscan_index is None else scan.bscan_index),
    )
    return output_path.resolve()


def load_preprocessed_scan(input_path: str | Path) -> PreprocessedArtifact:
    """Load an artifact written by :func:`save_preprocessed_scan`."""
    with np.load(Path(input_path), allow_pickle=False) as archive:
        index = int(archive["bscan_index"])
        return PreprocessedArtifact(
            image=archive["image"].astype(np.float32),
            source_image=archive["source_image"].astype(np.float32),
            metadata=json.loads(str(archive["metadata"])),
            bscan_index=None if index < 0 else index,
        )


@dataclass
class PreprocessedScan:
    """
    Complete preprocessing output for one OCT B-scan.

    Attributes
    ----------
    bscan_index:
        Zero-based B-scan index.
    layer_name:
        Retinal-layer annotation used for flattening.
    raw_bscan:
        Original two-dimensional B-scan.
    layer_boundary:
        Interpolated layer coordinate for every image column.
    flattened_bscan:
        B-scan after aligning the selected layer to a common row.
    sub_layer_crop:
        Fixed-depth crop beginning at or below the flattened layer.
    normalized_scan:
        Cropped image after the selected normalization method.
    denoised_scan:
        Normalized image after optional denoising.
    reference_row:
        Row occupied by the selected layer after flattening.
    depth_below_layer:
        Requested crop depth.
    normalization_method:
        Normalization method applied to the crop.
    denoise_method:
        Denoising method applied after normalization.
    metadata:
        Configuration and diagnostic information.
    """

    bscan_index: int
    layer_name: str

    raw_bscan: np.ndarray
    layer_boundary: np.ndarray
    flattened_bscan: np.ndarray
    sub_layer_crop: np.ndarray
    normalized_scan: np.ndarray
    denoised_scan: np.ndarray

    reference_row: int
    depth_below_layer: int

    normalization_method: str
    denoise_method: str

    normalization_metadata: dict[str, Any]
    denoising_metadata: dict[str, Any]

    metadata: dict[str, Any]

def get_layer_boundary(
    volume,
    bscan_index: int,
    layer_name: str = "BM",
) -> np.ndarray:
    """
    Extract one retinal-layer boundary from an eyepy EyeVolume.

    Parameters
    ----------
    volume:
        Loaded eyepy EyeVolume.
    bscan_index:
        Zero-based index of the requested B-scan.
    layer_name:
        Retinal layer name, such as ``"BM"``.

    Returns
    -------
    np.ndarray
        One vertical boundary coordinate per image column, with shape
        ``(bscan_width,)``.
    """
    available_layers = inspect_volume_layers(volume)

    if layer_name not in available_layers:
        raise KeyError(
            f"Layer '{layer_name}' was not found. "
            f"Available layers: {available_layers}"
        )

    layer_annotation = volume.layers[layer_name]

    # eyepy stores boundary coordinates in the annotation's data property.
    if hasattr(layer_annotation, "data"):
        layer_array = np.asarray(
            layer_annotation.data,
            dtype=np.float32,
        )
    else:
        layer_array = np.asarray(
            layer_annotation,
            dtype=np.float32,
        )

    if layer_array.ndim == 1:
        # Only valid when the object already represents one B-scan.
        boundary = layer_array

    elif layer_array.ndim == 2:
        # Expected eyepy format: (number of B-scans, image width).
        if bscan_index < 0 or bscan_index >= layer_array.shape[0]:
            raise IndexError(
                f"B-scan index {bscan_index} is outside the layer-data "
                f"range 0 to {layer_array.shape[0] - 1}."
            )

        boundary = layer_array[bscan_index]

    elif layer_array.ndim == 3:
        # Handle a possible singleton channel dimension.
        squeezed = np.squeeze(layer_array)

        if squeezed.ndim != 2:
            raise ValueError(
                f"Could not interpret layer '{layer_name}' with shape "
                f"{layer_array.shape} after squeezing."
            )

        boundary = squeezed[bscan_index]

    else:
        raise ValueError(
            f"Unsupported data shape for layer '{layer_name}': "
            f"{layer_array.shape}"
        )

    boundary = np.asarray(boundary, dtype=np.float32)

    expected_width = int(volume[bscan_index].shape[1])

    if boundary.shape != (expected_width,):
        raise ValueError(
            f"Expected the '{layer_name}' boundary to have shape "
            f"({expected_width},), but received {boundary.shape}."
        )

    return interpolate_missing_boundary(boundary)

def interpolate_missing_boundary(
    boundary: np.ndarray,
) -> np.ndarray:
    """
    Fill missing retinal-boundary coordinates by linear interpolation.

    Non-finite and negative coordinates are treated as missing.
    """
    boundary = np.asarray(
        boundary,
        dtype=np.float32,
    ).copy()

    x = np.arange(boundary.size)

    valid = (
        np.isfinite(boundary)
        & (boundary >= 0)
    )

    if valid.sum() < 2:
        raise ValueError(
            "Boundary contains fewer than two valid coordinates and "
            "cannot be interpolated."
        )

    boundary[~valid] = np.interp(
        x[~valid],
        x[valid],
        boundary[valid],
    )

    return boundary

def flatten_to_boundary(
    image: np.ndarray,
    boundary: np.ndarray,
    reference_row: int | None = None,
    fill_value: float = 0.0,
) -> tuple[np.ndarray, int]:
    """
    Vertically shift each image column so the supplied boundary becomes flat.

    Parameters
    ----------
    image:
        Two-dimensional B-scan.
    boundary:
        Vertical boundary coordinate for each image column.
    reference_row:
        Row onto which the boundary is aligned. If omitted, the median
        boundary location is used.
    fill_value:
        Value used where shifting creates empty pixels.

    Returns
    -------
    flattened:
        Flattened B-scan.
    reference_row:
        Row corresponding to the flattened boundary.
    """
    image = np.asarray(image, dtype=np.float32)
    boundary = np.asarray(boundary, dtype=np.float32)

    height, width = image.shape

    if boundary.shape != (width,):
        raise ValueError(
            f"Boundary shape must be ({width},), received {boundary.shape}."
        )

    boundary_int = np.rint(boundary).astype(int)

    if reference_row is None:
        reference_row = int(np.median(boundary_int))

    if reference_row < 0 or reference_row >= height:
        raise ValueError(
            f"reference_row must be between 0 and {height - 1}."
        )

    flattened = np.full_like(image, fill_value)

    for x in range(width):
        shift = reference_row - boundary_int[x]

        if shift >= 0:
            source_start = 0
            source_end = height - shift
            target_start = shift
            target_end = height
        else:
            source_start = -shift
            source_end = height
            target_start = 0
            target_end = height + shift

        if source_end > source_start and target_end > target_start:
            flattened[target_start:target_end, x] = image[
                source_start:source_end,
                x,
            ]

    return flattened, reference_row

def crop_below_boundary(
    flattened_image: np.ndarray,
    reference_row: int,
    depth_below_layer: int = 150,
    include_boundary: bool = True,
    require_full_depth: bool = False,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Crop a fixed-depth region at and below a flattened retinal boundary.

    Parameters
    ----------
    flattened_image:
        B-scan already flattened to the selected layer.
    reference_row:
        Row occupied by the selected boundary after flattening.
    depth_below_layer:
        Number of rows retained at and below the selected layer.
    include_boundary:
        Whether the boundary row itself is included.
    require_full_depth:
        If ``True``, raise an error when fewer than the requested number
        of rows are available. If ``False``, pad the crop to the requested
        depth using ``fill_value``.
    fill_value:
        Value used when padding is needed.

    Returns
    -------
    np.ndarray
        Crop with shape ``(depth_below_layer, image_width)`` unless
        ``require_full_depth=True`` raises an exception.
    """
    image = np.asarray(
        flattened_image,
        dtype=np.float32,
    )

    if image.ndim != 2:
        raise ValueError(
            "flattened_image must be two-dimensional; "
            f"received shape {image.shape}."
        )

    if depth_below_layer <= 0:
        raise ValueError(
            "depth_below_layer must be positive."
        )

    height, width = image.shape

    start = (
        reference_row
        if include_boundary
        else reference_row + 1
    )

    if start < 0 or start >= height:
        raise ValueError(
            "The reference row lies outside the available image region."
        )

    available_depth = height - start

    if (
        require_full_depth
        and available_depth < depth_below_layer
    ):
        raise ValueError(
            f"Requested {depth_below_layer} rows below the boundary, "
            f"but only {available_depth} rows are available."
        )

    end = min(
        start + depth_below_layer,
        height,
    )

    crop = image[
        start:end,
        :
    ].copy()

    if crop.shape[0] < depth_below_layer:
        padded_crop = np.full(
            (
                depth_below_layer,
                width,
            ),
            fill_value,
            dtype=np.float32,
        )

        padded_crop[
            :crop.shape[0],
            :,
        ] = crop

        crop = padded_crop

    return crop.astype(
        np.float32
    )

def zscore_normalize(
    image: np.ndarray,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Apply whole-image z-score normalization.

    Each pixel is transformed using

        z = (intensity - mean) / standard deviation

    Parameters
    ----------
    image:
        Two-dimensional input image.
    epsilon:
        Small positive value used to avoid division by zero.

    Returns
    -------
    normalized:
        Float32 z-score-normalized image.
    statistics:
        Mean and standard deviation used for normalization.
    """
    image = np.asarray(
        image,
        dtype=np.float32,
    )

    finite_values = image[
        np.isfinite(image)
    ]

    if finite_values.size == 0:
        raise ValueError(
            "Cannot normalize an image containing no finite values."
        )

    mean_value = float(
        finite_values.mean()
    )

    standard_deviation = float(
        finite_values.std()
    )

    if standard_deviation <= epsilon:
        normalized = np.zeros_like(
            image,
            dtype=np.float32,
        )
    else:
        normalized = (
            image - mean_value
        ) / (
            standard_deviation + epsilon
        )

    return (
        normalized.astype(np.float32),
        {
            "mean": mean_value,
            "standard_deviation": standard_deviation,
            "epsilon": float(epsilon),
        },
    )

def percentile_normalize(
    image: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Apply whole-image percentile normalization to the range [0, 1].

    Parameters
    ----------
    image:
        Two-dimensional input image.
    lower_percentile:
        Lower clipping percentile.
    upper_percentile:
        Upper clipping percentile.

    Returns
    -------
    normalized:
        Float32 image scaled to [0, 1].
    statistics:
        Percentiles and intensity values used for normalization.
    """
    if not (
        0
        <= lower_percentile
        < upper_percentile
        <= 100
    ):
        raise ValueError(
            "Percentiles must satisfy "
            "0 <= lower_percentile < upper_percentile <= 100."
        )

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    finite_values = image[
        np.isfinite(image)
    ]

    if finite_values.size == 0:
        raise ValueError(
            "Cannot normalize an image containing no finite values."
        )

    lower_value = float(
        np.percentile(
            finite_values,
            lower_percentile,
        )
    )

    upper_value = float(
        np.percentile(
            finite_values,
            upper_percentile,
        )
    )

    if upper_value <= lower_value:
        normalized = np.zeros_like(
            image,
            dtype=np.float32,
        )
    else:
        normalized = (
            image - lower_value
        ) / (
            upper_value - lower_value
        )

        normalized = np.clip(
            normalized,
            0.0,
            1.0,
        )

    return (
        normalized.astype(np.float32),
        {
            "lower_percentile": float(
                lower_percentile
            ),
            "upper_percentile": float(
                upper_percentile
            ),
            "lower_intensity_value": lower_value,
            "upper_intensity_value": upper_value,
        },
    )

def minmax_normalize(
    image: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """
    Scale the full image intensity range to [0, 1].
    """
    image = np.asarray(
        image,
        dtype=np.float32,
    )

    finite_values = image[
        np.isfinite(image)
    ]

    if finite_values.size == 0:
        raise ValueError(
            "Cannot normalize an image containing no finite values."
        )

    minimum = float(
        finite_values.min()
    )

    maximum = float(
        finite_values.max()
    )

    if maximum <= minimum:
        normalized = np.zeros_like(
            image,
            dtype=np.float32,
        )
    else:
        normalized = (
            image - minimum
        ) / (
            maximum - minimum
        )

        normalized = np.clip(
            normalized,
            0.0,
            1.0,
        )

    return (
        normalized.astype(np.float32),
        {
            "minimum": minimum,
            "maximum": maximum,
        },
    )

def normalize_image(
    image: np.ndarray,
    method: str = "zscore",
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    zscore_epsilon: float = 1e-8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Normalize an image using a configurable method.

    Supported methods
    -----------------
    ``"zscore"``
        Whole-image z-score normalization.
    ``"percentile"``
        Percentile clipping and scaling to [0, 1].
    ``"minmax"``
        Full-range scaling to [0, 1].
    ``"none"``
        Return the image unchanged.
    """
    method = method.lower()

    if method == "zscore":
        normalized, statistics = (
            zscore_normalize(
                image=image,
                epsilon=zscore_epsilon,
            )
        )

    elif method == "percentile":
        normalized, statistics = (
            percentile_normalize(
                image=image,
                lower_percentile=lower_percentile,
                upper_percentile=upper_percentile,
            )
        )

    elif method == "minmax":
        normalized, statistics = (
            minmax_normalize(
                image=image,
            )
        )

    elif method == "none":
        normalized = np.asarray(
            image,
            dtype=np.float32,
        ).copy()

        statistics = {}

    else:
        raise ValueError(
            "normalization method must be one of: "
            "'zscore', 'percentile', 'minmax', or 'none'."
        )

    metadata = {
        "method": method,
        **statistics,
    }

    return (
        normalized.astype(np.float32),
        metadata,
    )

def denoise_image(
    image: np.ndarray,
    method: str = "gaussian",
    *,
    gaussian_sigma: (
        float
        | tuple[float, float]
    ) = (1.0, 0.5),
    median_size: (
        int
        | tuple[int, int]
    ) = 3,
    gaussian_mode: str = "reflect",
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Denoise a processed OCT image using a configurable method.

    Parameters
    ----------
    image:
        Two-dimensional normalized image.
    method:
        Denoising method. Supported values are ``"gaussian"``,
        ``"median"``, and ``"none"``.
    gaussian_sigma:
        Gaussian standard deviation. A tuple is interpreted as
        ``(depth_sigma, horizontal_sigma)``.
    median_size:
        Median-filter neighborhood size.
    gaussian_mode:
        Boundary mode used by the Gaussian filter.

    Returns
    -------
    denoised:
        Float32 denoised image.
    metadata:
        Denoising method and parameters.
    """
    image = np.asarray(
        image,
        dtype=np.float32,
    )

    if image.ndim != 2:
        raise ValueError(
            "Denoising requires a two-dimensional image."
        )

    method = method.lower()

    if method == "gaussian":
        sigma_array = np.asarray(
            gaussian_sigma,
            dtype=np.float64,
        )

        if sigma_array.ndim > 1:
            raise ValueError(
                "gaussian_sigma must be a scalar or a length-two tuple."
            )

        if sigma_array.size not in {
            1,
            2,
        }:
            raise ValueError(
                "gaussian_sigma must be a scalar or a length-two tuple."
            )

        if np.any(
            sigma_array < 0
        ):
            raise ValueError(
                "Gaussian sigma values must be nonnegative."
            )

        denoised = gaussian_filter(
            image,
            sigma=gaussian_sigma,
            mode=gaussian_mode,
        )

        metadata = {
            "method": method,
            "gaussian_sigma": (
                [
                    float(value)
                    for value in sigma_array
                ]
                if sigma_array.size == 2
                else float(
                    sigma_array.item()
                )
            ),
            "gaussian_mode": gaussian_mode,
        }

    elif method == "median":
        size_array = np.asarray(
            median_size,
            dtype=np.int64,
        )

        if size_array.ndim > 1:
            raise ValueError(
                "median_size must be an integer or a length-two tuple."
            )

        if size_array.size not in {
            1,
            2,
        }:
            raise ValueError(
                "median_size must be an integer or a length-two tuple."
            )

        if np.any(
            size_array <= 0
        ):
            raise ValueError(
                "Median-filter sizes must be positive."
            )

        denoised = median_filter(
            image,
            size=median_size,
            mode="reflect",
        )

        metadata = {
            "method": method,
            "median_size": (
                [
                    int(value)
                    for value in size_array
                ]
                if size_array.size == 2
                else int(
                    size_array.item()
                )
            ),
        }

    elif method == "none":
        denoised = image.copy()

        metadata = {
            "method": method,
        }

    else:
        raise ValueError(
            "denoise method must be one of: "
            "'gaussian', 'median', or 'none'."
        )

    return (
        np.asarray(
            denoised,
            dtype=np.float32,
        ),
        metadata,
    )

def preprocess_bscan(
    volume,
    bscan_index: int,
    *,
    layer_name: str = "BM",
    reference_row: int | None = None,
    flatten_fill_value: float = 0.0,
    depth_below_layer: int = 150,
    include_boundary: bool = True,
    require_full_depth: bool = False,
    crop_fill_value: float = 0.0,
    normalization_method: str = "zscore",
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
    zscore_epsilon: float = 1e-8,
    denoise_method: str = "gaussian",
    gaussian_sigma: (
        float
        | tuple[float, float]
    ) = (1.0, 0.5),
    median_size: (
        int
        | tuple[int, int]
    ) = 3,
    gaussian_mode: str = "reflect",
) -> PreprocessedScan:
    """
    Run the complete preprocessing pipeline for one OCT B-scan.

    Processing order
    ----------------
    1. Load the selected B-scan.
    2. Retrieve and interpolate the selected retinal-layer boundary.
    3. Flatten the B-scan to that boundary.
    4. Crop a fixed-depth region below the flattened boundary.
    5. Normalize the cropped image.
    6. Denoise the normalized image.
    """
    raw_bscan = load_bscan(
        volume=volume,
        bscan_index=bscan_index,
    )

    layer_boundary = get_layer_boundary(
        volume=volume,
        bscan_index=bscan_index,
        layer_name=layer_name,
    )

    flattened_bscan, resolved_reference_row = (
        flatten_to_boundary(
            image=raw_bscan,
            boundary=layer_boundary,
            reference_row=reference_row,
            fill_value=flatten_fill_value,
        )
    )

    available_depth = (
        flattened_bscan.shape[0]
        - (
            resolved_reference_row
            if include_boundary
            else resolved_reference_row + 1
        )
    )

    sub_layer_crop = crop_below_boundary(
        flattened_image=flattened_bscan,
        reference_row=resolved_reference_row,
        depth_below_layer=depth_below_layer,
        include_boundary=include_boundary,
        require_full_depth=require_full_depth,
        fill_value=crop_fill_value,
    )

    normalized_scan, normalization_metadata = (
        normalize_image(
            image=sub_layer_crop,
            method=normalization_method,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
            zscore_epsilon=zscore_epsilon,
        )
    )

    denoised_scan, denoising_metadata = (
        denoise_image(
            image=normalized_scan,
            method=denoise_method,
            gaussian_sigma=gaussian_sigma,
            median_size=median_size,
            gaussian_mode=gaussian_mode,
        )
    )

    requested_start_row = (
        resolved_reference_row
        if include_boundary
        else resolved_reference_row + 1
    )

    actual_unpadded_depth = max(
        0,
        min(
            depth_below_layer,
            raw_bscan.shape[0]
            - requested_start_row,
        ),
    )

    padded_rows = (
        depth_below_layer
        - actual_unpadded_depth
    )

    bscan_object = volume[
        bscan_index
    ]

    metadata = {
        "volume_shape": tuple(
            volume.shape
        ),
        "bscan_shape": tuple(
            raw_bscan.shape
        ),
        "layer_name": layer_name,
        "available_layers": (
            inspect_volume_layers(
                volume
            )
        ),
        "reference_row": int(
            resolved_reference_row
        ),
        "boundary_minimum": float(
            layer_boundary.min()
        ),
        "boundary_maximum": float(
            layer_boundary.max()
        ),
        "boundary_median": float(
            np.median(
                layer_boundary
            )
        ),
        "crop": {
            "depth_below_layer": int(
                depth_below_layer
            ),
            "include_boundary": bool(
                include_boundary
            ),
            "require_full_depth": bool(
                require_full_depth
            ),
            "available_depth": int(
                available_depth
            ),
            "actual_unpadded_depth": int(
                actual_unpadded_depth
            ),
            "padded_rows": int(
                padded_rows
            ),
            "fill_value": float(
                crop_fill_value
            ),
        },
        "normalization": (
            normalization_metadata
        ),
        "denoising": (
            denoising_metadata
        ),
        "bscan_meta": getattr(
            getattr(
                bscan_object,
                "meta",
                None,
            ),
            "_store",
            {},
        ),
    }

    return PreprocessedScan(
        bscan_index=int(
            bscan_index
        ),
        layer_name=layer_name,
        raw_bscan=raw_bscan,
        layer_boundary=layer_boundary,
        flattened_bscan=flattened_bscan,
        sub_layer_crop=sub_layer_crop,
        normalized_scan=normalized_scan,
        denoised_scan=denoised_scan,
        reference_row=int(
            resolved_reference_row
        ),
        depth_below_layer=int(
            depth_below_layer
        ),
        normalization_method=(
            normalization_method.lower()
        ),
        denoise_method=(
            denoise_method.lower()
        ),
        normalization_metadata=normalization_metadata,
        denoising_metadata=denoising_metadata,
        metadata=metadata,
    )
