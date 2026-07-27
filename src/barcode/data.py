from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class PreprocessedBscan:
    """Container for one preprocessed OCT B-scan."""

    bscan_index: int
    raw_bscan: np.ndarray
    bm_boundary: np.ndarray
    flattened_bscan: np.ndarray
    sub_bm_crop: np.ndarray

    # This contains either the normalized crop or the unchanged crop,
    # depending on whether normalization was enabled.
    normalized_crop: np.ndarray

    reference_row: int
    depth_below_bm: int

    normalization_enabled: bool
    lower_percentile: float | None
    upper_percentile: float | None

    metadata: dict[str, Any]


def load_e2e_volume(e2e_path: str | Path):
    """
    Load a Heidelberg E2E OCT volume with eyepy.

    Parameters
    ----------
    e2e_path:
        Path to the E2E file.

    Returns
    -------
    eyepy.EyeVolume
        Loaded OCT volume.
    """
    try:
        import eyepy as ep
    except ImportError as exc:
        raise ImportError(
            "eyepy is required to load E2E files. "
            "Install it with `pip install eyepy`."
        ) from exc

    e2e_path = Path(e2e_path)

    if not e2e_path.exists():
        raise FileNotFoundError(f"E2E file not found: {e2e_path}")

    volume = ep.import_heyex_e2e(str(e2e_path))

    return volume


def load_bscan(volume, bscan_index: int) -> np.ndarray:
    """
    Extract one B-scan as a 2D NumPy array.

    Parameters
    ----------
    volume:
        Loaded EyeVolume.
    bscan_index:
        Zero-based B-scan index.

    Returns
    -------
    np.ndarray
        B-scan with shape (height, width).
    """
    if bscan_index < 0 or bscan_index >= len(volume):
        raise IndexError(
            f"B-scan index {bscan_index} is outside valid range "
            f"0 to {len(volume) - 1}."
        )

    bscan = np.asarray(volume[bscan_index].data)

    if bscan.ndim != 2:
        raise ValueError(
            f"Expected a 2D B-scan, received shape {bscan.shape}."
        )

    return bscan.astype(np.float32)


def inspect_volume_layers(volume) -> list[str]:
    """
    Return the available layer-annotation names in an EyeVolume.

    This helper is useful because eyepy layer names may differ between
    imported datasets or software versions.
    """
    layers = getattr(volume, "layers", None)

    if layers is None:
        return []

    if isinstance(layers, dict):
        return list(layers.keys())

    try:
        return list(layers)
    except TypeError:
        return []


def get_bm_boundary(
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
    depth_below_bm: int = 150,
    include_boundary: bool = True,
) -> np.ndarray:
    """
    Crop a fixed-depth region at and below the flattened BM.

    Parameters
    ----------
    flattened_image:
        B-scan already flattened to BM.
    reference_row:
        Row occupied by BM after flattening.
    depth_below_bm:
        Number of pixels retained beneath BM.
    include_boundary:
        Whether the reference row itself should be included.
    """
    if depth_below_bm <= 0:
        raise ValueError("depth_below_bm must be positive.")

    height = flattened_image.shape[0]

    start = reference_row if include_boundary else reference_row + 1
    end = min(start + depth_below_bm, height)

    if start >= height:
        raise ValueError(
            "The reference row lies outside the available image region."
        )

    crop = flattened_image[start:end].copy()

    return crop


def robust_normalize(
    image: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> np.ndarray:
    """
    Robustly scale image intensities to [0, 1].

    Intensities below the selected lower percentile are mapped to 0.
    Intensities above the selected upper percentile are mapped to 1.

    Parameters
    ----------
    image:
        Input image.
    lower_percentile:
        Lower clipping percentile.
    upper_percentile:
        Upper clipping percentile.

    Returns
    -------
    np.ndarray
        Float32 image scaled to [0, 1].
    """
    image = np.asarray(image, dtype=np.float32)

    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError(
            "Percentiles must satisfy "
            "0 <= lower_percentile < upper_percentile <= 100."
        )

    finite_values = image[np.isfinite(image)]

    if finite_values.size == 0:
        raise ValueError(
            "Cannot normalize an image containing no finite values."
        )

    lower = float(
        np.percentile(
            finite_values,
            lower_percentile,
        )
    )

    upper = float(
        np.percentile(
            finite_values,
            upper_percentile,
        )
    )

    if upper <= lower:
        return np.zeros_like(
            image,
            dtype=np.float32,
        )

    normalized = (
        image - lower
    ) / (
        upper - lower
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0,
    )

    return normalized.astype(np.float32)


def preprocess_bscan(
    volume,
    bscan_index: int,
    bm_layer_name: str = "BM",
    depth_below_bm: int = 150,
    reference_row: int | None = None,
    normalize: bool = True,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> PreprocessedBscan:
    """
    Run the full preprocessing pipeline for one OCT B-scan.

    Parameters
    ----------
    volume:
        Loaded eyepy EyeVolume.
    bscan_index:
        Zero-based index of the requested B-scan.
    bm_layer_name:
        Name of the retinal boundary used for flattening.
    depth_below_bm:
        Number of pixels retained at and below the flattened BM.
    reference_row:
        Optional row onto which BM is flattened. If omitted, the median
        BM location is used.
    normalize:
        Whether to apply robust percentile normalization to the sub-BM crop.
    lower_percentile:
        Lower clipping percentile used when normalization is enabled.
    upper_percentile:
        Upper clipping percentile used when normalization is enabled.

    Returns
    -------
    PreprocessedBscan
        Container holding the original image, flattened image, crop,
        optionally normalized crop, and associated metadata.
    """
    raw_bscan = load_bscan(
        volume=volume,
        bscan_index=bscan_index,
    )

    bm_boundary = get_bm_boundary(
        volume=volume,
        bscan_index=bscan_index,
        layer_name=bm_layer_name,
    )

    flattened_bscan, resolved_reference_row = flatten_to_boundary(
        image=raw_bscan,
        boundary=bm_boundary,
        reference_row=reference_row,
    )

    sub_bm_crop = crop_below_boundary(
        flattened_image=flattened_bscan,
        reference_row=resolved_reference_row,
        depth_below_bm=depth_below_bm,
    )

    if normalize:
        normalized_crop = robust_normalize(
            image=sub_bm_crop,
            lower_percentile=lower_percentile,
            upper_percentile=upper_percentile,
        )

        recorded_lower_percentile: float | None = lower_percentile
        recorded_upper_percentile: float | None = upper_percentile

    else:
        # Preserve the original crop without modifying its intensity scale.
        normalized_crop = sub_bm_crop.copy().astype(np.float32)

        recorded_lower_percentile = None
        recorded_upper_percentile = None

    bscan_object = volume[bscan_index]

    metadata = {
        "volume_shape": tuple(volume.shape),
        "bscan_shape": tuple(raw_bscan.shape),
        "available_layers": inspect_volume_layers(volume),
        "bscan_meta": getattr(
            bscan_object.meta,
            "_store",
            {},
        ),
        "normalization": {
            "enabled": normalize,
            "method": (
                "robust_percentile"
                if normalize
                else "none"
            ),
            "lower_percentile": recorded_lower_percentile,
            "upper_percentile": recorded_upper_percentile,
        },
    }

    return PreprocessedBscan(
        bscan_index=bscan_index,
        raw_bscan=raw_bscan,
        bm_boundary=bm_boundary,
        flattened_bscan=flattened_bscan,
        sub_bm_crop=sub_bm_crop,
        normalized_crop=normalized_crop,
        reference_row=resolved_reference_row,
        depth_below_bm=depth_below_bm,
        normalization_enabled=normalize,
        lower_percentile=recorded_lower_percentile,
        upper_percentile=recorded_upper_percentile,
        metadata=metadata,
    )