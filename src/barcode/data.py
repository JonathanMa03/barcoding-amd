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
    normalized_crop: np.ndarray
    reference_row: int
    depth_below_bm: int
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
    Extract the Bruch's membrane boundary for one B-scan.

    Parameters
    ----------
    volume:
        Loaded EyeVolume.
    bscan_index:
        Zero-based B-scan index.
    layer_name:
        Name used by eyepy for Bruch's membrane.

    Returns
    -------
    np.ndarray
        One vertical coordinate per image column.

    Notes
    -----
    The exact eyepy API for imported HEYEX layers can vary. This function
    attempts common storage patterns and raises an informative error if the
    layer cannot be located.
    """
    available_layers = inspect_volume_layers(volume)

    candidate = None

    if hasattr(volume, "layers"):
        layers = volume.layers

        if isinstance(layers, dict) and layer_name in layers:
            candidate = layers[layer_name]

        elif hasattr(layers, "__getitem__"):
            try:
                candidate = layers[layer_name]
            except (KeyError, TypeError, IndexError):
                candidate = None

    if candidate is None and hasattr(volume, "layer_annotations"):
        annotations = volume.layer_annotations

        if isinstance(annotations, dict) and layer_name in annotations:
            candidate = annotations[layer_name]

    if candidate is None:
        raise KeyError(
            f"Could not find layer '{layer_name}'. "
            f"Available layers: {available_layers}"
        )

    layer_array = np.asarray(candidate)

    if layer_array.ndim == 2:
        boundary = layer_array[bscan_index]
    elif layer_array.ndim == 3:
        boundary = layer_array[bscan_index, 0]
    else:
        raise ValueError(
            f"Unsupported layer-array shape for '{layer_name}': "
            f"{layer_array.shape}"
        )

    boundary = boundary.astype(np.float32)

    return interpolate_missing_boundary(boundary)


def interpolate_missing_boundary(boundary: np.ndarray) -> np.ndarray:
    """
    Fill missing or non-finite layer coordinates by linear interpolation.
    """
    boundary = np.asarray(boundary, dtype=np.float32).copy()

    x = np.arange(boundary.size)
    valid = np.isfinite(boundary)

    if valid.sum() < 2:
        raise ValueError(
            "Boundary contains fewer than two valid coordinates."
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
    """
    image = np.asarray(image, dtype=np.float32)

    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("Percentiles must satisfy 0 <= lower < upper <= 100.")

    lower = float(np.percentile(image, lower_percentile))
    upper = float(np.percentile(image, upper_percentile))

    if upper <= lower:
        return np.zeros_like(image, dtype=np.float32)

    normalized = (image - lower) / (upper - lower)
    normalized = np.clip(normalized, 0.0, 1.0)

    return normalized.astype(np.float32)


def preprocess_bscan(
    volume,
    bscan_index: int,
    bm_layer_name: str = "BM",
    depth_below_bm: int = 150,
    reference_row: int | None = None,
) -> PreprocessedBscan:
    """
    Run the full preprocessing pipeline for one OCT B-scan.
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

    normalized_crop = robust_normalize(sub_bm_crop)

    bscan_object = volume[bscan_index]

    metadata = {
        "volume_shape": tuple(volume.shape),
        "bscan_shape": tuple(raw_bscan.shape),
        "available_layers": inspect_volume_layers(volume),
        "bscan_meta": getattr(bscan_object.meta, "_store", {}),
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
        metadata=metadata,
    )