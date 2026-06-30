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