from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import (
    gaussian_filter1d,
    map_coordinates,
)

@dataclass
class IntensityProfileResult:
    """
    Output from ImageJ-like line-profile extraction.

    Attributes
    ----------
    start:
        Resolved starting coordinate as ``(x, y)``.
    end:
        Resolved ending coordinate as ``(x, y)``.
    depth:
        Vertical depth of the profile center, measured in pixels from
        the top of the supplied B-scan.
    profile_margin:
        Number of rows above and below ``depth`` used when extracting
        the profile. A value of zero corresponds to a single horizontal
        line.
    aggregation:
        Aggregation method applied across the profile band. Supported
        values are ``"none"``, ``"mean"``, and ``"median"``.
    stepsize:
        Distance between sampled profile measurements, in pixels.
    distance:
        Distance along the extracted profile. This is ``None`` when
        ``data=False``.
    gray_values:
        Extracted grayscale intensity values. This is ``None`` when
        ``data=False``.
    profile_data:
        Two-column array containing distance and gray value. This is
        ``None`` when ``data=False``.
    figure:
        Matplotlib figure containing the marked B-scan and profile plot.
        This is ``None`` when ``plot=False``.
    plot_path:
        Location where the plot was saved, when applicable.
    data_path:
        Location where the numerical profile was saved, when applicable.
    metadata:
        Additional information about the image, visualization, and
        extraction settings.
    """

    start: tuple[float, float]
    end: tuple[float, float]

    depth: float
    profile_margin: int
    aggregation: str
    stepsize: float

    distance: np.ndarray | None
    raw_gray_values: np.ndarray | None # profile immediately after row aggregation
    gray_values: np.ndarray | None # profile used for display and analysis

    denoise_enabled: bool
    denoise_method: str
    denoise_sigma: float | None
    profile_data: np.ndarray | None

    figure: Any | None

    plot_path: Path | None
    data_path: Path | None

    metadata: dict[str, Any]


def _validate_bscan(bscan: np.ndarray) -> np.ndarray:
    """
    Validate and convert a B-scan to a two-dimensional float32 array.
    """
    image = np.asarray(bscan, dtype=np.float32)

    if image.ndim != 2:
        raise ValueError(
            f"Expected a two-dimensional B-scan, received shape "
            f"{image.shape}."
        )

    if image.size == 0:
        raise ValueError("The supplied B-scan is empty.")

    if not np.isfinite(image).any():
        raise ValueError(
            "The supplied B-scan contains no finite intensity values."
        )

    return image


def _convert_to_imagej_gray_values(
    image: np.ndarray,
) -> np.ndarray:
    """
    Convert normalized images to an ImageJ-like 8-bit gray-value scale.

    Images whose finite values lie entirely between 0 and 1 are assumed
    to be normalized and are multiplied by 255. Images already outside
    that range retain their original intensity scale.
    """
    image = np.asarray(image, dtype=np.float32)

    finite_values = image[np.isfinite(image)]

    if finite_values.size == 0:
        raise ValueError(
            "Cannot convert an image containing no finite values."
        )

    image_min = float(finite_values.min())
    image_max = float(finite_values.max())

    if image_min >= 0.0 and image_max <= 1.0:
        return image * 255.0

    return image.copy()


def _resolve_line_coordinates(
    image_shape: tuple[int, int],
    start: tuple[float, float],
    end: tuple[float, float],
    depth: float | None,
    clip_to_image: bool,
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """
    Resolve and validate the line coordinates.

    When ``depth`` is supplied, the y-coordinate of both endpoints is
    replaced by that value, producing a horizontal line.
    """
    height, width = image_shape

    if len(start) != 2 or len(end) != 2:
        raise ValueError(
            "start and end must each contain exactly two values: (x, y)."
        )

    start_x = float(start[0])
    start_y = float(start[1])
    end_x = float(end[0])
    end_y = float(end[1])

    if depth is not None:
        if not np.isfinite(depth):
            raise ValueError("depth must be finite.")

        start_y = float(depth)
        end_y = float(depth)

    elif not np.isclose(start_y, end_y):
        raise ValueError(
            "The ImageJ-like barcode profile must be horizontal. "
            "Provide endpoints with identical y-coordinates or supply "
            "the depth argument."
        )

    resolved_depth = start_y

    coordinates = np.array(
        [
            [start_x, start_y],
            [end_x, end_y],
        ],
        dtype=np.float64,
    )

    if not np.isfinite(coordinates).all():
        raise ValueError(
            "start and end coordinates must contain finite values."
        )

    outside_image = (
        start_x < 0
        or start_x > width - 1
        or end_x < 0
        or end_x > width - 1
        or start_y < 0
        or start_y > height - 1
        or end_y < 0
        or end_y > height - 1
    )

    if outside_image:
        if not clip_to_image:
            raise ValueError(
                "The requested profile line extends outside the supplied "
                f"B-scan. Image dimensions are width={width}, "
                f"height={height}; requested start={start}, end={end}, "
                f"depth={resolved_depth}."
            )

        warnings.warn(
            "The requested profile coordinates extend outside the "
            "supplied B-scan and will be clipped to the image boundaries.",
            stacklevel=2,
        )

        start_x = float(np.clip(start_x, 0, width - 1))
        end_x = float(np.clip(end_x, 0, width - 1))
        start_y = float(np.clip(start_y, 0, height - 1))
        end_y = float(np.clip(end_y, 0, height - 1))

        resolved_depth = start_y

    if np.isclose(start_x, end_x) and np.isclose(start_y, end_y):
        raise ValueError(
            "The profile line must have nonzero length."
        )

    return (
        (start_x, start_y),
        (end_x, end_y),
        resolved_depth,
    )


def _sample_single_profile(
    image: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
    stepsize: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample grayscale intensity along a line segment.

    Bilinear interpolation is used when the requested coordinates do not
    fall exactly on integer pixel locations.
    """
    if stepsize <= 0:
        raise ValueError("stepsize must be greater than zero.")

    start_x, start_y = start
    end_x, end_y = end

    delta_x = end_x - start_x
    delta_y = end_y - start_y

    line_length = float(
        np.hypot(delta_x, delta_y)
    )

    distances = np.arange(
        0.0,
        line_length,
        stepsize,
        dtype=np.float32,
    )

    if distances.size == 0 or not np.isclose(
        distances[-1],
        line_length,
    ):
        distances = np.append(
            distances,
            np.float32(line_length),
        )

    fractions = distances / line_length

    x_coordinates = start_x + fractions * delta_x
    y_coordinates = start_y + fractions * delta_y

    gray_values = map_coordinates(
        image,
        coordinates=np.vstack(
            [
                y_coordinates,
                x_coordinates,
            ]
        ),
        order=1,
        mode="nearest",
    )

    return (
        distances.astype(np.float32),
        gray_values.astype(np.float32),
    )

def _extract_band_profile(
    image: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
    stepsize: float,
    profile_margin: int = 0,
    aggregation: str = "none",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract a grayscale intensity profile from either a single
    horizontal line or a horizontal band.

    Parameters
    ----------
    image:
        Two-dimensional grayscale image.
    start:
        Starting coordinate as (x, y).
    end:
        Ending coordinate as (x, y).
    stepsize:
        Distance between sampled measurements.
    profile_margin:
        Number of pixels above and below the center profile to include.
        A value of zero extracts a single line profile.
    aggregation:
        Aggregation method across the band.

        Supported values are

        - "none"   : single profile only (margin must be zero)
        - "mean"   : average all sampled profiles
        - "median" : median of all sampled profiles

    Returns
    -------
    distance:
        Distance along the profile.
    gray_values:
        Extracted (or aggregated) grayscale profile.
    """
    if profile_margin < 0:
        raise ValueError(
            "profile_margin must be greater than or equal to zero."
        )

    aggregation = aggregation.lower()

    valid_methods = {"none", "mean", "median"}

    if aggregation not in valid_methods:
        raise ValueError(
            f"aggregation must be one of {sorted(valid_methods)}."
        )

    if profile_margin == 0:
        return _sample_single_profile(
            image=image,
            start=start,
            end=end,
            stepsize=stepsize,
        )

    if aggregation == "none":
        raise ValueError(
            "aggregation='none' may only be used when "
            "profile_margin=0."
        )

    start_x, start_y = start
    end_x, end_y = end

    image_height = image.shape[0]

    first_row = max(
        0,
        int(np.floor(start_y - profile_margin)),
    )

    last_row = min(
        image_height - 1,
        int(np.ceil(start_y + profile_margin)),
    )

    profiles = []

    distance = None

    for row in range(first_row, last_row + 1):

        current_start = (start_x, float(row))
        current_end = (end_x, float(row))

        current_distance, current_profile = (
            _sample_single_profile(
                image=image,
                start=current_start,
                end=current_end,
                stepsize=stepsize,
            )
        )

        if distance is None:
            distance = current_distance

        profiles.append(current_profile)

    profile_stack = np.vstack(profiles)

    if aggregation == "mean":
        gray_values = profile_stack.mean(axis=0)

    else:
        gray_values = np.median(profile_stack, axis=0)

    return (
        distance.astype(np.float32),
        gray_values.astype(np.float32),
    )

def _denoise_profile(
    gray_values: np.ndarray,
    *,
    enabled: bool = True,
    method: str = "gaussian",
    sigma: float = 1.5,
) -> np.ndarray:
    """
    Denoise a one-dimensional intensity profile.

    Parameters
    ----------
    gray_values:
        Extracted one-dimensional intensity profile.
    enabled:
        Whether denoising is applied.
    method:
        Denoising method. Currently supports ``"gaussian"`` and ``"none"``.
    sigma:
        Standard deviation of the Gaussian smoothing kernel, measured in
        profile samples. Larger values produce stronger smoothing.

    Returns
    -------
    np.ndarray
        Raw or denoised profile as float32.
    """
    values = np.asarray(
        gray_values,
        dtype=np.float32,
    )

    if not enabled or method == "none":
        return values.copy()

    if method != "gaussian":
        raise ValueError(
            "denoise_method must be 'gaussian' or 'none'."
        )

    if sigma <= 0:
        raise ValueError(
            "denoise_sigma must be greater than zero."
        )

    return gaussian_filter1d(
        values,
        sigma=sigma,
        mode="nearest",
    ).astype(np.float32)

def _create_profile_figure(
    image: np.ndarray,
    start: tuple[float, float],
    end: tuple[float, float],
    distance: np.ndarray,
    gray_values: np.ndarray,
    gray_value_limits: tuple[float, float],
    *,
    profile_margin: int = 0,
    overlay: bool = False,
    overlay_color: str = "cyan",
    overlay_alpha: float = 0.30,
):
    """
    Create the marked-image and intensity-profile visualization.
    """
    lower_limit, upper_limit = gray_value_limits

    if lower_limit >= upper_limit:
        raise ValueError(
            "gray_value_limits must satisfy lower < upper."
        )

    figure, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 8),
        gridspec_kw={
            "height_ratios": [1.25, 1.0],
        },
    )

    image_axis = axes[0]
    profile_axis = axes[1]

    image_axis.imshow(
        image,
        cmap="gray",
        aspect="auto",
    )

    if overlay and profile_margin > 0:

        center_y = start[1]

        image_axis.axhspan(
            center_y - profile_margin,
            center_y + profile_margin,
            color=overlay_color,
            alpha=overlay_alpha,
            linewidth=0,
            zorder=1,
        )

    image_axis.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color="yellow",
        linewidth=2,
    )

    image_axis.scatter(
        [start[0], end[0]],
        [start[1], end[1]],
        s=35,
        facecolors="white",
        edgecolors="yellow",
        linewidths=1.5,
        zorder=3,
    )

    if profile_margin == 0:
        title = "Processed B-scan with intensity-profile line"
    else:
        title = (
            "Processed B-scan with aggregated "
            f"{2 * profile_margin + 1}-pixel profile band"
        )

    image_axis.set_title(title)
    image_axis.set_xlabel("Horizontal position")
    image_axis.set_ylabel("Depth from top")

    profile_axis.plot(
        distance,
        gray_values,
        linewidth=1.0,
    )

    profile_axis.set_xlim(
        distance.min(),
        distance.max(),
    )
    
    profile_axis.set_ylim(
        lower_limit,
        upper_limit,
    )

    profile_axis.set_title(
        "ImageJ-like intensity profile"
    )
    profile_axis.set_xlabel(
        "Distance (pixels)"
    )
    profile_axis.set_ylabel(
        "Gray value"
    )
    profile_axis.grid(
        alpha=0.25,
    )

    figure.tight_layout()

    return figure


def extract_intensity_profile(
    bscan: np.ndarray,
    start: tuple[float, float] = (73, 135),
    depth: float | None = None,
    end: tuple[float, float] = (1169, 135),
    stepsize: float = 1.0,
    profile_margin: int = 0,
    aggregation: str = "none",
    denoise: bool = True,
    denoise_method: str = "gaussian",
    denoise_sigma: float = 1.5,
    plot: bool = True,
    data: bool = True,
    plot_path: str | Path | None = None,
    data_path: str | Path | None = None,
    gray_value_limits: tuple[float, float] = (0.0, 300.0),
    overlay: bool = False,
    overlay_color: str = "cyan",
    overlay_alpha: float = 0.30,
    clip_to_image: bool = True,
) -> IntensityProfileResult:
    """
    Extract an ImageJ-like grayscale profile from a processed B-scan.

    A line segment is placed on the supplied B-scan and grayscale
    intensity is sampled along that line. By default, the function
    reproduces the manually selected ImageJ line running from
    ``(73, 135)`` to ``(1169, 135)``.

    Parameters
    ----------
    bscan:
        Two-dimensional raw, flattened, cropped, or normalized B-scan.
    start:
        Starting line coordinate as ``(x, y)``. The default is
        ``(73, 135)``.
    depth:
        Vertical position of the line measured in pixels from the top
        of the supplied image. When provided, this replaces the
        y-coordinate in both ``start`` and ``end``.
    end:
        Ending line coordinate as ``(x, y)``. The default is
        ``(1169, 135)``.
    stepsize:
        Distance between profile measurements, in pixels. The default
        is one measurement per pixel.
    profile_margin:
        Number of pixels above and below the center profile used during
        extraction. A value of zero extracts a single horizontal profile.
    aggregation:
        Aggregation method applied across the extraction band. Supported
        values are "none", "mean", and "median".
    plot:
        Whether to generate the marked B-scan and profile plot.
    data:
        Whether to return the numerical distance and gray-value data.
    plot_path:
        Optional output path for the plot. When omitted, the figure is
        created but not saved.
    data_path:
        Optional CSV output path for the numerical profile. This
        requires ``data=True``.
    gray_value_limits:
        Fixed y-axis range for the profile plot. The default is
        ``(0, 300)`` so scans use a consistent display scale.
    overlay:
        Whether to display the extraction band on the B-scan.
    overlay_color:
        Color used for the extraction band overlay.
    overlay_alpha:
        Transparency of the extraction band overlay.
    clip_to_image:
        Whether endpoints outside the image should be clipped to valid
        image coordinates. This defaults to ``True``.
    

    Returns
    -------
    IntensityProfileResult
        Extracted profile, coordinates, optional numerical data, plot,
        and metadata.

    Notes
    -----
    If the supplied image is normalized to the range [0, 1], its
    intensities are automatically converted to an 8-bit-style scale
    from 0 to 255 before extracting the profile.

    The default endpoint ``x=1169`` comes from the original ImageJ
    screenshot. For a 512-pixel-wide EyePy B-scan, that coordinate lies
    outside the image and will therefore be clipped to ``x=511`` unless
    different coordinates are supplied.
    """
    image = _validate_bscan(bscan)

    gray_image = _convert_to_imagej_gray_values(
        image
    )

    resolved_start, resolved_end, resolved_depth = (
        _resolve_line_coordinates(
            image_shape=gray_image.shape,
            start=start,
            end=end,
            depth=depth,
            clip_to_image=clip_to_image,
        )
    )

    distance_values, raw_gray_values = _extract_band_profile(
        image=gray_image,
        start=resolved_start,
        end=resolved_end,
        stepsize=stepsize,
        profile_margin=profile_margin,
        aggregation=aggregation,
    )

    gray_values = _denoise_profile(
        raw_gray_values,
        enabled=denoise,
        method=denoise_method,
        sigma=denoise_sigma,
    )

    figure = None
    resolved_plot_path = None

    if plot:
        figure = _create_profile_figure(
            image=gray_image,
            start=resolved_start,
            end=resolved_end,
            distance=distance_values,
            gray_values=gray_values,
            gray_value_limits=gray_value_limits,
            profile_margin=profile_margin,
            overlay=overlay,
            overlay_color=overlay_color,
            overlay_alpha=overlay_alpha,
        )

        if plot_path is not None:
            resolved_plot_path = Path(plot_path)
            resolved_plot_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            figure.savefig(
                resolved_plot_path,
                dpi=300,
                bbox_inches="tight",
            )

    elif plot_path is not None:
        warnings.warn(
            "plot_path was supplied, but plot=False. No plot was saved.",
            stacklevel=2,
        )

    profile_data = None
    returned_distance = None
    returned_gray_values = None
    resolved_data_path = None

    if data:
        returned_distance = distance_values.copy()
        returned_gray_values = gray_values.copy()

        profile_data = np.column_stack(
            [
                returned_distance,
                returned_gray_values,
            ]
        ).astype(np.float32)

        if data_path is not None:
            resolved_data_path = Path(data_path)
            resolved_data_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            np.savetxt(
                resolved_data_path,
                profile_data,
                delimiter=",",
                header="distance_pixels,gray_value",
                comments="",
                fmt="%.6f",
            )

    elif data_path is not None:
        warnings.warn(
            "data_path was supplied, but data=False. No numerical data "
            "was saved.",
            stacklevel=2,
        )

    metadata = {
        "image_shape": tuple(gray_image.shape),
        "original_dtype": str(
            np.asarray(bscan).dtype
        ),
        "start": resolved_start,
        "end": resolved_end,
        "depth": resolved_depth,
        "stepsize": float(stepsize),
        "profile_margin": int(profile_margin),
        "aggregation": aggregation.lower(),
        "profile_band_start": max(
            0,
            int(resolved_depth - profile_margin),
        ),
        "profile_band_end": min(
            gray_image.shape[0] - 1,
            int(resolved_depth + profile_margin),
        ),
        "profile_band_height": int(2 * profile_margin + 1),
        "number_of_measurements": int(
            distance_values.size
        ),
        "line_length_pixels": (
            float(distance_values[-1])
            if distance_values.size > 0
            else 0.0
        ),
        "gray_value_min": float(
            gray_values.min()
        ),
        "gray_value_max": float(
            gray_values.max()
        ),
        "gray_value_mean": float(
            gray_values.mean()
        ),
        "gray_value_limits": tuple(
            float(value)
            for value in gray_value_limits
        ),
        "overlay": bool(overlay),
        "overlay_color": overlay_color,
        "overlay_alpha": float(overlay_alpha),
        "plot_created": bool(plot),
        "data_returned": bool(data),
        "denoise_enabled": bool(denoise),
        "denoise_method": (
            denoise_method
            if denoise
            else "none"
        ),
        "denoise_sigma": (
            float(denoise_sigma)
            if denoise
            else None
        ),
        "raw_gray_value_min": float(
            raw_gray_values.min()
        ),
        "raw_gray_value_max": float(
            raw_gray_values.max()
        ),
        "raw_gray_value_mean": float(
            raw_gray_values.mean()
        ),
    }

    return IntensityProfileResult(
        start=resolved_start,
        end=resolved_end,
        depth=resolved_depth,
        profile_margin=profile_margin,
        aggregation=aggregation,
        stepsize=float(stepsize),
        distance=returned_distance,
        gray_values=returned_gray_values,
        profile_data=profile_data,
        figure=figure,
        plot_path=resolved_plot_path,
        data_path=resolved_data_path,
        metadata=metadata,
        raw_gray_values=(
            raw_gray_values.copy()
            if data
            else None
        ),
        denoise_enabled=bool(denoise),
        denoise_method=(
            denoise_method
            if denoise
            else "none"
        ),
        denoise_sigma=(
            float(denoise_sigma)
            if denoise
            else None
        ),
    )
