from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, SpanSelector
from src.barcode.model_threshold import (
    extract_intensity_profile,
)

DEFAULT_LABEL_COLORS: dict[str, str] = {
    "Normal": "tab:green",
    "Early Atrophy (EA)": "tab:orange",
    "Barcoding": "tab:red",
}


@dataclass
class AnnotationInterval:
    """
    One manually annotated horizontal interval.

    Attributes
    ----------
    label:
        Assigned annotation class.
    x_start:
        Left endpoint in horizontal image coordinates.
    x_end:
        Right endpoint in horizontal image coordinates.
    width_pixels:
        Horizontal interval width in pixels.
    """

    label: str
    x_start: float
    x_end: float
    width_pixels: float


class ManualProfileAnnotator:
    """
    Interactive annotation interface for an OCT scan and intensity profile.

    The processed scan and its intensity profile are displayed with a
    shared horizontal axis. Users select a class and drag across either
    panel to mark a contiguous horizontal interval.

    Parameters
    ----------
    image:
        Two-dimensional processed OCT image.
    profile:
        Intensity-profile result returned by
        ``extract_intensity_profile``.
    label_colors:
        Mapping from annotation labels to Matplotlib colors.
    source_metadata:
        Optional dictionary containing scan, preprocessing, or source-file
        metadata.
    save_enabled:
        Whether JSON saving is enabled.
    output_path:
        Optional default JSON output path.
    overwrite:
        Whether an existing output file may be overwritten.
    figure_size:
        Matplotlib figure size.
    gray_value_limits:
        Optional fixed limits for the profile y-axis.
    """

    def __init__(
        self,
        image: np.ndarray,
        profile: Any,
        *,
        label_colors: Mapping[str, str] | None = None,
        source_metadata: Mapping[str, Any] | None = None,
        save_enabled: bool = True,
        output_path: str | Path | None = None,
        overwrite: bool = False,
        figure_size: tuple[float, float] = (14, 8),
        gray_value_limits: tuple[float, float] | None = (0.0, 300.0),
    ) -> None:
        self.image = self._validate_image(image)
        self.profile = profile

        self.label_colors = dict(
            DEFAULT_LABEL_COLORS
            if label_colors is None
            else label_colors
        )

        if not self.label_colors:
            raise ValueError(
                "label_colors must contain at least one annotation class."
            )

        self.source_metadata = dict(
            source_metadata
            if source_metadata is not None
            else {}
        )

        self.save_enabled = bool(save_enabled)
        self.output_path = (
            Path(output_path)
            if output_path is not None
            else None
        )
        self.overwrite = bool(overwrite)
        self.figure_size = figure_size
        self.gray_value_limits = gray_value_limits

        self.intervals: list[AnnotationInterval] = []
        self._artists: list[tuple[Any, Any]] = []

        self.selected_label = next(
            iter(self.label_colors)
        )

        self._validate_profile()
        self.profile_x = self._resolve_profile_x()
        self.profile_y = np.asarray(
            self.profile.gray_values,
            dtype=np.float32,
        )

        self.figure: Any | None = None
        self.scan_axis: Any | None = None
        self.profile_axis: Any | None = None
        self.control_axis: Any | None = None
        self.status_text: Any | None = None

        self.radio_buttons: Any | None = None
        self.undo_button: Any | None = None
        self.clear_button: Any | None = None
        self.move_line_button: Any | None = None
        self.save_button: Any | None = None

        self.scan_selector: Any | None = None
        self.profile_selector: Any | None = None
        self._line_placement_enabled = False
        self._click_connection: int | None = None

        self._profile_center_artist: Any | None = None
        self._profile_band_artist: Any | None = None
        self._profile_curve_artist: Any | None = None

        self.profile_depth_source = "configured_default"

        self.last_saved_path: Path | None = None

        self._create_interface()

    @staticmethod
    def _validate_image(
        image: np.ndarray,
    ) -> np.ndarray:
        """Validate and convert the supplied image to float32."""

        array = np.asarray(
            image,
            dtype=np.float32,
        )

        if array.ndim != 2:
            raise ValueError(
                "Manual annotation requires a two-dimensional image; "
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

        return array

    def _validate_profile(self) -> None:
        """Validate the supplied intensity-profile result."""

        required_attributes = {
            "start",
            "end",
            "depth",
            "stepsize",
            "profile_margin",
            "aggregation",
            "distance",
            "gray_values",
        }

        missing = [
            attribute
            for attribute in required_attributes
            if not hasattr(self.profile, attribute)
        ]

        if missing:
            raise TypeError(
                "profile is missing required attributes: "
                f"{missing}"
            )

        if (
            self.profile.distance is None
            or self.profile.gray_values is None
        ):
            raise ValueError(
                "The profile must be generated with data=True before "
                "manual annotation."
            )

        distance = np.asarray(
            self.profile.distance,
            dtype=np.float32,
        )

        gray_values = np.asarray(
            self.profile.gray_values,
            dtype=np.float32,
        )

        if distance.ndim != 1 or gray_values.ndim != 1:
            raise ValueError(
                "Profile distance and gray values must both be "
                "one-dimensional."
            )

        if distance.size != gray_values.size:
            raise ValueError(
                "Profile distance and gray values must have equal lengths."
            )

        if distance.size == 0:
            raise ValueError(
                "The supplied profile contains no measurements."
            )

    def _resolve_profile_x(self) -> np.ndarray:
        """
        Convert line distances to horizontal image coordinates.

        This supports profiles extracted from left to right or right to
        left.
        """

        distances = np.asarray(
            self.profile.distance,
            dtype=np.float32,
        )

        start_x = float(
            self.profile.start[0]
        )

        end_x = float(
            self.profile.end[0]
        )

        line_length = float(
            distances[-1]
        )

        if line_length <= 0:
            return np.full_like(
                distances,
                start_x,
                dtype=np.float32,
            )

        fractions = distances / line_length

        return (
            start_x
            + fractions * (end_x - start_x)
        ).astype(np.float32)

    def _create_interface(self) -> None:
        """Build the interactive Matplotlib annotation interface."""

        self.figure = plt.figure(
            figsize=self.figure_size,
        )

        grid = self.figure.add_gridspec(
            nrows=2,
            ncols=2,
            width_ratios=[5.5, 1.25],
            height_ratios=[1.4, 1.0],
            hspace=0.10,
            wspace=0.15,
        )

        self.scan_axis = self.figure.add_subplot(
            grid[0, 0]
        )

        self.profile_axis = self.figure.add_subplot(
            grid[1, 0],
            sharex=self.scan_axis,
        )

        self.control_axis = self.figure.add_subplot(
            grid[:, 1]
        )

        self._draw_scan()
        self._draw_profile()
        self._create_controls()
        self._create_selectors()

        self.figure.canvas.draw_idle()

    def _draw_scan(self) -> None:
        """Display the processed scan and extraction region."""

        image_height, image_width = self.image.shape

        self.scan_axis.imshow(
            self.image,
            cmap="gray",
            aspect="auto",
            extent=(
                -0.5,
                image_width - 0.5,
                image_height - 0.5,
                -0.5,
            ),
        )

        profile_depth = float(
            self.profile.depth
        )

        profile_margin = int(
            self.profile.profile_margin
        )

        self._profile_band_artist = None

        if profile_margin > 0:
            band_start = max(
                -0.5,
                profile_depth
                - profile_margin
                - 0.5,
            )

            band_end = min(
                image_height - 0.5,
                profile_depth
                + profile_margin
                + 0.5,
            )

            self._profile_band_artist = (
                self.scan_axis.axhspan(
                    band_start,
                    band_end,
                    alpha=0.20,
                    label=(
                        f"{2 * profile_margin + 1}-pixel "
                        "extraction band"
                    ),
                )
            )

        self._profile_center_artist = (
            self.scan_axis.axhline(
                profile_depth,
                linestyle="--",
                linewidth=1.5,
                label="Profile center",
            )
        )

        self.scan_axis.set_title(
            "Processed sub-BM region with manual annotations"
        )

        self.scan_axis.set_ylabel(
            "Depth below BM"
        )

        self.scan_axis.set_xlim(
            -0.5,
            image_width - 0.5,
        )

        self.scan_axis.legend(
            loc="upper right",
        )

        plt.setp(
            self.scan_axis.get_xticklabels(),
            visible=False,
        )

    def _draw_profile(self) -> None:
        """Display the extracted intensity profile."""

        (
            self._profile_curve_artist,
        ) = self.profile_axis.plot(
            self.profile_x,
            self.profile_y,
            linewidth=1.0,
        )

        self.profile_axis.set_title(
            "Intensity profile"
        )

        self.profile_axis.set_xlabel(
            "Horizontal position"
        )

        self.profile_axis.set_ylabel(
            "Gray value"
        )

        self.profile_axis.grid(
            alpha=0.25,
        )

        self.profile_axis.set_xlim(
            -0.5,
            self.image.shape[1] - 0.5,
        )

        if self.gray_value_limits is not None:
            lower, upper = self.gray_value_limits

            if lower >= upper:
                raise ValueError(
                    "gray_value_limits must satisfy lower < upper."
                )

            self.profile_axis.set_ylim(
                lower,
                upper,
            )

    def _create_controls(self) -> None:
        """Create annotation-class and action controls."""

        self.control_axis.set_title(
            "Manual annotation",
            pad=15,
        )

        self.control_axis.set_xticks([])
        self.control_axis.set_yticks([])

        radio_axis = self.control_axis.inset_axes(
            [0.05, 0.64, 0.90, 0.27]
        )

        self.radio_buttons = RadioButtons(
            radio_axis,
            tuple(self.label_colors),
            active=0,
        )

        move_line_axis = self.control_axis.inset_axes(
            [0.10, 0.49, 0.80, 0.08]
        )

        undo_axis = self.control_axis.inset_axes(
            [0.10, 0.38, 0.80, 0.08]
        )

        clear_axis = self.control_axis.inset_axes(
            [0.10, 0.27, 0.80, 0.08]
        )

        self.move_line_button = Button(
            move_line_axis,
            "Move profile line",
        )

        self.undo_button = Button(
            undo_axis,
            "Undo last",
        )

        self.clear_button = Button(
            clear_axis,
            "Clear all",
        )

        if self.save_enabled:
            save_axis = self.control_axis.inset_axes(
                [0.10, 0.16, 0.80, 0.08]
            )

            self.save_button = Button(
                save_axis,
                "Save JSON",
            )

            status_y = 0.055

        else:
            self.save_button = None
            status_y = 0.12

        self.status_text = self.control_axis.text(
            0.5,
            status_y,
            (
                f"Selected: {self.selected_label}\n"
                "Drag horizontally to annotate."
            ),
            ha="center",
            va="center",
            transform=self.control_axis.transAxes,
            wrap=True,
        )

        self.radio_buttons.on_clicked(
            self._set_selected_label
        )

        self.move_line_button.on_clicked(
            self._toggle_line_placement
        )

        self.undo_button.on_clicked(
            self._handle_undo
        )

        self.clear_button.on_clicked(
            self._handle_clear
        )

        if self.save_button is not None:
            self.save_button.on_clicked(
                self._handle_save
            )

        self._click_connection = (
            self.figure.canvas.mpl_connect(
                "button_press_event",
                self._handle_profile_depth_click,
            )
        )

    def _create_selectors(self) -> None:
        """Allow interval selection from either aligned panel."""

        selector_properties = {
            "alpha": 0.20,
            "facecolor": "tab:blue",
        }

        self.scan_selector = SpanSelector(
            self.scan_axis,
            self._select_interval,
            direction="horizontal",
            useblit=True,
            interactive=True,
            drag_from_anywhere=True,
            props=selector_properties,
        )

        self.profile_selector = SpanSelector(
            self.profile_axis,
            self._select_interval,
            direction="horizontal",
            useblit=True,
            interactive=True,
            drag_from_anywhere=True,
            props=selector_properties,
        )

    def _toggle_line_placement(
        self,
        _event: Any,
    ) -> None:
        """Enable or disable manual profile-depth selection."""

        self._line_placement_enabled = (
            not self._line_placement_enabled
        )

        if self._line_placement_enabled:
            self.move_line_button.label.set_text(
                "Click scan to place"
            )

            # Temporarily disable interval annotation so the scan click
            # is interpreted only as a profile-depth selection.
            self.scan_selector.set_active(False)
            self.profile_selector.set_active(False)

            self._set_status(
                "Click vertically on the scan\n"
                "to choose the profile depth."
            )

        else:
            self.move_line_button.label.set_text(
                "Move profile line"
            )

            self.scan_selector.set_active(True)
            self.profile_selector.set_active(True)

            self._set_status(
                f"Selected: {self.selected_label}\n"
                "Drag horizontally to annotate."
            )


    def _handle_profile_depth_click(
        self,
        event: Any,
    ) -> None:
        """Move the extraction band to the clicked scan depth."""

        if not self._line_placement_enabled:
            return

        if event.inaxes is not self.scan_axis:
            return

        if event.ydata is None:
            return

        selected_depth = int(
            np.clip(
                round(event.ydata),
                0,
                self.image.shape[0] - 1,
            )
        )

        self._recompute_profile(
            selected_depth
        )

        self._line_placement_enabled = False
        self.profile_depth_source = "manual"

        self.move_line_button.label.set_text(
            "Move profile line"
        )

        self.scan_selector.set_active(True)
        self.profile_selector.set_active(True)

        self._set_status(
            f"Profile moved to depth {selected_depth} px.\n"
            "Drag horizontally to annotate."
        )


    def _recompute_profile(
        self,
        selected_depth: int,
    ) -> None:
        """Re-extract and redraw the profile at a new depth."""

        image_width = self.image.shape[1]

        denoise_enabled = bool(
            getattr(
                self.profile,
                "denoise_enabled",
                False,
            )
        )

        denoise_method = str(
            getattr(
                self.profile,
                "denoise_method",
                "none",
            )
        )

        denoise_sigma = getattr(
            self.profile,
            "denoise_sigma",
            None,
        )

        if denoise_sigma is None:
            denoise_sigma = 1.5

        new_profile = extract_intensity_profile(
            bscan=self.image,
            start=(0, selected_depth),
            depth=selected_depth,
            end=(image_width - 1, selected_depth),
            stepsize=float(
                self.profile.stepsize
            ),
            profile_margin=int(
                self.profile.profile_margin
            ),
            aggregation=str(
                self.profile.aggregation
            ),
            denoise=denoise_enabled,
            denoise_method=denoise_method,
            denoise_sigma=float(
                denoise_sigma
            ),
            plot=False,
            data=True,
            overlay=False,
        )

        self.profile = new_profile
        self.profile_x = self._resolve_profile_x()
        self.profile_y = np.asarray(
            new_profile.gray_values,
            dtype=np.float32,
        )

        # Move the center line.
        self._profile_center_artist.set_ydata(
            [
                selected_depth,
                selected_depth,
            ]
        )

        # Replace the old extraction-band shading.
        if self._profile_band_artist is not None:
            self._profile_band_artist.remove()
            self._profile_band_artist = None

        profile_margin = int(
            new_profile.profile_margin
        )

        if profile_margin > 0:
            band_start = max(
                -0.5,
                selected_depth
                - profile_margin
                - 0.5,
            )

            band_end = min(
                self.image.shape[0] - 0.5,
                selected_depth
                + profile_margin
                + 0.5,
            )

            self._profile_band_artist = (
                self.scan_axis.axhspan(
                    band_start,
                    band_end,
                    alpha=0.20,
                    label=(
                        f"{2 * profile_margin + 1}-pixel "
                        "extraction band"
                    ),
                )
            )

        # Replace the displayed profile values.
        self._profile_curve_artist.set_data(
            self.profile_x,
            self.profile_y,
        )

        self.profile_axis.set_xlim(
            -0.5,
            self.image.shape[1] - 0.5,
        )

        if self.gray_value_limits is None:
            self.profile_axis.relim()
            self.profile_axis.autoscale_view()

        self.scan_axis.legend(
            loc="upper right",
        )

        self.figure.canvas.draw_idle()

    def _set_selected_label(
        self,
        label: str,
    ) -> None:
        """Set the active annotation class."""

        self.selected_label = label

        self._set_status(
            f"Selected: {label}\n"
            "Drag horizontally to annotate."
        )

    def _select_interval(
        self,
        x_min: float,
        x_max: float,
    ) -> None:
        """Create an interval from a horizontal drag selection."""

        if x_min is None or x_max is None:
            return

        image_width = self.image.shape[1]

        x_start = float(
            np.clip(
                min(x_min, x_max),
                0,
                image_width - 1,
            )
        )

        x_end = float(
            np.clip(
                max(x_min, x_max),
                0,
                image_width - 1,
            )
        )

        if x_end <= x_start:
            self._set_status(
                "Selection must have positive width."
            )
            return

        interval = AnnotationInterval(
            label=self.selected_label,
            x_start=x_start,
            x_end=x_end,
            width_pixels=x_end - x_start,
        )

        self.intervals.append(
            interval
        )

        self._draw_interval(
            interval
        )

        self._set_status(
            f"Added {interval.label}\n"
            f"x={interval.x_start:.1f}–"
            f"{interval.x_end:.1f}"
        )

    def _draw_interval(
        self,
        interval: AnnotationInterval,
    ) -> None:
        """Shade an annotation interval on both panels."""

        color = self.label_colors[
            interval.label
        ]

        scan_artist = self.scan_axis.axvspan(
            interval.x_start,
            interval.x_end,
            color=color,
            alpha=0.25,
        )

        profile_artist = self.profile_axis.axvspan(
            interval.x_start,
            interval.x_end,
            color=color,
            alpha=0.25,
        )

        self._artists.append(
            (
                scan_artist,
                profile_artist,
            )
        )

        self.figure.canvas.draw_idle()

    def undo(self) -> AnnotationInterval | None:
        """
        Remove and return the most recently created annotation.
        """

        if not self.intervals:
            self._set_status(
                "There are no annotations to undo."
            )
            return None

        removed_interval = self.intervals.pop()

        scan_artist, profile_artist = (
            self._artists.pop()
        )

        scan_artist.remove()
        profile_artist.remove()

        self._set_status(
            "Removed the most recent annotation."
        )

        return removed_interval

    def clear(self) -> None:
        """Remove all annotation intervals."""

        for scan_artist, profile_artist in self._artists:
            scan_artist.remove()
            profile_artist.remove()

        self.intervals.clear()
        self._artists.clear()

        self._set_status(
            "All annotations cleared."
        )

    def summary(self) -> dict[str, dict[str, float | int]]:
        """
        Return class-level annotation counts and total widths.
        """

        summary: dict[
            str,
            dict[str, float | int],
        ] = {}

        for label in self.label_colors:
            matching = [
                interval
                for interval in self.intervals
                if interval.label == label
            ]

            summary[label] = {
                "interval_count": len(
                    matching
                ),
                "total_width_pixels": float(
                    sum(
                        interval.width_pixels
                        for interval in matching
                    )
                ),
            }

        return summary

    def to_dict(self) -> dict[str, Any]:
        """
        Convert annotations and metadata into a JSON-compatible record.
        """

        profile_metadata = {
            "depth": float(
                self.profile.depth
            ),
            "depth_source": self.profile_depth_source,
            "start": [
                float(value)
                for value in self.profile.start
            ],
            "end": [
                float(value)
                for value in self.profile.end
            ],
            "stepsize": float(
                self.profile.stepsize
            ),
            "profile_margin": int(
                self.profile.profile_margin
            ),
            "aggregation": str(
                self.profile.aggregation
            ),
            "number_of_measurements": int(
                self.profile_y.size
            ),
            "denoise_enabled": bool(
                getattr(
                    self.profile,
                    "denoise_enabled",
                    False,
                )
            ),
            "denoise_method": str(
                getattr(
                    self.profile,
                    "denoise_method",
                    "none",
                )
            ),
            "denoise_sigma": getattr(
                self.profile,
                "denoise_sigma",
                None,
            ),
        }

        return {
            "created_at_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "image_shape": [
                int(value)
                for value in self.image.shape
            ],
            "profile": profile_metadata,
            "source_metadata": self._make_json_compatible(
                self.source_metadata
            ),
            "annotations": [
                asdict(interval)
                for interval in self.intervals
            ],
            "summary": self.summary(),
        }

    def save(
        self,
        output_path: str | Path | None = None,
        *,
        overwrite: bool | None = None,
    ) -> Path:
        """
        Save annotations and metadata to JSON.

        Parameters
        ----------
        output_path:
            Optional output path. If omitted, the path supplied during
            initialization is used. If neither is available, a default
            filename is created in the current working directory.
        overwrite:
            Optional override of the object's overwrite setting.

        Returns
        -------
        Path
            Path to the saved JSON file.
        """

        if not self.save_enabled:
            raise RuntimeError(
                "Saving is disabled for this annotator."
            )

        resolved_path = self._resolve_output_path(
            output_path
        )

        resolved_overwrite = (
            self.overwrite
            if overwrite is None
            else bool(overwrite)
        )

        if (
            resolved_path.exists()
            and not resolved_overwrite
        ):
            raise FileExistsError(
                f"Annotation file already exists: {resolved_path}. "
                "Set overwrite=True to replace it."
            )

        resolved_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with resolved_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.to_dict(),
                file,
                indent=2,
            )

        self.last_saved_path = resolved_path

        self._set_status(
            f"Saved {len(self.intervals)} intervals\n"
            f"{resolved_path.name}"
        )

        return resolved_path

    def show(self) -> None:
        """Display the interactive annotation interface."""

        plt.show()

    def _handle_undo(
        self,
        _event: Any,
    ) -> None:
        self.undo()

    def _handle_clear(
        self,
        _event: Any,
    ) -> None:
        self.clear()

    def _handle_save(
        self,
        _event: Any,
    ) -> None:
        try:
            self.save()
        except Exception as exc:
            self._set_status(
                f"Save failed:\n{exc}"
            )

    def _set_status(
        self,
        message: str,
    ) -> None:
        """Update the interface status message."""

        if self.status_text is not None:
            self.status_text.set_text(
                message
            )

        if self.figure is not None:
            self.figure.canvas.draw_idle()

    def _resolve_output_path(
        self,
        output_path: str | Path | None,
    ) -> Path:
        """Resolve an explicit, configured, or generated output path."""

        if output_path is not None:
            path = Path(
                output_path
            )

        elif self.output_path is not None:
            path = self.output_path

        else:
            source_name = self.source_metadata.get(
                "source_name",
                "manual",
            )

            bscan_index = self.source_metadata.get(
                "bscan_index",
            )

            source_stem = Path(
                str(source_name)
            ).stem

            if bscan_index is None:
                filename = (
                    f"{source_stem}_annotations.json"
                )
            else:
                filename = (
                    f"{source_stem}_"
                    f"bscan_{int(bscan_index):03d}_"
                    "annotations.json"
                )

            path = Path.cwd() / filename

        if path.suffix.lower() != ".json":
            path = path.with_suffix(
                ".json"
            )

        return path.resolve()

    @staticmethod
    def _make_json_compatible(
        value: Any,
    ) -> Any:
        """Recursively convert common NumPy and Path values for JSON."""

        if isinstance(
            value,
            dict,
        ):
            return {
                str(key): ManualProfileAnnotator._make_json_compatible(
                    item
                )
                for key, item in value.items()
            }

        if isinstance(
            value,
            (list, tuple),
        ):
            return [
                ManualProfileAnnotator._make_json_compatible(
                    item
                )
                for item in value
            ]

        if isinstance(
            value,
            np.ndarray,
        ):
            return value.tolist()

        if isinstance(
            value,
            np.generic,
        ):
            return value.item()

        if isinstance(
            value,
            Path,
        ):
            return str(value)

        return value


def create_manual_annotator(
    image: np.ndarray,
    profile: Any,
    *,
    label_colors: Mapping[str, str] | None = None,
    source_metadata: Mapping[str, Any] | None = None,
    save: bool = True,
    output_path: str | Path | None = None,
    overwrite: bool = False,
    figure_size: tuple[float, float] = (14, 8),
    gray_value_limits: tuple[float, float] | None = (0.0, 300.0),
) -> ManualProfileAnnotator:
    """
    Create an interactive OCT profile annotation interface.

    Parameters
    ----------
    image:
        Processed two-dimensional OCT image.
    profile:
        Result returned by ``extract_intensity_profile`` with
        ``data=True``.
    label_colors:
        Optional mapping from class names to Matplotlib colors.
    source_metadata:
        Optional source and preprocessing metadata included in saved
        annotation files.
    save:
        Whether annotation saving is enabled. If ``False``, the save
        button is not displayed, but annotations remain available in
        memory through the returned object.
    output_path:
        Optional default JSON output path.
    overwrite:
        Whether an existing output file may be replaced.
    figure_size:
        Matplotlib figure size.
    gray_value_limits:
        Optional profile y-axis limits. Use ``None`` for automatic limits.

    Returns
    -------
    ManualProfileAnnotator
        Interactive annotator object.
    """

    return ManualProfileAnnotator(
        image=image,
        profile=profile,
        label_colors=label_colors,
        source_metadata=source_metadata,
        save_enabled=save,
        output_path=output_path,
        overwrite=overwrite,
        figure_size=figure_size,
        gray_value_limits=gray_value_limits,
    )