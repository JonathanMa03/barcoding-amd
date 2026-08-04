# visualization.py
#
# Responsibilities:
# - Display preprocessing stages.
# - Plot raw and standardized feature signals.
# - Overlay numerical feature signals on the processed scan.
# - Allow interactive feature visibility controls.
# - Display the combined barcoding score beneath the processed scan.
# - Overlay detected barcode intervals on the processed scan.
# - Display the detector threshold and raw or cleaned candidate regions.
# - Compare structural and hypertransmission group scores.
# - Compare raw and cleaned detection masks.
#
# Inputs:
# - Preprocessing results.
# - Feature-extraction results.
# - Detection results.
# - Measurement results.
#
# Outputs:
# - Matplotlib figures.
# - Interactive visualization objects.
#
# This module should not modify preprocessing results, feature values,
# detector scores, masks, intervals, or measurements.


from __future__ import annotations

from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import CheckButtons


FEATURE_NAMES = (
    "verticality",
    "persistence",
    "continuity",
    "amplitude",
    "heterogeneity",
)


DEFAULT_FEATURE_CMAPS: dict[str, str] = {
    "verticality": "Reds",
    "persistence": "Blues",
    "continuity": "Purples",
    "amplitude": "Oranges",
    "heterogeneity": "Greens",
}


class FeatureOverlayViewer:
    """
    Interactive overlay of numerical feature signals on a processed scan.

    Each feature is a one-dimensional signal indexed by horizontal image
    position. For visualization, the signal is interpolated to every image
    column and repeated across the complete image depth.

    Check buttons independently show or hide:

    - Verticality
    - Persistence
    - Periodicity
    - Amplitude
    - Heterogeneity

    Moving the mouse across the scan reports the numerical value of every
    currently visible feature at the selected horizontal position.

    Parameters
    ----------
    image:
        Two-dimensional preprocessed or denoised scan.
    feature_result:
        FeatureSignals object returned by ``extract_feature_signals``.
    use_standardized:
        Whether standardized or raw feature values are displayed.
    initially_visible:
        Features visible when the interface first opens.
    feature_colormaps:
        Optional mapping from feature names to Matplotlib colormaps.
    maximum_alpha:
        Maximum opacity of a feature overlay. Actual opacity varies by
        horizontal position according to the normalized feature value.
    display_percentiles:
        Percentiles used to map feature values into the display range
        from zero to one. This affects visualization only and does not
        modify the numerical feature signals.
    figure_size:
        Matplotlib figure size.
    """

    def __init__(
        self,
        image: np.ndarray,
        feature_result: Any,
        *,
        use_standardized: bool = True,
        initially_visible: tuple[str, ...] = (
            "verticality",
        ),
        feature_colormaps: Mapping[str, str] | None = None,
        maximum_alpha: float = 0.50,
        display_percentiles: tuple[float, float] = (
            2.0,
            98.0,
        ),
        figure_size: tuple[float, float] = (
            15.0,
            7.0,
        ),
    ) -> None:
        self.image = self._validate_image(
            image
        )

        self.feature_result = feature_result
        self.use_standardized = bool(
            use_standardized
        )

        self.feature_colormaps = dict(
            DEFAULT_FEATURE_CMAPS
            if feature_colormaps is None
            else feature_colormaps
        )

        self.maximum_alpha = float(
            maximum_alpha
        )

        if not (
            0.0
            <= self.maximum_alpha
            <= 1.0
        ):
            raise ValueError(
                "maximum_alpha must lie between 0 and 1."
            )

        lower_percentile, upper_percentile = (
            display_percentiles
        )

        if not (
            0.0
            <= lower_percentile
            < upper_percentile
            <= 100.0
        ):
            raise ValueError(
                "display_percentiles must satisfy "
                "0 <= lower < upper <= 100."
            )

        self.display_percentiles = (
            float(lower_percentile),
            float(upper_percentile),
        )

        self.figure_size = figure_size

        self._validate_feature_result()

        self.x_positions = np.asarray(
            self.feature_result.x_positions,
            dtype=np.float32,
        )

        self.feature_signals = (
            self.feature_result.standardized_features
            if self.use_standardized
            else self.feature_result.raw_features
        )

        self.full_width_signals = {
            feature_name: self._interpolate_to_image_width(
                self.feature_signals[
                    feature_name
                ]
            )
            for feature_name in FEATURE_NAMES
        }

        self.display_signals = {
            feature_name: self._normalize_for_display(
                self.full_width_signals[
                    feature_name
                ]
            )
            for feature_name in FEATURE_NAMES
        }

        invalid_initial_features = (
            set(initially_visible)
            - set(FEATURE_NAMES)
        )

        if invalid_initial_features:
            raise ValueError(
                "Unknown initially visible features: "
                f"{sorted(invalid_initial_features)}"
            )

        self.feature_visibility = {
            feature_name: (
                feature_name
                in initially_visible
            )
            for feature_name in FEATURE_NAMES
        }

        self.figure: Any | None = None
        self.scan_axis: Any | None = None
        self.control_axis: Any | None = None
        self.check_buttons: Any | None = None
        self.status_text: Any | None = None
        self.cursor_line: Any | None = None

        self.overlay_artists: dict[
            str,
            Any,
        ] = {}

        self._motion_connection: int | None = None
        self._leave_connection: int | None = None

        self._create_interface()

    @staticmethod
    def _validate_image(
        image: np.ndarray,
    ) -> np.ndarray:
        """Validate and convert a processed scan to float32."""

        array = np.asarray(
            image,
            dtype=np.float32,
        )

        if array.ndim != 2:
            raise ValueError(
                "Feature overlay requires a two-dimensional image; "
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

            replacement = float(
                np.median(
                    finite_values
                )
            )

            array = np.where(
                np.isfinite(array),
                array,
                replacement,
            ).astype(np.float32)

        return array

    def _validate_feature_result(
        self,
    ) -> None:
        """Validate the supplied feature result."""

        required_attributes = {
            "x_positions",
            "raw_features",
            "standardized_features",
        }

        missing = [
            attribute
            for attribute in required_attributes
            if not hasattr(
                self.feature_result,
                attribute,
            )
        ]

        if missing:
            raise TypeError(
                "feature_result is missing required attributes: "
                f"{missing}"
            )

        x_positions = np.asarray(
            self.feature_result.x_positions,
        )

        if x_positions.ndim != 1:
            raise ValueError(
                "feature_result.x_positions must be one-dimensional."
            )

        if x_positions.size == 0:
            raise ValueError(
                "feature_result contains no horizontal positions."
            )

        feature_dictionary = (
            self.feature_result.standardized_features
            if self.use_standardized
            else self.feature_result.raw_features
        )

        for feature_name in FEATURE_NAMES:
            if feature_name not in feature_dictionary:
                raise KeyError(
                    f"Feature result is missing '{feature_name}'."
                )

            values = np.asarray(
                feature_dictionary[
                    feature_name
                ]
            )

            if values.ndim != 1:
                raise ValueError(
                    f"Feature '{feature_name}' must be one-dimensional."
                )

            if values.size != x_positions.size:
                raise ValueError(
                    f"Feature '{feature_name}' has {values.size} values, "
                    f"but x_positions contains {x_positions.size}."
                )

            if not np.isfinite(values).all():
                raise ValueError(
                    f"Feature '{feature_name}' contains non-finite values."
                )

    def _interpolate_to_image_width(
        self,
        feature_values: np.ndarray,
    ) -> np.ndarray:
        """
        Interpolate one feature signal to every image column.

        This is necessary when feature extraction uses a stride larger
        than one pixel.
        """
        values = np.asarray(
            feature_values,
            dtype=np.float32,
        )

        image_width = self.image.shape[1]

        full_x = np.arange(
            image_width,
            dtype=np.float32,
        )

        interpolated = np.interp(
            full_x,
            self.x_positions,
            values,
            left=float(
                values[0]
            ),
            right=float(
                values[-1]
            ),
        )

        return interpolated.astype(
            np.float32
        )

    def _normalize_for_display(
        self,
        values: np.ndarray,
    ) -> np.ndarray:
        """
        Robustly map feature values into [0, 1] for overlay display.

        This transformation changes only the opacity and color used in the
        figure. It does not change the stored numerical feature values.
        """
        lower_percentile, upper_percentile = (
            self.display_percentiles
        )

        lower_value = float(
            np.percentile(
                values,
                lower_percentile,
            )
        )

        upper_value = float(
            np.percentile(
                values,
                upper_percentile,
            )
        )

        if upper_value <= lower_value:
            return np.zeros_like(
                values,
                dtype=np.float32,
            )

        normalized = (
            values - lower_value
        ) / (
            upper_value - lower_value
        )

        return np.clip(
            normalized,
            0.0,
            1.0,
        ).astype(np.float32)

    def _create_interface(
        self,
    ) -> None:
        """Create the interactive feature-overlay interface."""

        self.figure = plt.figure(
            figsize=self.figure_size,
        )

        grid = self.figure.add_gridspec(
            nrows=1,
            ncols=2,
            width_ratios=[
                5.5,
                1.3,
            ],
            wspace=0.12,
        )

        self.scan_axis = self.figure.add_subplot(
            grid[0, 0]
        )

        self.control_axis = self.figure.add_subplot(
            grid[0, 1]
        )

        self._draw_base_image()
        self._create_overlays()
        self._create_controls()

        self._motion_connection = (
            self.figure.canvas.mpl_connect(
                "motion_notify_event",
                self._handle_mouse_motion,
            )
        )

        self._leave_connection = (
            self.figure.canvas.mpl_connect(
                "axes_leave_event",
                self._handle_axes_leave,
            )
        )

        self.figure.canvas.draw_idle()

    def _draw_base_image(
        self,
    ) -> None:
        """Draw the processed scan."""

        image_height, image_width = (
            self.image.shape
        )

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
            zorder=0,
        )

        signal_type = (
            "standardized"
            if self.use_standardized
            else "raw"
        )

        self.scan_axis.set_title(
            "Numerical feature overlay on processed scan\n"
            f"Displaying {signal_type} feature values"
        )

        self.scan_axis.set_xlabel(
            "Horizontal position"
        )

        self.scan_axis.set_ylabel(
            "Depth below BM"
        )

        self.scan_axis.set_xlim(
            -0.5,
            image_width - 0.5,
        )

        self.scan_axis.set_ylim(
            image_height - 0.5,
            -0.5,
        )

        self.cursor_line = (
            self.scan_axis.axvline(
                0,
                linestyle="--",
                linewidth=1.0,
                visible=False,
                zorder=20,
            )
        )

    def _create_overlays(
        self,
    ) -> None:
        """Create one initially hidden overlay artist per feature."""

        image_height, image_width = (
            self.image.shape
        )

        for feature_name in FEATURE_NAMES:
            display_signal = (
                self.display_signals[
                    feature_name
                ]
            )

            overlay_values = np.tile(
                display_signal[
                    np.newaxis,
                    :,
                ],
                (
                    image_height,
                    1,
                ),
            )

            alpha_values = (
                self.maximum_alpha
                * overlay_values
            )

            artist = self.scan_axis.imshow(
                overlay_values,
                cmap=self.feature_colormaps[
                    feature_name
                ],
                aspect="auto",
                interpolation="nearest",
                extent=(
                    -0.5,
                    image_width - 0.5,
                    image_height - 0.5,
                    -0.5,
                ),
                vmin=0.0,
                vmax=1.0,
                alpha=alpha_values,
                visible=self.feature_visibility[
                    feature_name
                ],
                zorder=5,
            )

            self.overlay_artists[
                feature_name
            ] = artist

    def _create_controls(
        self,
    ) -> None:
        """Create feature visibility controls and numerical status text."""

        self.control_axis.set_title(
            "Feature overlays",
            pad=15,
        )

        self.control_axis.set_xticks([])
        self.control_axis.set_yticks([])

        check_axis = self.control_axis.inset_axes(
            [
                0.05,
                0.56,
                0.90,
                0.35,
            ]
        )

        check_axis.set_title(
            "Show / hide",
            fontsize=10,
        )

        labels = tuple(
            feature_name.title()
            for feature_name in FEATURE_NAMES
        )

        active = tuple(
            self.feature_visibility[
                feature_name
            ]
            for feature_name in FEATURE_NAMES
        )

        self.check_buttons = CheckButtons(
            check_axis,
            labels,
            active,
        )

        for label_artist, feature_name in zip(
            self.check_buttons.labels,
            FEATURE_NAMES,
        ):
            colormap = plt.get_cmap(
                self.feature_colormaps[
                    feature_name
                ]
            )

            label_artist.set_color(
                colormap(
                    0.75
                )
            )

        self.check_buttons.on_clicked(
            self._toggle_feature
        )

        signal_description = (
            "Standardized values"
            if self.use_standardized
            else "Raw values"
        )

        self.status_text = self.control_axis.text(
            0.5,
            0.35,
            (
                f"{signal_description}\n\n"
                "Move the cursor across the scan\n"
                "to inspect feature values."
            ),
            ha="center",
            va="center",
            transform=self.control_axis.transAxes,
            fontsize=9,
        )

        self.control_axis.text(
            0.5,
            0.09,
            (
                "Overlay brightness represents\n"
                "relative feature magnitude.\n\n"
                "The display scaling does not\n"
                "change the numerical values."
            ),
            ha="center",
            va="center",
            transform=self.control_axis.transAxes,
            fontsize=8,
        )

    def _toggle_feature(
        self,
        label: str,
    ) -> None:
        """Toggle one feature overlay on or off."""

        feature_name = (
            label.strip()
            .lower()
            .replace(
                " ",
                "_",
            )
        )

        if feature_name not in self.overlay_artists:
            return

        new_visibility = (
            not self.feature_visibility[
                feature_name
            ]
        )

        self.feature_visibility[
            feature_name
        ] = new_visibility

        self.overlay_artists[
            feature_name
        ].set_visible(
            new_visibility
        )

        self._update_default_status()

        self.figure.canvas.draw_idle()

    def _handle_mouse_motion(
        self,
        event: Any,
    ) -> None:
        """Report active feature values at the cursor position."""

        if event.inaxes is not self.scan_axis:
            return

        if event.xdata is None:
            return

        image_width = self.image.shape[1]

        x_position = int(
            np.clip(
                round(
                    event.xdata
                ),
                0,
                image_width - 1,
            )
        )

        self.cursor_line.set_xdata(
            [
                x_position,
                x_position,
            ]
        )

        self.cursor_line.set_visible(
            True
        )

        visible_features = [
            feature_name
            for feature_name in FEATURE_NAMES
            if self.feature_visibility[
                feature_name
            ]
        ]

        lines = [
            f"x = {x_position}",
        ]

        if not visible_features:
            lines.append(
                "No feature overlays selected."
            )
        else:
            for feature_name in visible_features:
                value = float(
                    self.full_width_signals[
                        feature_name
                    ][
                        x_position
                    ]
                )

                lines.append(
                    f"{feature_name.title()}: "
                    f"{value:.4f}"
                )

        self.status_text.set_text(
            "\n".join(
                lines
            )
        )

        self.figure.canvas.draw_idle()

    def _handle_axes_leave(
        self,
        event: Any,
    ) -> None:
        """Hide the cursor marker after leaving the scan axis."""

        if event.inaxes is not self.scan_axis:
            return

        self.cursor_line.set_visible(
            False
        )

        self._update_default_status()

        self.figure.canvas.draw_idle()

    def _update_default_status(
        self,
    ) -> None:
        """Display the currently active feature list."""

        active_features = [
            feature_name.title()
            for feature_name in FEATURE_NAMES
            if self.feature_visibility[
                feature_name
            ]
        ]

        if active_features:
            active_text = "\n".join(
                active_features
            )

            message = (
                "Active overlays:\n"
                f"{active_text}\n\n"
                "Move the cursor across the scan\n"
                "to inspect numerical values."
            )
        else:
            message = (
                "No feature overlays selected.\n\n"
                "Use the controls above to\n"
                "enable a feature."
            )

        self.status_text.set_text(
            message
        )

    def set_feature_visibility(
        self,
        feature_name: str,
        visible: bool,
    ) -> None:
        """
        Programmatically show or hide one feature.

        Parameters
        ----------
        feature_name:
            One of the supported mathematical feature names.
        visible:
            Desired visibility state.
        """
        normalized_name = (
            feature_name.strip()
            .lower()
            .replace(
                " ",
                "_",
            )
        )

        if normalized_name not in FEATURE_NAMES:
            raise ValueError(
                f"feature_name must be one of {FEATURE_NAMES}."
            )

        visible = bool(
            visible
        )

        current_state = (
            self.feature_visibility[
                normalized_name
            ]
        )

        if visible != current_state:
            feature_index = FEATURE_NAMES.index(
                normalized_name
            )

            self.check_buttons.set_active(
                feature_index
            )

    @property
    def active_features(
        self,
    ) -> list[str]:
        """Return the currently visible feature names."""

        return [
            feature_name
            for feature_name in FEATURE_NAMES
            if self.feature_visibility[
                feature_name
            ]
        ]

    def show(
        self,
    ) -> None:
        """Display the interactive feature-overlay interface."""

        plt.show()


def create_feature_overlay_viewer(
        
    image: np.ndarray,
    feature_result: Any,
    **kwargs: Any,
) -> FeatureOverlayViewer:
    """
    Create an interactive numerical feature overlay.

    Parameters
    ----------
    image:
        Two-dimensional preprocessed or denoised scan.
    feature_result:
        FeatureSignals returned by ``extract_feature_signals``.
    **kwargs:
        Additional FeatureOverlayViewer configuration.

    Returns
    -------
    FeatureOverlayViewer
        Interactive feature-overlay object.
    """

    return FeatureOverlayViewer(
        image=image,
        feature_result=feature_result,
        **kwargs,
    )

class BarcodeScoreViewer:
    """
    Interactive visualization of the barcode detector output.

    The processed scan and continuous barcode score are displayed with a
    shared horizontal axis. Cleaned barcode intervals can be highlighted
    on the scan, while raw and cleaned candidate regions can be shown on
    the score panel.

    Available visibility controls include:

    - detected intervals on the processed scan;
    - raw threshold-positive regions;
    - cleaned threshold-positive regions;
    - detector threshold;
    - structural group score;
    - hypertransmission group score.

    Moving the cursor across either panel reports the horizontal position,
    combined barcode score, threshold status, cleaned detection status,
    and grouped component scores when available.

    Parameters
    ----------
    image:
        Two-dimensional preprocessed or denoised scan.
    detection_result:
        DetectionResult returned by ``detect_barcoding``.
    initially_visible:
        Visualization elements visible when the viewer is created.
    interval_alpha:
        Transparency of detected interval overlays on the scan.
    candidate_alpha:
        Transparency of raw and cleaned candidate-region shading on the
        score panel.
    score_limits:
        Optional fixed y-axis limits for the score plot. If omitted, the
        limits are determined automatically.
    figure_size:
        Matplotlib figure size.
    """

    CONTROL_NAMES = (
        "detected_intervals",
        "raw_candidates",
        "cleaned_candidates",
        "threshold",
        "structural_score",
        "hypertransmission_score",
    )

    CONTROL_LABELS = {
        "detected_intervals": "Detected intervals",
        "raw_candidates": "Raw candidates",
        "cleaned_candidates": "Cleaned candidates",
        "threshold": "Threshold",
        "structural_score": "Structural score",
        "hypertransmission_score": "Hypertransmission score",
    }

    def __init__(
        self,
        image: np.ndarray,
        detection_result: Any,
        *,
        initially_visible: tuple[str, ...] = (
            "detected_intervals",
            "cleaned_candidates",
            "threshold",
        ),
        interval_alpha: float = 0.25,
        candidate_alpha: float = 0.15,
        score_limits: tuple[float, float] | None = None,
        figure_size: tuple[float, float] = (
            15.0,
            9.0,
        ),
    ) -> None:
        self.image = self._validate_image(
            image
        )

        self.detection_result = (
            detection_result
        )

        self.interval_alpha = float(
            interval_alpha
        )

        self.candidate_alpha = float(
            candidate_alpha
        )

        if not (
            0.0
            <= self.interval_alpha
            <= 1.0
        ):
            raise ValueError(
                "interval_alpha must lie between 0 and 1."
            )

        if not (
            0.0
            <= self.candidate_alpha
            <= 1.0
        ):
            raise ValueError(
                "candidate_alpha must lie between 0 and 1."
            )

        if score_limits is not None:
            lower_limit, upper_limit = (
                score_limits
            )

            if lower_limit >= upper_limit:
                raise ValueError(
                    "score_limits must satisfy lower < upper."
                )

            self.score_limits = (
                float(lower_limit),
                float(upper_limit),
            )

        else:
            self.score_limits = None

        self.figure_size = figure_size

        self._validate_detection_result()

        self.x_positions = np.asarray(
            self.detection_result.x_positions,
            dtype=np.float32,
        )

        self.score = np.asarray(
            self.detection_result.score,
            dtype=np.float32,
        )

        self.raw_mask = np.asarray(
            self.detection_result.raw_mask,
            dtype=bool,
        )

        self.cleaned_mask = np.asarray(
            self.detection_result.cleaned_mask,
            dtype=bool,
        )

        self.threshold = float(
            self.detection_result.threshold
        )

        self.group_scores = {
            name: np.asarray(
                values,
                dtype=np.float32,
            )
            for name, values in (
                self.detection_result.group_scores.items()
            )
        }

        invalid_controls = (
            set(initially_visible)
            - set(self.CONTROL_NAMES)
        )

        if invalid_controls:
            raise ValueError(
                "Unknown initially visible controls: "
                f"{sorted(invalid_controls)}"
            )

        self.control_visibility = {
            name: (
                name in initially_visible
            )
            for name in self.CONTROL_NAMES
        }

        # Group-score controls are unavailable for individual scoring.
        if "structural" not in self.group_scores:
            self.control_visibility[
                "structural_score"
            ] = False

        if "hypertransmission" not in self.group_scores:
            self.control_visibility[
                "hypertransmission_score"
            ] = False

        self.figure: Any | None = None
        self.scan_axis: Any | None = None
        self.score_axis: Any | None = None
        self.control_axis: Any | None = None

        self.check_buttons: Any | None = None
        self.status_text: Any | None = None

        self.combined_score_artist: Any | None = None
        self.threshold_artist: Any | None = None
        self.structural_score_artist: Any | None = None
        self.hypertransmission_score_artist: Any | None = None

        self.scan_cursor_line: Any | None = None
        self.score_cursor_line: Any | None = None

        self.interval_artists: list[Any] = []
        self.raw_candidate_artists: list[Any] = []
        self.cleaned_candidate_artists: list[Any] = []

        self._motion_connection: int | None = None
        self._leave_connection: int | None = None

        self._create_interface()

    @staticmethod
    def _validate_image(
        image: np.ndarray,
    ) -> np.ndarray:
        """Validate and convert the processed image to float32."""

        array = np.asarray(
            image,
            dtype=np.float32,
        )

        if array.ndim != 2:
            raise ValueError(
                "Barcode score visualization requires a "
                "two-dimensional image; "
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

            replacement = float(
                np.median(
                    finite_values
                )
            )

            array = np.where(
                np.isfinite(array),
                array,
                replacement,
            ).astype(np.float32)

        return array

    def _validate_detection_result(
        self,
    ) -> None:
        """Validate the supplied detector output."""

        required_attributes = {
            "x_positions",
            "score",
            "raw_mask",
            "cleaned_mask",
            "intervals",
            "scoring_mode",
            "threshold",
            "group_scores",
        }

        missing = [
            attribute
            for attribute in required_attributes
            if not hasattr(
                self.detection_result,
                attribute,
            )
        ]

        if missing:
            raise TypeError(
                "detection_result is missing required attributes: "
                f"{missing}"
            )

        x_positions = np.asarray(
            self.detection_result.x_positions,
        )

        score = np.asarray(
            self.detection_result.score,
        )

        raw_mask = np.asarray(
            self.detection_result.raw_mask,
        )

        cleaned_mask = np.asarray(
            self.detection_result.cleaned_mask,
        )

        if x_positions.ndim != 1:
            raise ValueError(
                "detection_result.x_positions must be one-dimensional."
            )

        if x_positions.size == 0:
            raise ValueError(
                "detection_result contains no horizontal positions."
            )

        expected_length = (
            x_positions.size
        )

        for name, values in (
            ("score", score),
            ("raw_mask", raw_mask),
            ("cleaned_mask", cleaned_mask),
        ):
            if values.ndim != 1:
                raise ValueError(
                    f"detection_result.{name} must be one-dimensional."
                )

            if values.size != expected_length:
                raise ValueError(
                    f"detection_result.{name} contains "
                    f"{values.size} values, but expected "
                    f"{expected_length}."
                )

        if not np.isfinite(
            score
        ).all():
            raise ValueError(
                "The detector score contains non-finite values."
            )

        if not np.isfinite(
            float(
                self.detection_result.threshold
            )
        ):
            raise ValueError(
                "The detector threshold must be finite."
            )

        group_scores = (
            self.detection_result.group_scores
        )

        if not isinstance(
            group_scores,
            Mapping,
        ):
            raise TypeError(
                "detection_result.group_scores must be a mapping."
            )

        for name, values in group_scores.items():
            group_signal = np.asarray(
                values
            )

            if group_signal.ndim != 1:
                raise ValueError(
                    f"Group score '{name}' must be one-dimensional."
                )

            if group_signal.size != expected_length:
                raise ValueError(
                    f"Group score '{name}' contains "
                    f"{group_signal.size} values, but expected "
                    f"{expected_length}."
                )

            if not np.isfinite(
                group_signal
            ).all():
                raise ValueError(
                    f"Group score '{name}' contains non-finite values."
                )

    def _create_interface(
        self,
    ) -> None:
        """Create the shared-axis detector visualization."""

        self.figure = plt.figure(
            figsize=self.figure_size,
        )

        grid = self.figure.add_gridspec(
            nrows=2,
            ncols=2,
            width_ratios=[
                5.5,
                1.3,
            ],
            height_ratios=[
                1.4,
                1.0,
            ],
            hspace=0.12,
            wspace=0.12,
        )

        self.scan_axis = self.figure.add_subplot(
            grid[0, 0]
        )

        self.score_axis = self.figure.add_subplot(
            grid[1, 0],
            sharex=self.scan_axis,
        )

        self.control_axis = self.figure.add_subplot(
            grid[:, 1]
        )

        self._draw_scan()
        self._draw_score()
        self._create_detection_artists()
        self._create_controls()

        self._motion_connection = (
            self.figure.canvas.mpl_connect(
                "motion_notify_event",
                self._handle_mouse_motion,
            )
        )

        self._leave_connection = (
            self.figure.canvas.mpl_connect(
                "axes_leave_event",
                self._handle_axes_leave,
            )
        )

        self.figure.canvas.draw_idle()

    def _draw_scan(
        self,
    ) -> None:
        """Draw the processed scan."""

        image_height, image_width = (
            self.image.shape
        )

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
            zorder=0,
        )

        scoring_mode = str(
            self.detection_result.scoring_mode
        ).title()

        self.scan_axis.set_title(
            "Detected barcoding intervals on processed scan\n"
            f"{scoring_mode} scoring mode"
        )

        self.scan_axis.set_ylabel(
            "Depth below BM"
        )

        self.scan_axis.set_xlim(
            -0.5,
            image_width - 0.5,
        )

        self.scan_axis.set_ylim(
            image_height - 0.5,
            -0.5,
        )

        plt.setp(
            self.scan_axis.get_xticklabels(),
            visible=False,
        )

        self.scan_cursor_line = (
            self.scan_axis.axvline(
                0,
                linestyle="--",
                linewidth=1.0,
                visible=False,
                zorder=30,
            )
        )

    def _draw_score(
        self,
    ) -> None:
        """Draw the continuous score and optional group signals."""

        (
            self.combined_score_artist,
        ) = self.score_axis.plot(
            self.x_positions,
            self.score,
            linewidth=1.5,
            label="Combined barcoding score",
            zorder=10,
        )

        self.threshold_artist = (
            self.score_axis.axhline(
                self.threshold,
                linestyle="--",
                linewidth=1.25,
                label=(
                    f"Threshold = "
                    f"{self.threshold:.3f}"
                ),
                visible=self.control_visibility[
                    "threshold"
                ],
                zorder=9,
            )
        )

        if "structural" in self.group_scores:
            (
                self.structural_score_artist,
            ) = self.score_axis.plot(
                self.x_positions,
                self.group_scores[
                    "structural"
                ],
                linewidth=1.0,
                alpha=0.80,
                label="Structural score",
                visible=self.control_visibility[
                    "structural_score"
                ],
                zorder=7,
            )

        if (
            "hypertransmission"
            in self.group_scores
        ):
            (
                self.hypertransmission_score_artist,
            ) = self.score_axis.plot(
                self.x_positions,
                self.group_scores[
                    "hypertransmission"
                ],
                linewidth=1.0,
                alpha=0.80,
                label="Hypertransmission score",
                visible=self.control_visibility[
                    "hypertransmission_score"
                ],
                zorder=7,
            )

        self.score_axis.set_title(
            "Continuous barcoding score"
        )

        self.score_axis.set_xlabel(
            "Horizontal position"
        )

        self.score_axis.set_ylabel(
            "Standardized weighted score"
        )

        self.score_axis.set_xlim(
            -0.5,
            self.image.shape[1] - 0.5,
        )

        if self.score_limits is not None:
            self.score_axis.set_ylim(
                *self.score_limits
            )

        self.score_axis.grid(
            alpha=0.25,
        )

        self.score_cursor_line = (
            self.score_axis.axvline(
                0,
                linestyle="--",
                linewidth=1.0,
                visible=False,
                zorder=30,
            )
        )

        self._refresh_score_legend()

    def _create_detection_artists(
        self,
    ) -> None:
        """Create interval and candidate-region shading."""

        self.interval_artists.clear()
        self.raw_candidate_artists.clear()
        self.cleaned_candidate_artists.clear()

        for interval in (
            self.detection_result.intervals
        ):
            artist = self.scan_axis.axvspan(
                float(
                    interval.x_start
                ),
                float(
                    interval.x_end
                ),
                alpha=self.interval_alpha,
                visible=self.control_visibility[
                    "detected_intervals"
                ],
                zorder=5,
            )

            self.interval_artists.append(
                artist
            )

        raw_runs = self._find_positive_runs(
            self.raw_mask
        )

        for start_index, end_index in raw_runs:
            x_start, x_end = (
                self._resolve_run_bounds(
                    start_index,
                    end_index,
                )
            )

            artist = self.score_axis.axvspan(
                x_start,
                x_end,
                alpha=self.candidate_alpha,
                visible=self.control_visibility[
                    "raw_candidates"
                ],
                zorder=1,
            )

            self.raw_candidate_artists.append(
                artist
            )

        cleaned_runs = self._find_positive_runs(
            self.cleaned_mask
        )

        for start_index, end_index in cleaned_runs:
            x_start, x_end = (
                self._resolve_run_bounds(
                    start_index,
                    end_index,
                )
            )

            artist = self.score_axis.axvspan(
                x_start,
                x_end,
                alpha=self.candidate_alpha,
                visible=self.control_visibility[
                    "cleaned_candidates"
                ],
                zorder=2,
            )

            self.cleaned_candidate_artists.append(
                artist
            )

    @staticmethod
    def _find_positive_runs(
        mask: np.ndarray,
    ) -> list[tuple[int, int]]:
        """Return inclusive runs of true values."""

        boolean_mask = np.asarray(
            mask,
            dtype=bool,
        )

        runs: list[
            tuple[int, int]
        ] = []

        start_index: int | None = None

        for index, value in enumerate(
            boolean_mask
        ):
            if value:
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
                    boolean_mask.size - 1,
                )
            )

        return runs

    def _resolve_run_bounds(
        self,
        start_index: int,
        end_index: int,
    ) -> tuple[float, float]:
        """Convert inclusive signal indices into display coordinates."""

        if self.x_positions.size > 1:
            spacing = float(
                np.median(
                    np.diff(
                        self.x_positions
                    )
                )
            )
        else:
            spacing = 1.0

        half_spacing = spacing / 2.0

        x_start = float(
            self.x_positions[
                start_index
            ]
            - half_spacing
        )

        x_end = float(
            self.x_positions[
                end_index
            ]
            + half_spacing
        )

        return (
            x_start,
            x_end,
        )

    def _create_controls(
        self,
    ) -> None:
        """Create visibility controls and numerical status display."""

        self.control_axis.set_title(
            "Detector display",
            pad=15,
        )

        self.control_axis.set_xticks([])
        self.control_axis.set_yticks([])

        available_controls = [
            "detected_intervals",
            "raw_candidates",
            "cleaned_candidates",
            "threshold",
        ]

        if "structural" in self.group_scores:
            available_controls.append(
                "structural_score"
            )

        if (
            "hypertransmission"
            in self.group_scores
        ):
            available_controls.append(
                "hypertransmission_score"
            )

        self.available_controls = tuple(
            available_controls
        )

        check_axis = self.control_axis.inset_axes(
            [
                0.05,
                0.55,
                0.90,
                0.36,
            ]
        )

        check_axis.set_title(
            "Show / hide",
            fontsize=10,
        )

        labels = tuple(
            self.CONTROL_LABELS[
                name
            ]
            for name in self.available_controls
        )

        active = tuple(
            self.control_visibility[
                name
            ]
            for name in self.available_controls
        )

        self.check_buttons = CheckButtons(
            check_axis,
            labels,
            active,
        )

        self.check_buttons.on_clicked(
            self._toggle_control
        )

        self.status_text = (
            self.control_axis.text(
                0.5,
                0.32,
                self._default_status_message(),
                ha="center",
                va="center",
                transform=(
                    self.control_axis.transAxes
                ),
                fontsize=9,
                wrap=True,
            )
        )

        self.control_axis.text(
            0.5,
            0.085,
            (
                "Move the cursor across either\n"
                "panel to inspect the score.\n\n"
                "Highlighted scan regions come\n"
                "from the cleaned detection mask."
            ),
            ha="center",
            va="center",
            transform=(
                self.control_axis.transAxes
            ),
            fontsize=8,
        )

    def _toggle_control(
        self,
        label: str,
    ) -> None:
        """Toggle one detector visualization element."""

        control_name = next(
            (
                name
                for name in self.available_controls
                if self.CONTROL_LABELS[
                    name
                ] == label
            ),
            None,
        )

        if control_name is None:
            return

        new_visibility = (
            not self.control_visibility[
                control_name
            ]
        )

        self.control_visibility[
            control_name
        ] = new_visibility

        if control_name == "detected_intervals":
            self._set_artists_visible(
                self.interval_artists,
                new_visibility,
            )

        elif control_name == "raw_candidates":
            self._set_artists_visible(
                self.raw_candidate_artists,
                new_visibility,
            )

        elif control_name == "cleaned_candidates":
            self._set_artists_visible(
                self.cleaned_candidate_artists,
                new_visibility,
            )

        elif control_name == "threshold":
            self.threshold_artist.set_visible(
                new_visibility
            )

        elif control_name == "structural_score":
            if (
                self.structural_score_artist
                is not None
            ):
                self.structural_score_artist.set_visible(
                    new_visibility
                )

        elif control_name == (
            "hypertransmission_score"
        ):
            if (
                self.hypertransmission_score_artist
                is not None
            ):
                self.hypertransmission_score_artist.set_visible(
                    new_visibility
                )

        self._refresh_score_legend()

        self.status_text.set_text(
            self._default_status_message()
        )

        self.figure.canvas.draw_idle()

    @staticmethod
    def _set_artists_visible(
        artists: list[Any],
        visible: bool,
    ) -> None:
        """Set visibility for a collection of Matplotlib artists."""

        for artist in artists:
            artist.set_visible(
                visible
            )

    def _refresh_score_legend(
        self,
    ) -> None:
        """Refresh the score-panel legend using visible line artists."""

        handles: list[Any] = []
        labels: list[str] = []

        candidate_artists = [
            self.combined_score_artist,
            self.threshold_artist,
            self.structural_score_artist,
            self.hypertransmission_score_artist,
        ]

        for artist in candidate_artists:
            if (
                artist is not None
                and artist.get_visible()
            ):
                handles.append(
                    artist
                )

                labels.append(
                    artist.get_label()
                )

        existing_legend = (
            self.score_axis.get_legend()
        )

        if existing_legend is not None:
            existing_legend.remove()

        if handles:
            self.score_axis.legend(
                handles,
                labels,
                loc="upper right",
            )

    def _handle_mouse_motion(
        self,
        event: Any,
    ) -> None:
        """Display numerical detector values at the cursor position."""

        if event.inaxes not in {
            self.scan_axis,
            self.score_axis,
        }:
            return

        if event.xdata is None:
            return

        image_width = self.image.shape[1]

        x_coordinate = float(
            np.clip(
                event.xdata,
                0,
                image_width - 1,
            )
        )

        nearest_index = int(
            np.argmin(
                np.abs(
                    self.x_positions
                    - x_coordinate
                )
            )
        )

        resolved_x = float(
            self.x_positions[
                nearest_index
            ]
        )

        self.scan_cursor_line.set_xdata(
            [
                resolved_x,
                resolved_x,
            ]
        )

        self.score_cursor_line.set_xdata(
            [
                resolved_x,
                resolved_x,
            ]
        )

        self.scan_cursor_line.set_visible(
            True
        )

        self.score_cursor_line.set_visible(
            True
        )

        score_value = float(
            self.score[
                nearest_index
            ]
        )

        raw_positive = bool(
            self.raw_mask[
                nearest_index
            ]
        )

        cleaned_positive = bool(
            self.cleaned_mask[
                nearest_index
            ]
        )

        lines = [
            f"x = {resolved_x:.1f}",
            f"Combined score: {score_value:.4f}",
            f"Threshold: {self.threshold:.4f}",
            (
                "Raw candidate: "
                f"{'yes' if raw_positive else 'no'}"
            ),
            (
                "Cleaned detection: "
                f"{'yes' if cleaned_positive else 'no'}"
            ),
        ]

        if "structural" in self.group_scores:
            structural_value = float(
                self.group_scores[
                    "structural"
                ][
                    nearest_index
                ]
            )

            lines.append(
                f"Structural: {structural_value:.4f}"
            )

        if (
            "hypertransmission"
            in self.group_scores
        ):
            transmission_value = float(
                self.group_scores[
                    "hypertransmission"
                ][
                    nearest_index
                ]
            )

            lines.append(
                "Hypertransmission: "
                f"{transmission_value:.4f}"
            )

        self.status_text.set_text(
            "\n".join(
                lines
            )
        )

        self.figure.canvas.draw_idle()

    def _handle_axes_leave(
        self,
        event: Any,
    ) -> None:
        """Hide cursor lines after leaving an image or score panel."""

        if event.inaxes not in {
            self.scan_axis,
            self.score_axis,
        }:
            return

        self.scan_cursor_line.set_visible(
            False
        )

        self.score_cursor_line.set_visible(
            False
        )

        self.status_text.set_text(
            self._default_status_message()
        )

        self.figure.canvas.draw_idle()

    def _default_status_message(
        self,
    ) -> str:
        """Create the default detector summary text."""

        return (
            f"Scoring mode: "
            f"{self.detection_result.scoring_mode}\n"
            f"Threshold: {self.threshold:.3f}\n"
            f"Raw positive positions: "
            f"{int(self.raw_mask.sum())}\n"
            f"Cleaned positive positions: "
            f"{int(self.cleaned_mask.sum())}\n"
            f"Detected intervals: "
            f"{len(self.detection_result.intervals)}"
        )

    @property
    def active_controls(
        self,
    ) -> list[str]:
        """Return the currently visible detector elements."""

        return [
            name
            for name in self.available_controls
            if self.control_visibility[
                name
            ]
        ]

    def show(
        self,
    ) -> None:
        """Display the interactive detector visualization."""

        plt.show()

def create_barcode_score_viewer(
    image: np.ndarray,
    detection_result: Any,
    **kwargs: Any,
) -> BarcodeScoreViewer:
    """
    Create an interactive barcode-score visualization.

    Parameters
    ----------
    image:
        Two-dimensional preprocessed or denoised scan.
    detection_result:
        DetectionResult returned by ``detect_barcoding``.
    **kwargs:
        Additional BarcodeScoreViewer configuration.

    Returns
    -------
    BarcodeScoreViewer
        Interactive scan, score, threshold, and interval viewer.
    """

    return BarcodeScoreViewer(
        image=image,
        detection_result=detection_result,
        **kwargs,
    )