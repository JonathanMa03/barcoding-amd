# visualization.py
#
# Responsibilities:
# - Display preprocessing stages.
# - Plot raw and standardized feature signals.
# - Overlay numerical feature signals on the processed scan.
# - Allow interactive feature visibility controls.
# - Overlay detected intervals on the processed scan.
# - Plot the combined score and detection threshold.
# - Compare raw and cleaned detection masks.
#
# Inputs:
# - Preprocessing, feature, detection, and measurement results.
#
# Outputs:
# - Matplotlib figures and interactive visualization objects.
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