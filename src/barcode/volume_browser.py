from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np

from src.barcode.data import preprocess_bscan
from src.barcode.model_threshold import extract_intensity_profile


class VolumeProfileBrowser:
    """
    Interactive browser for preprocessing and profile extraction across
    all B-scans in one OCT volume.

    Navigation
    ----------
    Right arrow:
        Move to the next B-scan.
    Left arrow:
        Move to the previous B-scan.
    Home:
        Move to the first B-scan.
    End:
        Move to the final B-scan.
    c:
        Move to the central B-scan.

    Notes
    -----
    B-scans are processed lazily and cached after their first display.
    """

    def __init__(
        self,
        volume: Any,
        *,
        initial_index: int | None = None,
        bm_layer_name: str = "BM",
        depth_below_bm: int = 150,
        reference_row: int | None = None,
        normalize: bool = True,
        normalization_scope: str = "whole_roi",
        normalization_center_row: int | None = None,
        normalization_margin: int = 0,
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0,
        profile_depth: int = 50,
        profile_margin: int = 4,
        aggregation: str = "mean",
        stepsize: float = 1.0,
        denoise: bool = True,
        denoise_method: str = "gaussian",
        denoise_sigma: float = 1.5,
        gray_value_limits: tuple[float, float] = (0.0, 300.0),
        figure_size: tuple[float, float] = (14, 8),
    ) -> None:
        self.volume = volume
        self.number_of_scans = len(volume)

        if self.number_of_scans == 0:
            raise ValueError(
                "The supplied volume contains no B-scans."
            )

        if initial_index is None:
            initial_index = self.number_of_scans // 2

        if not 0 <= initial_index < self.number_of_scans:
            raise IndexError(
                f"initial_index must lie between 0 and "
                f"{self.number_of_scans - 1}."
            )

        self.current_index = int(initial_index)

        self.preprocessing_config = {
            "bm_layer_name": bm_layer_name,
            "depth_below_bm": depth_below_bm,
            "reference_row": reference_row,
            "normalize": normalize,
            "normalization_scope": normalization_scope,
            "normalization_center_row": normalization_center_row,
            "normalization_margin": normalization_margin,
            "lower_percentile": lower_percentile,
            "upper_percentile": upper_percentile,
        }

        self.profile_config = {
            "profile_depth": int(profile_depth),
            "profile_margin": int(profile_margin),
            "aggregation": aggregation,
            "stepsize": float(stepsize),
            "denoise": bool(denoise),
            "denoise_method": denoise_method,
            "denoise_sigma": float(denoise_sigma),
        }

        self.gray_value_limits = gray_value_limits
        self.figure_size = figure_size

        self._cache: dict[int, dict[str, Any]] = {}

        self.figure = None
        self.scan_axis = None
        self.profile_axis = None
        self.status_text = None
        self._key_connection = None

        # Manually selected profile depth for each B-scan.
        self.selected_depths: dict[int, int] = {}

        self._line_placement_enabled = False
        self._move_line_button = None
        self._click_connection = None

        self._create_figure()
        self._update_display()

    def _process_scan(
        self,
        bscan_index: int,
    ) -> dict[str, Any]:
        """Preprocess and extract the profile for one B-scan."""

        if bscan_index in self._cache:
            return self._cache[bscan_index]

        processed = preprocess_bscan(
            volume=self.volume,
            bscan_index=bscan_index,
            **self.preprocessing_config,
        )

        profile_depth = self.selected_depths.get(
            bscan_index,
            self.profile_config["profile_depth"],
        )

        crop_height, crop_width = (
            processed.normalized_crop.shape
        )

        if not 0 <= profile_depth < crop_height:
            raise ValueError(
                f"profile_depth={profile_depth} is outside the processed "
                f"crop for B-scan {bscan_index}. Valid rows are 0 to "
                f"{crop_height - 1}."
            )

        profile = extract_intensity_profile(
            bscan=processed.normalized_crop,
            start=(0, profile_depth),
            depth=profile_depth,
            end=(crop_width - 1, profile_depth),
            stepsize=self.profile_config["stepsize"],
            profile_margin=self.profile_config["profile_margin"],
            aggregation=self.profile_config["aggregation"],
            denoise=self.profile_config["denoise"],
            denoise_method=self.profile_config[
                "denoise_method"
            ],
            denoise_sigma=self.profile_config[
                "denoise_sigma"
            ],
            plot=False,
            data=True,
        )

        result = {
            "processed": processed,
            "profile": profile,
            "selected_profile_depth": int(profile_depth),
            "profile_depth_source": (
                "manual"
                if bscan_index in self.selected_depths
                else "configured_default"
            ),
        }

        self._cache[bscan_index] = result

        return result

    def _create_figure(self) -> None:
        """Create the aligned scan and profile figure."""

        self.figure, axes = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=self.figure_size,
            sharex=True,
            gridspec_kw={
                "height_ratios": [1.4, 1.0],
            },
        )

        self.scan_axis = axes[0]
        self.profile_axis = axes[1]

        button_axis = self.figure.add_axes(
            [0.81, 0.01, 0.16, 0.045]
        )

        self._move_line_button = Button(
            button_axis,
            "Move profile line",
        )

        self._move_line_button.on_clicked(
            self._toggle_line_placement
        )

        self.status_text = self.figure.text(
            0.5,
            0.01,
            "",
            ha="center",
            va="bottom",
        )

        self._key_connection = (
            self.figure.canvas.mpl_connect(
                "key_press_event",
                self._handle_key,
            )
        )

        self._click_connection = (
            self.figure.canvas.mpl_connect(
                "button_press_event",
                self._handle_scan_click,
            )
        )

    def _update_display(self) -> None:
        """Process and display the current B-scan."""

        result = self._process_scan(
            self.current_index
        )

        processed = result["processed"]
        profile = result["profile"]

        image = processed.normalized_crop
        profile_depth = profile.depth
        profile_margin = profile.profile_margin

        horizontal_position = (
            np.asarray(
                profile.distance,
                dtype=np.float32,
            )
            + float(profile.start[0])
        )

        self.scan_axis.clear()
        self.profile_axis.clear()

        self.scan_axis.imshow(
            image,
            cmap="gray",
            aspect="auto",
            extent=(
                -0.5,
                image.shape[1] - 0.5,
                image.shape[0] - 0.5,
                -0.5,
            ),
        )

        if profile_margin > 0:
            self.scan_axis.axhspan(
                max(
                    -0.5,
                    profile_depth
                    - profile_margin
                    - 0.5,
                ),
                min(
                    image.shape[0] - 0.5,
                    profile_depth
                    + profile_margin
                    + 0.5,
                ),
                alpha=0.20,
                label=(
                    f"{2 * profile_margin + 1}-row "
                    "profile band"
                ),
            )

        self.scan_axis.axhline(
            profile_depth,
            linestyle="--",
            linewidth=1.5,
            label="Profile center",
        )

        self.scan_axis.set_title(
            f"B-scan {self.current_index} of "
            f"{self.number_of_scans - 1}"
        )

        self.scan_axis.set_ylabel(
            "Depth below BM"
        )

        self.scan_axis.legend(
            loc="upper right"
        )

        self.profile_axis.plot(
            horizontal_position,
            profile.gray_values,
            linewidth=1.0,
        )

        self.profile_axis.set_xlim(
            -0.5,
            image.shape[1] - 0.5,
        )

        self.profile_axis.set_ylim(
            *self.gray_value_limits
        )

        self.profile_axis.set_title(
            "Extracted intensity profile"
        )

        self.profile_axis.set_xlabel(
            "Horizontal position"
        )

        self.profile_axis.set_ylabel(
            "Gray value"
        )

        self.profile_axis.grid(
            alpha=0.25
        )

        self.status_text.set_text(
            "← previous | → next | Home first | "
            "End last | C center  "
            f"| cached scans: {len(self._cache)}"
        )

        self.figure.tight_layout(
            rect=(0, 0.07, 1, 1)
        )

        self.figure.canvas.draw_idle()

        def _toggle_line_placement(
            self,
            _event: Any,
        ) -> None:
            """
            Enable or disable manual profile-line placement.
            """
            self._line_placement_enabled = (
                not self._line_placement_enabled
            )

            if self._line_placement_enabled:
                self._move_line_button.label.set_text(
                    "Click scan to place"
                )
            else:
                self._move_line_button.label.set_text(
                    "Move profile line"
                )

            self.figure.canvas.draw_idle()

        def _handle_scan_click(
            self,
            event: Any,
        ) -> None:
            """
            Move the profile line to the clicked vertical location.
            """
            if not self._line_placement_enabled:
                return

            if event.inaxes is not self.scan_axis:
                return

            if event.ydata is None:
                return

            current_result = self._process_scan(
                self.current_index
            )

            crop_height = current_result[
                "processed"
            ].normalized_crop.shape[0]

            selected_depth = int(
                np.clip(
                    round(event.ydata),
                    0,
                    crop_height - 1,
                )
            )

            self.selected_depths[
                self.current_index
            ] = selected_depth

            # Remove the old profile because it was produced at another depth.
            self._cache.pop(
                self.current_index,
                None,
            )

            self._line_placement_enabled = False

            self._move_line_button.label.set_text(
                "Move profile line"
            )

            self._update_display()

            depth_source = result[
                "profile_depth_source"
            ]

            self.status_text.set_text(
                "← previous | → next | Home first | End last | C center "
                "| Move profile line, then click scan "
                f"| depth={int(profile.depth)} ({depth_source}) "
                f"| cached scans: {len(self._cache)}"
            )

    def _handle_key(
        self,
        event: Any,
    ) -> None:
        """Handle keyboard navigation."""

        if event.key == "right":
            new_index = min(
                self.current_index + 1,
                self.number_of_scans - 1,
            )

        elif event.key == "left":
            new_index = max(
                self.current_index - 1,
                0,
            )

        elif event.key == "home":
            new_index = 0

        elif event.key == "end":
            new_index = (
                self.number_of_scans - 1
            )

        elif event.key in {"c", "C"}:
            new_index = (
                self.number_of_scans // 2
            )

        else:
            return

        if new_index != self.current_index:
            self.current_index = new_index
            self._update_display()

    def get_current_result(
        self,
    ) -> dict[str, Any]:
        """Return the current processed scan and profile."""

        return self._process_scan(
            self.current_index
        )

    @property
    def cached_indices(self) -> list[int]:
        """Return indices already processed and cached."""

        return sorted(
            self._cache
        )

    def clear_cache(self) -> None:
        """Remove all cached scans except the currently displayed scan."""

        current_result = self._cache.get(
            self.current_index
        )

        self._cache.clear()

        if current_result is not None:
            self._cache[
                self.current_index
            ] = current_result

        self._update_display()

    def show(self) -> None:
        """Display the interactive browser."""

        plt.show()


def create_volume_profile_browser(
    volume: Any,
    **kwargs: Any,
) -> VolumeProfileBrowser:
    """Create an interactive browser for one OCT volume."""

    return VolumeProfileBrowser(
        volume=volume,
        **kwargs,
    )

    def get_current_metadata(
        self,
    ) -> dict[str, Any]:
        """
        Return metadata for the currently displayed profile.
        """
        result = self.get_current_result()

        return {
            "bscan_index": int(
                self.current_index
            ),
            "profile_depth": int(
                result["profile"].depth
            ),
            "profile_depth_source": result[
                "profile_depth_source"
            ],
            "profile_margin": int(
                result["profile"].profile_margin
            ),
            "aggregation": str(
                result["profile"].aggregation
            ),
            "denoise_enabled": bool(
                result["profile"].denoise_enabled
            ),
            "denoise_method": str(
                result["profile"].denoise_method
            ),
            "denoise_sigma": (
                result["profile"].denoise_sigma
            ),
        }