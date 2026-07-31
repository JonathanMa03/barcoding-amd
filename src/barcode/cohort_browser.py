from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from src.barcode.data import (
    load_e2e_volume,
    preprocess_bscan,
)
from src.barcode.model_threshold import (
    extract_intensity_profile,
)


class CohortCenterProfileBrowser:
    """
    Interactive browser for center B-scans across multiple E2E volumes.

    Each volume is loaded independently, and its center B-scan is selected
    using ``len(volume) // 2``. This supports volumes containing different
    numbers of B-scans.

    Navigation
    ----------
    Right arrow:
        Move to the next volume.
    Left arrow:
        Move to the previous volume.
    Home:
        Move to the first volume.
    End:
        Move to the final volume.

    Notes
    -----
    Volumes are loaded, processed, and cached lazily. A volume is processed
    only when it is first displayed.
    """

    def __init__(
        self,
        volume_records: Sequence[Mapping[str, Any]],
        *,
        initial_position: int = 0,
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
        gray_value_limits: tuple[float, float] = (0.0, 300.0),
        figure_size: tuple[float, float] = (14, 8),
    ) -> None:
        self.volume_records = [
            dict(record)
            for record in volume_records
        ]

        if not self.volume_records:
            raise ValueError(
                "volume_records must contain at least one volume."
            )

        for position, record in enumerate(
            self.volume_records
        ):
            required_fields = {
                "subject_id",
                "progression_group",
                "e2e_path",
            }

            missing_fields = (
                required_fields
                - set(record)
            )

            if missing_fields:
                raise KeyError(
                    f"Volume record {position} is missing fields: "
                    f"{sorted(missing_fields)}."
                )

            record["e2e_path"] = Path(
                record["e2e_path"]
            )

            if not record["e2e_path"].exists():
                raise FileNotFoundError(
                    "E2E file not found for subject "
                    f"{record['subject_id']}: "
                    f"{record['e2e_path']}"
                )

        if not (
            0
            <= initial_position
            < len(self.volume_records)
        ):
            raise IndexError(
                "initial_position must lie between 0 and "
                f"{len(self.volume_records) - 1}."
            )

        self.current_position = int(
            initial_position
        )

        self.preprocessing_config = {
            "bm_layer_name": bm_layer_name,
            "depth_below_bm": depth_below_bm,
            "reference_row": reference_row,
            "normalize": normalize,
            "normalization_scope": normalization_scope,
            "normalization_center_row": (
                normalization_center_row
            ),
            "normalization_margin": normalization_margin,
            "lower_percentile": lower_percentile,
            "upper_percentile": upper_percentile,
        }

        self.profile_config = {
            "profile_depth": int(
                profile_depth
            ),
            "profile_margin": int(
                profile_margin
            ),
            "aggregation": aggregation,
            "stepsize": float(
                stepsize
            ),
        }

        self.gray_value_limits = (
            gray_value_limits
        )

        self.figure_size = figure_size

        # Cache keyed by subject ID.
        self._cache: dict[
            Any,
            dict[str, Any],
        ] = {}

        self.figure = None
        self.scan_axis = None
        self.profile_axis = None
        self.status_text = None
        self._key_connection = None

        self._create_figure()
        self._update_display()

    def _process_volume(
        self,
        position: int,
    ) -> dict[str, Any]:
        """
        Load one volume, select its center B-scan, preprocess it, and
        extract its intensity profile.
        """
        record = self.volume_records[
            position
        ]

        subject_id = record[
            "subject_id"
        ]

        if subject_id in self._cache:
            return self._cache[
                subject_id
            ]

        volume = load_e2e_volume(
            record["e2e_path"]
        )

        number_of_bscans = len(volume)

        if number_of_bscans == 0:
            raise ValueError(
                f"Subject {subject_id} contains no B-scans."
            )

        center_index = (
            number_of_bscans // 2
        )

        processed = preprocess_bscan(
            volume=volume,
            bscan_index=center_index,
            **self.preprocessing_config,
        )

        profile_depth = (
            self.profile_config[
                "profile_depth"
            ]
        )

        crop_height, crop_width = (
            processed.normalized_crop.shape
        )

        if not (
            0
            <= profile_depth
            < crop_height
        ):
            raise ValueError(
                f"profile_depth={profile_depth} is outside the "
                f"processed crop for subject {subject_id}. "
                f"Valid rows are 0 to {crop_height - 1}."
            )

        profile = extract_intensity_profile(
            bscan=processed.normalized_crop,
            start=(
                0,
                profile_depth,
            ),
            depth=profile_depth,
            end=(
                crop_width - 1,
                profile_depth,
            ),
            stepsize=self.profile_config[
                "stepsize"
            ],
            profile_margin=(
                self.profile_config[
                    "profile_margin"
                ]
            ),
            aggregation=(
                self.profile_config[
                    "aggregation"
                ]
            ),
            plot=False,
            data=True,
            overlay=False,
        )

        result = {
            "record": record,
            "volume": volume,
            "number_of_bscans": (
                number_of_bscans
            ),
            "center_index": center_index,
            "processed": processed,
            "profile": profile,
        }

        self._cache[
            subject_id
        ] = result

        return result

    def _create_figure(
        self,
    ) -> None:
        """Create the shared-axis scan and profile figure."""

        self.figure, axes = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=self.figure_size,
            sharex=True,
            gridspec_kw={
                "height_ratios": [
                    1.4,
                    1.0,
                ],
            },
        )

        self.scan_axis = axes[0]
        self.profile_axis = axes[1]

        self.status_text = (
            self.figure.text(
                0.5,
                0.01,
                "",
                ha="center",
                va="bottom",
            )
        )

        self._key_connection = (
            self.figure.canvas.mpl_connect(
                "key_press_event",
                self._handle_key,
            )
        )

    def _update_display(
        self,
    ) -> None:
        """Display the center B-scan for the current volume."""

        result = self._process_volume(
            self.current_position
        )

        record = result["record"]
        processed = result[
            "processed"
        ]
        profile = result["profile"]

        image = (
            processed.normalized_crop
        )

        profile_depth = float(
            profile.depth
        )

        profile_margin = int(
            profile.profile_margin
        )

        horizontal_position = (
            np.asarray(
                profile.distance,
                dtype=np.float32,
            )
            + float(
                profile.start[0]
            )
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

        progression_group = str(
            record["progression_group"]
        ).title()

        self.scan_axis.set_title(
            f"Subject {record['subject_id']} "
            f"({progression_group} progressor) | "
            f"Center B-scan {result['center_index']} "
            f"of {result['number_of_bscans'] - 1}"
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
            "Extracted center-scan intensity profile"
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
            "← previous volume | → next volume | "
            "Home first | End last  "
            f"| volume {self.current_position + 1}/"
            f"{len(self.volume_records)} "
            f"| cached: {len(self._cache)}"
        )

        self.figure.tight_layout(
            rect=(
                0,
                0.04,
                1,
                1,
            )
        )

        self.figure.canvas.draw_idle()

    def _handle_key(
        self,
        event: Any,
    ) -> None:
        """Handle navigation between volumes."""

        if event.key == "right":
            new_position = min(
                self.current_position + 1,
                len(self.volume_records) - 1,
            )

        elif event.key == "left":
            new_position = max(
                self.current_position - 1,
                0,
            )

        elif event.key == "home":
            new_position = 0

        elif event.key == "end":
            new_position = (
                len(self.volume_records) - 1
            )

        else:
            return

        if (
            new_position
            != self.current_position
        ):
            self.current_position = (
                new_position
            )

            self._update_display()

    def get_current_result(
        self,
    ) -> dict[str, Any]:
        """Return the current volume, processed scan, and profile."""

        return self._process_volume(
            self.current_position
        )

    @property
    def cached_subject_ids(
        self,
    ) -> list[Any]:
        """Return subject IDs already loaded and processed."""

        return list(
            self._cache.keys()
        )

    def clear_cache(
        self,
    ) -> None:
        """
        Remove cached volumes except the currently displayed subject.
        """
        current_result = (
            self._process_volume(
                self.current_position
            )
        )

        current_subject_id = (
            current_result["record"][
                "subject_id"
            ]
        )

        self._cache.clear()

        self._cache[
            current_subject_id
        ] = current_result

        self._update_display()

    def show(
        self,
    ) -> None:
        """Display the interactive browser."""

        plt.show()


def create_cohort_center_profile_browser(
    volume_records: Sequence[
        Mapping[str, Any]
    ],
    **kwargs: Any,
) -> CohortCenterProfileBrowser:
    """
    Create an interactive browser for center B-scans across a cohort.
    """

    return CohortCenterProfileBrowser(
        volume_records=volume_records,
        **kwargs,
    )