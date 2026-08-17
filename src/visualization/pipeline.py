"""Non-interactive plots for saved preprocessing and detector artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Any

import matplotlib.pyplot as plt
import numpy as np


LABEL_COLORS = {"ea": "tab:orange", "barcoding": "tab:red"}


def plot_detection_result(
    image: np.ndarray,
    labels: Iterable[str],
    output_path: str | Path,
    *,
    title: str = "OCT detector result",
    colors: Mapping[str, str] = LABEL_COLORS,
    figure_options: Mapping[str, Any] | None = None,
) -> Path:
    """Save an OCT image with contiguous EA/barcoding column overlays."""
    image = np.asarray(image)
    labels = np.asarray(list(labels))
    if image.ndim != 2 or labels.shape != (image.shape[1],):
        raise ValueError("labels must contain one value per image column.")

    options = {"figsize": (12, 4), "dpi": 150, **dict(figure_options or {})}
    figure, axis = plt.subplots(**options)
    axis.imshow(image, cmap="gray", aspect="auto")
    for label, color in colors.items():
        mask = labels == label
        starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
        ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
        for index, (start, end) in enumerate(zip(starts, ends)):
            axis.axvspan(start, end + 1, color=color, alpha=0.25,
                         label=label if index == 0 else None)
    axis.set(title=title, xlabel="B-scan column", ylabel="Depth")
    handles, legend_labels = axis.get_legend_handles_labels()
    if handles:
        axis.legend(handles, legend_labels)
    figure.tight_layout()
    output_path = Path(output_path).with_suffix(".png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path)
    plt.close(figure)
    return output_path.resolve()
