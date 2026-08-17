"""Render a saved preprocessed scan and detector labels."""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess.preprocessing import load_preprocessed_scan
from src.visualization.pipeline import plot_detection_result
from src.evaluation.result_naming import automatic_result_stem, resolve_result_identity


CONFIG = {
    "preprocessed_path": Path("results/pipeline/preprocessed_scan.npz"),
    "detection_path": Path("results/pipeline/detections.json"),
    "output_directory": Path("results/pipeline"),
    "identity_overrides": {
        "progression_group": None,
        "subject_id": None,
        "bscan_index": None,
    },
    "title": "EA and barcoding detector output",
    "colors": {"ea": "tab:orange", "barcoding": "tab:red"},
    "figure_options": {"figsize": (12, 4), "dpi": 150},
}


def main() -> None:
    processed = load_preprocessed_scan(CONFIG["preprocessed_path"])
    with CONFIG["detection_path"].open("r", encoding="utf-8") as stream:
        detection = json.load(stream)
    identity = resolve_result_identity(
        processed.metadata,
        bscan_index=processed.bscan_index,
        overrides=CONFIG.get("identity_overrides"),
    )
    output_path = (
        Path(CONFIG["output_directory"])
        / f"{automatic_result_stem(identity)}.png"
    )
    output = plot_detection_result(
        processed.image,
        detection["labels"],
        output_path,
        title=CONFIG["title"],
        colors=CONFIG["colors"],
        figure_options=CONFIG["figure_options"],
    )
    print(f"Visualization -> {output}")


if __name__ == "__main__":
    main()
