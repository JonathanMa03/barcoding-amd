"""Load one E2E+metadata or JSON+PNG input into a pipeline artifact."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loading.data_loading import load_scan, save_loaded_scan


# Edit this configuration before running: python scripts/load_data.py
CONFIG = {
    # "source_path": Path("results/manual_ground_truth/fast_08_bscan_048_ground_truth.png"),
    "source_path": Path("data/heyex/meta/ea8.E2E"),

    # "metadata_path": Path("results/manual_ground_truth/fast_08_bscan_048_ground_truth.json"),
    "metadata_path": None,

    "output_path": Path("results/pipeline/loaded_scan.npz"),

    # E2E-only options (ignored for PNG/JSON input):
    # "e2e_options": {"selection": "center", "layer_name": "BM"},
    "e2e_options": {"selection": "index", "bscan_index": 48, "layer_name": "BM"},
    
    # Required for automatic result filenames when it cannot be inferred from
    # an accompanying JSON file. Subject ID can also be inferred from ea8.E2E.
    "source_metadata": {"progression_group": "fast", "subject_id": 8},

}


def main() -> None:
    source_path = CONFIG["source_path"]
    suffix = source_path.suffix.lower()
    e2e_options = dict(CONFIG["e2e_options"] if suffix == ".e2e" else {})
    if suffix == ".e2e":
        e2e_options["metadata"] = dict(CONFIG.get("source_metadata", {}))
    scan = load_scan(
        source_path,
        metadata_path=CONFIG.get("metadata_path"),
        **e2e_options,
    )
    output = save_loaded_scan(scan, CONFIG["output_path"])
    print(f"Loaded {scan.source_type} scan {scan.image.shape} -> {output}")


if __name__ == "__main__":
    main()
