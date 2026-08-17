"""Preprocess a loaded pipeline artifact using the configuration below."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loading.data_loading import load_saved_scan
from src.preprocess.preprocessing import preprocess_loaded_scan, save_preprocessed_scan


PIPELINE_CONFIG = {
    "input_path": Path("results/pipeline/loaded_scan.npz"),
    "output_path": Path("results/pipeline/preprocessed_scan.npz"),
}

# Mirrors PREPROCESSING_CONFIG in pipeline_validation.ipynb.
PREPROCESSING_CONFIG = {
    "layer_name": "BM",
    "reference_row": None,
    "flatten_fill_value": 0.0,
    "depth_below_layer": 150,
    "include_boundary": True,
    "require_full_depth": False,
    "crop_fill_value": 0.0,
    "normalization_method": "zscore",
    "lower_percentile": 1.0,
    "upper_percentile": 99.0,
    "denoise_method": "gaussian",
    "gaussian_sigma": (1.0, 0.5),
}


def main() -> None:
    loaded = load_saved_scan(PIPELINE_CONFIG["input_path"])
    processed = preprocess_loaded_scan(loaded, **PREPROCESSING_CONFIG)
    output = save_preprocessed_scan(processed, PIPELINE_CONFIG["output_path"])
    print(f"Preprocessed scan {processed.image.shape} -> {output}")


if __name__ == "__main__":
    main()
