"""Batch-load and preprocess Heidelberg E2E volumes.

Edit ``BATCH_CONFIG`` and ``PREPROCESSING_CONFIG``, then run:

    .venv/bin/python scripts/batch_preprocess_e2e.py

Modes
-----
``volume_all_scans``
    A) Process every B-scan in one E2E volume.
``all_volumes_selected_scan``
    B) Process one selected B-scan from every E2E file.
``all_volumes_all_scans``
    C) Process every B-scan from every E2E file.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loading.data_loading import LoadedScan, load_bscan, load_e2e_volume
from src.preprocess.preprocessing import (
    get_layer_boundary,
    preprocess_loaded_scan,
    save_preprocessed_scan,
)


BATCH_CONFIG = {
    # Choose one:
    #   "volume_all_scans"
    #   "all_volumes_selected_scan"
    #   "all_volumes_all_scans"
    "mode": "all_volumes_all_scans",

    # Used only by mode A.
    "volume_path": Path("data/heyex/meta/example.E2E"),

    # Used by modes B and C. Matching is case-insensitive for .e2e/.E2E.
    "input_directory": Path("data/heyex/meta"),
    "recursive": False,

    # Used only by mode B.
    # "center" selects len(volume) // 2; "index" uses selected_bscan_index.
    "scan_selection": "center",
    "selected_bscan_index": None,

    "output_directory": Path("results/batch_preprocessed"),
    "overwrite": False,
    "continue_on_error": True,
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


def find_e2e_files(directory: Path, *, recursive: bool) -> list[Path]:
    """Return E2E files with case-insensitive suffix matching."""
    iterator = directory.rglob("*") if recursive else directory.glob("*")
    return sorted(
        path for path in iterator
        if path.is_file() and path.suffix.lower() == ".e2e"
    )


def resolve_volume_paths(config: dict[str, Any]) -> list[Path]:
    mode = config["mode"]
    if mode == "volume_all_scans":
        paths = [Path(config["volume_path"])]
    elif mode in {"all_volumes_selected_scan", "all_volumes_all_scans"}:
        input_directory = Path(config["input_directory"])
        if not input_directory.is_dir():
            raise NotADirectoryError(f"E2E directory not found: {input_directory}")
        paths = find_e2e_files(input_directory, recursive=bool(config["recursive"]))
    else:
        raise ValueError(
            "mode must be 'volume_all_scans', "
            "'all_volumes_selected_scan', or 'all_volumes_all_scans'."
        )

    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"E2E file not found: {missing[0]}")
    if not paths:
        raise FileNotFoundError("No E2E files matched the batch configuration.")
    return paths


def resolve_scan_indices(volume: Any, config: dict[str, Any]) -> range | tuple[int]:
    """Resolve all scans or the selected scan for the configured mode."""
    number_of_scans = len(volume)
    if number_of_scans == 0:
        raise ValueError("The E2E volume contains no B-scans.")

    if config["mode"] != "all_volumes_selected_scan":
        return range(number_of_scans)

    selection = str(config["scan_selection"]).lower()
    if selection == "center":
        index = number_of_scans // 2
    elif selection == "index":
        configured_index = config["selected_bscan_index"]
        if configured_index is None:
            raise ValueError(
                "selected_bscan_index is required when scan_selection='index'."
            )
        index = int(configured_index)
    else:
        raise ValueError("scan_selection must be 'center' or 'index'.")

    if not 0 <= index < number_of_scans:
        raise IndexError(
            f"B-scan {index} is outside the valid range 0 to {number_of_scans - 1}."
        )
    return (index,)


def output_path_for_scan(output_directory: Path, e2e_path: Path, index: int) -> Path:
    """Build a collision-resistant output path for one scan."""
    return output_directory / e2e_path.stem / f"bscan_{index:04d}.npz"


def preprocess_volume(
    e2e_path: Path,
    output_directory: Path,
    config: dict[str, Any],
) -> Iterator[dict[str, Any]]:
    """Preprocess configured B-scans from one volume and yield manifest rows."""
    volume = load_e2e_volume(e2e_path)
    scan_indices = resolve_scan_indices(volume, config)

    for index in scan_indices:
        output_path = output_path_for_scan(output_directory, e2e_path, index)
        if output_path.exists() and not config["overwrite"]:
            yield {
                "e2e_path": str(e2e_path.resolve()),
                "bscan_index": index,
                "output_path": str(output_path.resolve()),
                "status": "skipped_existing",
            }
            continue

        try:
            image = load_bscan(volume, index)
            boundary = get_layer_boundary(
                volume,
                index,
                layer_name=PREPROCESSING_CONFIG["layer_name"],
            )
            loaded = LoadedScan(
                image=image,
                source_type="e2e",
                source_path=e2e_path.resolve(),
                metadata={
                    "e2e_filename": e2e_path.name,
                    "number_of_bscans": len(volume),
                    "volume_shape": tuple(int(value) for value in volume.shape),
                },
                bscan_index=index,
                layer_boundary=boundary,
            )
            processed = preprocess_loaded_scan(loaded, **PREPROCESSING_CONFIG)
            saved_path = save_preprocessed_scan(processed, output_path)
            yield {
                "e2e_path": str(e2e_path.resolve()),
                "bscan_index": index,
                "output_path": str(saved_path),
                "output_shape": list(processed.image.shape),
                "status": "processed",
            }
        except Exception as exc:
            if not config["continue_on_error"]:
                raise
            yield {
                "e2e_path": str(e2e_path.resolve()),
                "bscan_index": index,
                "output_path": str(output_path.resolve()),
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }


def main() -> None:
    volume_paths = resolve_volume_paths(BATCH_CONFIG)
    output_directory = Path(BATCH_CONFIG["output_directory"])
    output_directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for volume_number, e2e_path in enumerate(volume_paths, start=1):
        print(f"[{volume_number}/{len(volume_paths)}] {e2e_path}")
        for row in preprocess_volume(e2e_path, output_directory, BATCH_CONFIG):
            rows.append(row)
            print(f"  B-scan {row['bscan_index']:04d}: {row['status']}")

    summary = {
        "mode": BATCH_CONFIG["mode"],
        "preprocessing_config": PREPROCESSING_CONFIG,
        "number_of_volumes": len(volume_paths),
        "counts": {
            status: sum(row["status"] == status for row in rows)
            for status in ("processed", "skipped_existing", "failed")
        },
        "scans": rows,
    }
    manifest_path = output_directory / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    print(f"Finished: {summary['counts']}")
    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
