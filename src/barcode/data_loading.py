# load_e2e(...)
# load_volume(...)
# list_available_layers(...)
# get_metadata(...)
# summarize_volume(...)
from pathlib import Path
from typing import Any

import eyepy as ep
import numpy as np


def find_e2e_files(data_dir: str | Path) -> list[Path]:
    """
    Find HEYEX E2E files in a directory.

    Parameters
    ----------
    data_dir:
        Directory containing HEYEX .E2E or .e2e files.

    Returns
    -------
    list[Path]
        Sorted list of E2E file paths.
    """
    data_dir = Path(data_dir)

    e2e_files = sorted(data_dir.glob("*.e2e")) + sorted(data_dir.glob("*.E2E"))

    return e2e_files


def load_e2e(path: str | Path):
    """
    Load a HEYEX E2E file as an eyepy EyeVolume object.

    Parameters
    ----------
    path:
        Path to a .E2E or .e2e file.

    Returns
    -------
    eyepy.core.eyevolume.EyeVolume
        Loaded OCT volume.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"E2E file not found: {path}")

    return ep.import_heyex_e2e(path)


def summarize_e2e_file(path: str | Path) -> dict[str, Any]:
    """
    Summarize one E2E file without loading all derived outputs.

    Parameters
    ----------
    path:
        Path to E2E file.

    Returns
    -------
    dict
        Basic file summary.
    """
    path = Path(path)

    return {
        "file_name": path.name,
        "path": str(path),
        "size_mb": path.stat().st_size / 1e6,
    }


def summarize_eyevolume(ev) -> dict[str, Any]:
    """
    Summarize a loaded eyepy EyeVolume.

    Parameters
    ----------
    ev:
        Loaded eyepy EyeVolume object.

    Returns
    -------
    dict
        Summary of volume shape, layers, scale, and metadata.
    """
    layer_names = list(ev.layers.keys()) if hasattr(ev, "layers") else []

    summary = {
        "shape": tuple(ev.data.shape) if hasattr(ev, "data") else None,
        "available_layers": layer_names,
        "has_rpe": "RPE" in layer_names,
        "has_bm": "BM" in layer_names,
        "scale": getattr(ev, "scale", None),
        "scale_unit": getattr(ev, "scale_unit", None),
        "scale_x": getattr(ev, "scale_x", None),
        "scale_y": getattr(ev, "scale_y", None),
        "scale_z": getattr(ev, "scale_z", None),
        "laterality": getattr(ev, "laterality", None),
    }

    return summary


def get_layer_array(ev, layer_name: str) -> np.ndarray:
    """
    Extract a segmentation layer as a NumPy array.

    Parameters
    ----------
    ev:
        Loaded eyepy EyeVolume object.
    layer_name:
        Name of layer, such as 'RPE', 'BM', or 'ILM'.

    Returns
    -------
    np.ndarray
        Layer array of shape (n_bscans, width).
    """
    if not hasattr(ev, "layers"):
        raise AttributeError("EyeVolume object does not have a layers attribute.")

    if layer_name not in ev.layers:
        available = list(ev.layers.keys())
        raise KeyError(
            f"Layer '{layer_name}' not found. Available layers: {available}"
        )

    return ev.layers[layer_name].data


def get_core_layers(ev) -> dict[str, np.ndarray]:
    """
    Extract commonly used retinal layers.

    Returns available layers among ILM, RPE, and BM.

    Parameters
    ----------
    ev:
        Loaded eyepy EyeVolume object.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping layer names to arrays.
    """
    core = {}

    for layer in ["ILM", "RPE", "BM"]:
        if hasattr(ev, "layers") and layer in ev.layers:
            core[layer] = ev.layers[layer].data

    return core


def print_e2e_summary(ev) -> None:
    """
    Print a readable summary of a loaded E2E EyeVolume.
    """
    summary = summarize_eyevolume(ev)

    print("EyeVolume Summary")
    print("-----------------")
    print(f"Shape: {summary['shape']}")
    print(f"Available layers: {summary['available_layers']}")
    print(f"Has RPE: {summary['has_rpe']}")
    print(f"Has BM: {summary['has_bm']}")
    print(f"Scale: {summary['scale']}")
    print(f"Scale unit: {summary['scale_unit']}")
    print(f"Scale x: {summary['scale_x']}")
    print(f"Scale y: {summary['scale_y']}")
    print(f"Scale z: {summary['scale_z']}")
    print(f"Laterality: {summary['laterality']}")