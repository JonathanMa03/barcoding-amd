# data.py
#
# Responsibilities:
# - Load Heidelberg E2E volumes.
# - Inspect available retinal-layer annotations.
# - Load a selected B-scan.
# - Select the central B-scan or a user-specified B-scan.
# - Iterate through B-scans in one volume.
# - Construct subject and volume registries.
#
# Inputs:
# - E2E file paths
# - Subject identifiers
# - B-scan selection parameters
#
# Outputs:
# - EyeVolume objects
# - Two-dimensional B-scan arrays
# - Volume registry records
#
# This module should not perform flattening, normalization,
# denoising, feature extraction, or detection.
from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class LoadedScan:
    """Source-independent input consumed by the preprocessing stage."""

    image: np.ndarray
    source_type: str
    source_path: Path
    metadata: dict[str, Any]
    bscan_index: int | None = None
    layer_boundary: np.ndarray | None = None


def _read_grayscale_png(png_path: Path) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python is required to load PNG files.") from exc

    image = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read PNG image: {png_path}")
    return image.astype(np.float32)


def load_e2e_with_metadata(
    e2e_path: str | Path,
    *,
    bscan_index: int | None = None,
    selection: str = "center",
    layer_name: str = "BM",
    metadata: Mapping[str, Any] | None = None,
) -> LoadedScan:
    """Load one E2E B-scan together with its selected layer and metadata."""
    path = Path(e2e_path)
    volume = load_e2e_volume(path)
    resolved_index, image = select_bscan(
        volume, bscan_index=bscan_index, selection=selection
    )

    # Imported lazily to keep loading responsibilities in this module while
    # avoiding a preprocessing dependency at module import time.
    from src.preprocess.preprocessing import get_layer_boundary

    boundary = get_layer_boundary(volume, resolved_index, layer_name=layer_name)
    combined_metadata = dict(metadata or {})
    combined_metadata.update({
        "layer_name": layer_name,
        "available_layers": inspect_volume_layers(volume),
        "volume_shape": tuple(int(value) for value in volume.shape),
    })
    return LoadedScan(
        image=image,
        source_type="e2e",
        source_path=path.resolve(),
        metadata=combined_metadata,
        bscan_index=resolved_index,
        layer_boundary=boundary,
    )


def load_json_png(
    json_path: str | Path,
    png_path: str | Path | None = None,
) -> LoadedScan:
    """Load a JSON metadata/annotation file and its companion PNG image."""
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    if png_path is None:
        png_path = json_path.with_suffix(".png")
    png_path = Path(png_path)
    if not png_path.exists():
        raise FileNotFoundError(f"PNG file not found: {png_path}")

    with json_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if not isinstance(metadata, dict):
        raise ValueError("The top-level JSON value must be an object.")

    image = _read_grayscale_png(png_path)
    expected_shape = metadata.get("image_shape")
    if expected_shape is not None and tuple(expected_shape) != image.shape:
        metadata["declared_image_shape"] = expected_shape
        metadata["png_image_shape"] = list(image.shape)

    boundary_value = metadata.get("layer_boundary")
    boundary = (
        np.asarray(boundary_value, dtype=np.float32)
        if boundary_value is not None
        else None
    )
    return LoadedScan(
        image=image,
        source_type="json_png",
        source_path=png_path.resolve(),
        metadata=metadata,
        bscan_index=metadata.get("bscan_index"),
        layer_boundary=boundary,
    )


def load_scan(
    source_path: str | Path,
    *,
    metadata_path: str | Path | None = None,
    **e2e_options: Any,
) -> LoadedScan:
    """Dispatch to the E2E or JSON+PNG loader from the file suffix."""
    path = Path(source_path)
    suffix = path.suffix.lower()
    if suffix == ".e2e":
        return load_e2e_with_metadata(path, **e2e_options)
    if suffix == ".json":
        return load_json_png(path)
    if suffix == ".png":
        if metadata_path is None:
            metadata_path = path.with_suffix(".json")
        return load_json_png(metadata_path, path)
    raise ValueError("source_path must end in .E2E, .json, or .png")


def save_loaded_scan(scan: LoadedScan, output_path: str | Path) -> Path:
    """Persist a loaded scan as a compressed handoff between scripts."""
    output_path = Path(output_path).with_suffix(".npz")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        image=scan.image,
        layer_boundary=(
            scan.layer_boundary
            if scan.layer_boundary is not None
            else np.asarray([], dtype=np.float32)
        ),
        source_type=np.asarray(scan.source_type),
        source_path=np.asarray(str(scan.source_path)),
        bscan_index=np.asarray(-1 if scan.bscan_index is None else scan.bscan_index),
        metadata=np.asarray(json.dumps(scan.metadata, default=str)),
    )
    return output_path.resolve()


def load_saved_scan(input_path: str | Path) -> LoadedScan:
    """Restore a scan written by :func:`save_loaded_scan`."""
    with np.load(Path(input_path), allow_pickle=False) as archive:
        boundary = archive["layer_boundary"].astype(np.float32)
        index = int(archive["bscan_index"])
        return LoadedScan(
            image=archive["image"].astype(np.float32),
            source_type=str(archive["source_type"]),
            source_path=Path(str(archive["source_path"])),
            metadata=json.loads(str(archive["metadata"])),
            bscan_index=None if index < 0 else index,
            layer_boundary=boundary if boundary.size else None,
        )

@dataclass(frozen=True)
class VolumeRecord:
    """
    Metadata describing one E2E volume.

    Attributes
    ----------
    subject_id:
        Subject or volume identifier.
    e2e_path:
        Path to the E2E file.
    progression_group:
        Optional progression category, such as ``"fast"`` or ``"slow"``.
    file_exists:
        Whether the E2E file exists when the record is created.
    metadata:
        Optional additional subject or acquisition metadata.
    """

    subject_id: str | int
    e2e_path: Path
    progression_group: str | None
    file_exists: bool
    metadata: dict[str, Any]

def load_e2e_volume(e2e_path: str | Path):
    """
    Load a Heidelberg E2E OCT volume with eyepy.

    Parameters
    ----------
    e2e_path:
        Path to the E2E file.

    Returns
    -------
    eyepy.EyeVolume
        Loaded OCT volume.
    """
    try:
        import eyepy as ep
    except ImportError as exc:
        raise ImportError(
            "eyepy is required to load E2E files. "
            "Install it with `pip install eyepy`."
        ) from exc

    e2e_path = Path(e2e_path)

    if not e2e_path.exists():
        raise FileNotFoundError(f"E2E file not found: {e2e_path}")

    volume = ep.import_heyex_e2e(str(e2e_path))

    return volume

def load_bscan(volume, bscan_index: int) -> np.ndarray:
    """
    Extract one B-scan as a 2D NumPy array.

    Parameters
    ----------
    volume:
        Loaded EyeVolume.
    bscan_index:
        Zero-based B-scan index.

    Returns
    -------
    np.ndarray
        B-scan with shape (height, width).
    """
    if bscan_index < 0 or bscan_index >= len(volume):
        raise IndexError(
            f"B-scan index {bscan_index} is outside valid range "
            f"0 to {len(volume) - 1}."
        )

    bscan = np.asarray(volume[bscan_index].data)

    if bscan.ndim != 2:
        raise ValueError(
            f"Expected a 2D B-scan, received shape {bscan.shape}."
        )

    return bscan.astype(np.float32)

def inspect_volume_layers(volume) -> list[str]:


    """
    Return the available layer-annotation names in an EyeVolume.

    This helper is useful because eyepy layer names may differ between
    imported datasets or software versions.
    """
    layers = getattr(volume, "layers", None)

    if layers is None:
        return []

    if isinstance(layers, dict):
        return list(layers.keys())

    try:
        return list(layers)
    except TypeError:
        return []

def resolve_bscan_index(
    volume,
    bscan_index: int | None = None,
    selection: str = "center",
) -> int:
    """
    Resolve the B-scan index to use from a volume.

    Parameters
    ----------
    volume:
        Loaded EyeVolume.
    bscan_index:
        Explicit zero-based B-scan index. This must be supplied when
        ``selection="index"``.
    selection:
        Scan-selection strategy. Supported values are:

        - ``"center"``: select ``len(volume) // 2``.
        - ``"index"``: use the explicitly supplied ``bscan_index``.

    Returns
    -------
    int
        Resolved zero-based B-scan index.
    """
    number_of_bscans = len(volume)

    if number_of_bscans == 0:
        raise ValueError(
            "The supplied volume contains no B-scans."
        )

    normalized_selection = selection.lower()

    if normalized_selection == "center":
        return number_of_bscans // 2

    if normalized_selection == "index":
        if bscan_index is None:
            raise ValueError(
                "bscan_index must be provided when selection='index'."
            )

        if not 0 <= bscan_index < number_of_bscans:
            raise IndexError(
                f"B-scan index {bscan_index} is outside valid range "
                f"0 to {number_of_bscans - 1}."
            )

        return int(bscan_index)

    raise ValueError(
        "selection must be either 'center' or 'index'."
    )

def select_bscan(
    volume,
    *,
    bscan_index: int | None = None,
    selection: str = "center",
) -> tuple[int, np.ndarray]:
    """
    Select and load one B-scan from a volume.

    Parameters
    ----------
    volume:
        Loaded EyeVolume.
    bscan_index:
        Explicit B-scan index when ``selection="index"``.
    selection:
        Either ``"center"`` or ``"index"``.

    Returns
    -------
    resolved_index:
        Selected zero-based B-scan index.
    bscan:
        Selected B-scan as a two-dimensional float32 array.
    """
    resolved_index = resolve_bscan_index(
        volume=volume,
        bscan_index=bscan_index,
        selection=selection,
    )

    bscan = load_bscan(
        volume=volume,
        bscan_index=resolved_index,
    )

    return resolved_index, bscan

def iterate_bscans(
    volume,
    indices: Sequence[int] | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """
    Iterate through selected B-scans in one volume.

    Parameters
    ----------
    volume:
        Loaded EyeVolume.
    indices:
        Optional sequence of zero-based B-scan indices. If omitted,
        every B-scan is returned.

    Yields
    ------
    index:
        Zero-based B-scan index.
    bscan:
        Two-dimensional float32 B-scan.
    """
    if indices is None:
        resolved_indices = range(
            len(volume)
        )
    else:
        resolved_indices = indices

    for index in resolved_indices:
        resolved_index = int(index)

        yield (
            resolved_index,
            load_bscan(
                volume=volume,
                bscan_index=resolved_index,
            ),
        )

def build_volume_registry(
    e2e_directory: str | Path,
    subject_ids: Sequence[str | int],
    *,
    filename_template: str = "ea{subject_id}.E2E",
    progression_group: str | None = None,
    metadata_by_subject: (
        dict[str | int, dict[str, Any]]
        | None
    ) = None,
) -> list[VolumeRecord]:
    """
    Construct records for a collection of E2E volumes.

    Parameters
    ----------
    e2e_directory:
        Directory containing the E2E files.
    subject_ids:
        Subject or volume identifiers.
    filename_template:
        Filename template containing ``{subject_id}``.
    progression_group:
        Optional group assigned to all provided subjects.
    metadata_by_subject:
        Optional mapping from subject ID to additional metadata.

    Returns
    -------
    list[VolumeRecord]
        Structured volume registry.
    """
    e2e_directory = Path(
        e2e_directory
    )

    metadata_by_subject = (
        {}
        if metadata_by_subject is None
        else metadata_by_subject
    )

    records: list[VolumeRecord] = []

    for subject_id in subject_ids:
        filename = filename_template.format(
            subject_id=subject_id
        )

        e2e_path = (
            e2e_directory
            / filename
        )

        subject_metadata = dict(
            metadata_by_subject.get(
                subject_id,
                {},
            )
        )

        records.append(
            VolumeRecord(
                subject_id=subject_id,
                e2e_path=e2e_path,
                progression_group=progression_group,
                file_exists=e2e_path.exists(),
                metadata=subject_metadata,
            )
        )

    return records

def build_grouped_volume_registry(
    e2e_directory: str | Path,
    progression_groups: Mapping[
        str,
        Sequence[str | int],
    ],
    *,
    filename_template: str = "ea{subject_id}.E2E",
) -> list[VolumeRecord]:
    """
    Build one registry from multiple progression groups.

    Parameters
    ----------
    e2e_directory:
        Directory containing the E2E files.
    progression_groups:
        Mapping such as ``{"fast": [8, 9], "slow": [17, 23]}``.
    filename_template:
        Filename template containing ``{subject_id}``.

    Returns
    -------
    list[VolumeRecord]
        Combined volume records.
    """
    records: list[VolumeRecord] = []

    for group_name, subject_ids in progression_groups.items():
        records.extend(
            build_volume_registry(
                e2e_directory=e2e_directory,
                subject_ids=subject_ids,
                filename_template=filename_template,
                progression_group=group_name,
            )
        )

    return records
