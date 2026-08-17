"""Consistent filenames for automatic detector JSON and PNG outputs."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping


def resolve_result_identity(
    metadata: Mapping[str, Any],
    *,
    bscan_index: int | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve progression group, subject ID, and the actual B-scan index."""
    values = dict(metadata)
    values.update({key: value for key, value in dict(overrides or {}).items()
                   if value is not None})

    subject_id = values.get("subject_id")
    if subject_id is None:
        candidates = (
            values.get("e2e_filename"),
            values.get("source_path"),
        )
        for candidate in candidates:
            if candidate:
                match = re.search(r"ea[_-]?(\d+)", Path(str(candidate)).stem,
                                  flags=re.IGNORECASE)
                if match:
                    subject_id = int(match.group(1))
                    break

    progression_group = values.get("progression_group")
    resolved_scan = values.get("bscan_index", bscan_index)
    if subject_id is None:
        raise ValueError(
            "Could not resolve subject_id. Add it to source_metadata or the "
            "result script's identity_overrides."
        )
    if progression_group is None:
        raise ValueError(
            "Could not resolve progression_group ('fast' or 'slow'). Add it "
            "to source_metadata or identity_overrides."
        )
    if resolved_scan is None:
        raise ValueError("Could not resolve the B-scan index from the artifact.")

    group = str(progression_group).strip().lower()
    if group not in {"fast", "slow"}:
        raise ValueError("progression_group must be 'fast' or 'slow'.")
    return {
        "progression_group": group,
        "subject_id": int(subject_id),
        "bscan_index": int(resolved_scan),
    }


def automatic_result_stem(identity: Mapping[str, Any]) -> str:
    """Return a stem such as ``fast_08_bscan_048_automatic``."""
    return (
        f"{identity['progression_group']}_"
        f"{int(identity['subject_id']):02d}_bscan_"
        f"{int(identity['bscan_index']):03d}_automatic"
    )
