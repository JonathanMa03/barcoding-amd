"""Open the clinician-facing manual interval annotation window.

Run load_data.py and preprocess_data.py first, then invoke this script with a
patient identifier and B-scan number, for example:

    python scripts/manual_annotator.py ea8 48 --group fast
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.manual_annotation import ManualIntervalAnnotator
from src.preprocess.preprocessing import load_preprocessed_scan


def patient_number(value: str) -> int:
    match = re.fullmatch(r"(?:ea)?[_-]?(\d+)", value.strip(), re.IGNORECASE)
    if match is None:
        raise argparse.ArgumentTypeError("patient must look like ea8 or 8")
    return int(match.group(1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annotate EA, Barcoding, and Normal intervals on one scan."
    )
    parser.add_argument("patient", type=patient_number, help="patient, e.g. ea8")
    parser.add_argument("scan", type=int, help="zero-based B-scan number")
    parser.add_argument(
        "--group", choices=("fast", "slow"),
        help="progression group; optional when stored in the artifact metadata",
    )
    parser.add_argument(
        "--input", type=Path,
        default=Path("results/pipeline/preprocessed_scan.npz"),
        help="preprocessed NPZ produced by scripts/preprocess_data.py",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("results/manual_ground_truth"),
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="allow the save button to replace existing ground truth",
    )
    return parser


def resolve_identity(args: argparse.Namespace, artifact) -> dict[str, object]:
    metadata = dict(artifact.metadata)
    stored_subject = metadata.get("subject_id")
    stored_scan = metadata.get("bscan_index", artifact.bscan_index)
    stored_group = metadata.get("progression_group")
    if stored_subject is not None and int(stored_subject) != args.patient:
        raise ValueError(
            f"Requested ea{args.patient}, but the artifact says ea{stored_subject}."
        )
    if stored_scan is not None and int(stored_scan) != args.scan:
        raise ValueError(
            f"Requested B-scan {args.scan}, but the artifact contains {stored_scan}."
        )
    group = args.group or stored_group
    if group is None:
        raise ValueError(
            "Progression group is missing. Pass --group fast or --group slow, "
            "or include progression_group when running load_data.py."
        )
    group = str(group).lower()
    if args.group is not None and stored_group is not None:
        if str(stored_group).lower() != args.group:
            raise ValueError(
                f"Requested group {args.group}, but the artifact says {stored_group}."
            )
    return {
        "progression_group": group,
        "subject_id": args.patient,
        "bscan_index": args.scan,
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.scan < 0:
        raise ValueError("scan must be a nonnegative zero-based index.")
    if not args.input.is_file():
        raise FileNotFoundError(
            f"Preprocessed artifact not found: {args.input}. Run "
            "scripts/load_data.py and scripts/preprocess_data.py first."
        )
    artifact = load_preprocessed_scan(args.input)
    identity = resolve_identity(args, artifact)
    annotator = ManualIntervalAnnotator(
        artifact.image, identity,
        output_directory=args.output_dir,
        source_metadata={**artifact.metadata, "preprocessed_path": args.input.resolve()},
        overwrite=args.overwrite,
    )
    print("Choose EA, Barcoding, or Normal and drag horizontally over the scan.")
    print("Use Save PNG + JSON when annotation is complete.")
    annotator.show()


if __name__ == "__main__":
    main()
