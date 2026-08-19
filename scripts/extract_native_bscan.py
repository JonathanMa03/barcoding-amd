"""Extract one unprocessed native B-scan directly from a HEYEX E2E volume.

Run from the repository root, for example:

    python scripts/extract_native_bscan.py --subject 8 --scan 48 --group fast

This script deliberately does not flatten, crop, normalize, denoise, or run
the detector. It saves a report-ready grayscale PNG under ``results/original``
using the same identity convention as automatic detector outputs, with the
suffix ``_original`` instead of ``_automatic``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loading.data_loading import load_bscan, load_e2e_volume


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract an untreated native B-scan from an E2E volume."
    )
    parser.add_argument("--subject", type=int, required=True,
                        help="Patient number, for example 8 for ea8.E2E.")
    parser.add_argument("--scan", type=int, required=True,
                        help="Zero-based B-scan index to extract.")
    parser.add_argument("--group", choices=("fast", "slow"), required=True,
                        help="Progression group used in the output filename.")
    parser.add_argument(
        "--e2e-directory", type=Path, default=Path("data/heyex/meta"),
        help="Directory searched recursively for ea<subject>.E2E.",
    )
    parser.add_argument(
        "--output-directory", type=Path, default=Path("results/original"),
        help="Directory in which the native PNG is saved.",
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="PNG resolution; this affects presentation only, not scan data.",
    )
    return parser.parse_args()


def find_subject_e2e(directory: Path, subject: int) -> Path:
    """Find a case-insensitive ea<subject>.E2E file recursively."""
    if not directory.is_dir():
        raise NotADirectoryError(f"E2E directory not found: {directory}")
    matches = []
    for path in directory.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".e2e":
            continue
        match = re.fullmatch(r"ea[_-]?(\d+)", path.stem, re.IGNORECASE)
        if match and int(match.group(1)) == subject:
            matches.append(path)
    if not matches:
        raise FileNotFoundError(
            f"No E2E file for subject {subject} found under {directory}."
        )
    if len(matches) > 1:
        listed = "\n  ".join(str(path) for path in matches)
        raise RuntimeError(
            f"Multiple E2E files matched subject {subject}:\n  {listed}"
        )
    return matches[0]


def output_stem(group: str, subject: int, scan: int) -> str:
    """Match automatic naming: fast_08_bscan_048_original."""
    return f"{group}_{subject:02d}_bscan_{scan:03d}_original"


def save_native_png(image, output_path: Path, *, dpi: int) -> Path:
    """Render the unchanged native array as a grayscale, axis-free PNG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = image.shape
    figure = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, frameon=False)
    axis = figure.add_axes((0, 0, 1, 1))
    axis.imshow(image, cmap="gray", aspect="auto", interpolation="none")
    axis.set_axis_off()
    figure.savefig(output_path, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(figure)
    return output_path.resolve()


def main() -> None:
    args = parse_args()
    e2e_path = find_subject_e2e(args.e2e_directory, args.subject)
    volume = load_e2e_volume(e2e_path)
    native_bscan = load_bscan(volume, args.scan)

    output_path = args.output_directory / (
        output_stem(args.group, args.subject, args.scan) + ".png"
    )
    saved_path = save_native_png(native_bscan, output_path, dpi=args.dpi)
    print(f"E2E source: {e2e_path.resolve()}")
    print(f"Native B-scan index: {args.scan}")
    print(f"Native shape: {native_bscan.shape}")
    print("Preprocessing applied: none")
    print(f"Saved: {saved_path}")


if __name__ == "__main__":
    main()
