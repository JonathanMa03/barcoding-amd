"""Extract native E2E B-scans for every saved automatic detector result.

Run from the repository root:

    python scripts/extract_all_native_bscans.py

By default, case identities are read from
``results/automatic_detector/*_automatic.json`` and untreated PNGs are saved
under ``results/original/`` with matching identity-based filenames.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loading.data_loading import load_bscan, load_e2e_volume
from extract_native_bscan import find_subject_e2e, output_stem, save_native_png


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract native scans corresponding to automatic results."
    )
    parser.add_argument(
        "--automatic-directory", type=Path,
        default=Path("results/automatic_detector"),
        help="Directory containing *_automatic.json result files.",
    )
    parser.add_argument(
        "--e2e-directory", type=Path, default=Path("data/heyex/meta"),
        help="Directory searched recursively for ea<subject>.E2E files.",
    )
    parser.add_argument(
        "--output-directory", type=Path, default=Path("results/original"),
        help="Directory in which native PNGs are saved.",
    )
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace native PNGs that already exist.",
    )
    return parser.parse_args()


def read_identity(json_path: Path) -> dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    required = ("progression_group", "subject_id", "bscan_index")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Missing fields {missing} in {json_path.name}")
    group = str(payload["progression_group"]).strip().lower()
    if group not in {"fast", "slow"}:
        raise ValueError(f"Invalid progression_group in {json_path.name}: {group}")
    return {
        "group": group,
        "subject": int(payload["subject_id"]),
        "scan": int(payload["bscan_index"]),
    }


def main() -> None:
    args = parse_args()
    if not args.automatic_directory.is_dir():
        raise NotADirectoryError(
            f"Automatic-results directory not found: {args.automatic_directory}"
        )
    result_paths = sorted(args.automatic_directory.glob("*_automatic.json"))
    if not result_paths:
        raise FileNotFoundError(
            f"No *_automatic.json files found in {args.automatic_directory}"
        )

    processed = skipped = failed = 0
    for position, result_path in enumerate(result_paths, start=1):
        print(f"[{position}/{len(result_paths)}] {result_path.name}")
        try:
            identity = read_identity(result_path)
            output_path = args.output_directory / (
                output_stem(identity["group"], identity["subject"],
                            identity["scan"]) + ".png"
            )
            if output_path.exists() and not args.overwrite:
                skipped += 1
                print(f"  skipped existing: {output_path}")
                continue

            e2e_path = find_subject_e2e(args.e2e_directory, identity["subject"])
            volume = load_e2e_volume(e2e_path)
            native_bscan = load_bscan(volume, identity["scan"])
            saved = save_native_png(native_bscan, output_path, dpi=args.dpi)
            processed += 1
            print(f"  E2E: {e2e_path}")
            print(f"  native shape: {native_bscan.shape}; preprocessing: none")
            print(f"  saved: {saved}")
        except Exception as exc:
            failed += 1
            print(f"  failed: {type(exc).__name__}: {exc}")

    print(
        f"Finished {len(result_paths)} case(s): "
        f"{processed} saved, {skipped} skipped, {failed} failed."
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
