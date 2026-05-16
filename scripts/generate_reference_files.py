"""
Generate yoy_summary.csv and title_category_map.json from the full payroll CSVs.

Reads from a source directory containing train_set.csv, valid_set.csv, and
test_set.csv, then writes the two pre-computed reference files used by the
Streamlit app into data/processed/.

Usage:
    # From backup directory (default)
    uv run python scripts/generate_reference_files.py

    # From a custom directory
    uv run python scripts/generate_reference_files.py --source /path/to/csvs
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from data622.paths import PROCESSED_DATA_DIR

DEFAULT_SOURCE = Path.home() / "data622_csv_backup"


def generate_yoy_summary(source_dir: Path, out_dir: Path) -> None:
    cols = ["fiscal_year", "title_std", "agency_std", "base_salary", "regular_hours"]
    frames = []
    for name in ("train_set.csv", "valid_set.csv", "test_set.csv"):
        p = source_dir / name
        if p.exists():
            frames.append(pd.read_csv(p, usecols=cols, low_memory=False))
        else:
            print(f"  warning: {p} not found, skipping")

    if not frames:
        raise FileNotFoundError(f"No split CSVs found in {source_dir}")

    df = pd.concat(frames, ignore_index=True)
    summary = (
        df.groupby(["fiscal_year", "title_std", "agency_std"])
        .agg(
            base_salary=("base_salary", "median"),
            headcount=("base_salary", "count"),
            regular_hours=("regular_hours", "median"),
        )
        .reset_index()
    )
    out_path = out_dir / "yoy_summary.csv"
    summary.to_csv(out_path, index=False)
    print(f"  wrote {out_path} ({len(summary):,} rows)")


def generate_title_category_map(source_dir: Path, out_dir: Path) -> None:
    p = source_dir / "train_set.csv"
    if not p.exists():
        raise FileNotFoundError(f"train_set.csv not found in {source_dir}")

    df = pd.read_csv(p, usecols=["title_std", "title_category"], low_memory=False)
    mapping = (
        df.dropna(subset=["title_std", "title_category"])
        .groupby("title_std")["title_category"]
        .agg(lambda x: x.value_counts().index[0])
        .to_dict()
    )
    out_path = out_dir / "title_category_map.json"
    with open(out_path, "w") as f:
        json.dump(mapping, f)
    print(f"  wrote {out_path} ({len(mapping):,} keys)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Directory containing the split CSVs (default: {DEFAULT_SOURCE})",
    )
    args = parser.parse_args()

    source_dir: Path = args.source
    if not source_dir.exists():
        raise SystemExit(f"Source directory not found: {source_dir}")

    print(f"Reading from: {source_dir}")
    print(f"Writing to:  {PROCESSED_DATA_DIR}\n")

    print("Generating yoy_summary.csv...")
    generate_yoy_summary(source_dir, PROCESSED_DATA_DIR)

    print("Generating title_category_map.json...")
    generate_title_category_map(source_dir, PROCESSED_DATA_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()
