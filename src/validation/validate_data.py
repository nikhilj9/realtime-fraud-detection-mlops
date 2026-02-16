"""
Standalone validation script for raw data.

Usage:
    uv run python -m src.validation.validate_data --data-path data/raw/creditcard.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.validation.expectations import SuiteType, validate_dataframe


def validate_raw_data(data_path: str) -> bool:
    """Validate raw data file."""
    path = Path(data_path)

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return False

    print(f"Loading data from {path}...")

    # Load based on extension
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        print(f"ERROR: Unsupported format: {path.suffix}")
        return False

    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Validate as RAW data
    print("Running RAW data validation...")
    success, errors = validate_dataframe(df, suite_type=SuiteType.RAW)

    print(f"\n{'=' * 50}")
    print(f"VALIDATION {'PASSED ✓' if success else 'FAILED ✗'}")
    print(f"{'=' * 50}")

    if not success:
        for error in errors:
            print(f"  - {error}")

    return success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate raw data.")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/creditcard.csv",
        help="Path to raw data file",
    )
    args = parser.parse_args()

    success = validate_raw_data(args.data_path)
    sys.exit(0 if success else 1)
