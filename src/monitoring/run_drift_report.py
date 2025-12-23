"""
Standalone CLI for running drift detection reports.

Usage:
    # Simulate mild drift (default)
    uv run python -m src.monitoring.run_drift_report
    
    # Simulate severe drift
    uv run python -m src.monitoring.run_drift_report --simulate-drift severe
    
    # Use custom current data file
    uv run python -m src.monitoring.run_drift_report --current-data path/to/current.parquet

Exit codes:
    0 - No significant drift detected
    1 - Warning-level drift (logged, pipeline should continue)
    2 - Critical drift (pipeline should stop)
    3 - Runtime error (file not found, etc.)
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from src.monitoring.drift_config import DriftSeverity
from src.monitoring.drift_detection import detect_drift
from src.monitoring.drift_simulation import (
    DriftType,
    apply_drift_scenario,
    split_data_for_drift,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default paths
DEFAULT_REFERENCE_DATA = Path("data/processed/transactions_enriched.parquet")
DEFAULT_OUTPUT_DIR = Path("reports")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run drift detection on fraud detection data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    parser.add_argument(
        "--reference-data",
        type=Path,
        default=DEFAULT_REFERENCE_DATA,
        help=f"Path to reference (baseline) data. Default: {DEFAULT_REFERENCE_DATA}",
    )
    
    # Mutually exclusive: either simulate drift OR provide current data
    data_source = parser.add_mutually_exclusive_group()
    
    data_source.add_argument(
        "--simulate-drift",
        type=str,
        choices=["none", "mild", "severe"],
        default="mild",
        help="Simulate drift scenario. Default: mild",
    )
    
    data_source.add_argument(
        "--current-data",
        type=Path,
        default=None,
        help="Path to current (production) data. Overrides --simulate-drift",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save HTML reports. Default: {DEFAULT_OUTPUT_DIR}",
    )
    
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Skip saving HTML report (console output only)",
    )
    
    parser.add_argument(
        "--reference-ratio",
        type=float,
        default=0.5,
        help="Ratio of data to use as reference when simulating. Default: 0.5",
    )
    
    return parser.parse_args()


def load_parquet(path: Path) -> pd.DataFrame:
    """Load parquet file with error handling."""
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} rows from {path}")
    return df


def main() -> int:
    """
    Main entry point for drift detection CLI.
    
    Returns:
        Exit code (0=none, 1=warning, 2=critical, 3=error)
    """
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("DRIFT DETECTION CLI")
    logger.info("=" * 60)
    
    try:
        # Load source data
        source_df = load_parquet(args.reference_data)
        
        # Determine reference and current data
        if args.current_data:
            # Mode: Real current data provided
            logger.info(f"Using provided current data: {args.current_data}")
            reference_df = source_df
            current_df = load_parquet(args.current_data)
        else:
            # Mode: Simulate drift
            logger.info(f"Simulating drift scenario: {args.simulate_drift}")
            
            # Split source data into reference and current
            reference_df, current_df = split_data_for_drift(
                source_df,
                reference_ratio=args.reference_ratio,
            )
            
            # Apply drift scenario to current data
            drift_type: DriftType = args.simulate_drift  # type: ignore
            current_df = apply_drift_scenario(current_df, drift_type)
        
        # Run drift detection
        result = detect_drift(
            reference_df=reference_df,
            current_df=current_df,
            save_report=not args.no_report,
            output_dir=args.output_dir,
        )
        
        # Determine exit code based on severity
        if result.severity == DriftSeverity.CRITICAL:
            logger.error("CRITICAL DRIFT DETECTED - Pipeline should STOP")
            return 2
        elif result.severity == DriftSeverity.WARNING:
            logger.warning("WARNING: Drift detected but within acceptable range")
            return 1
        else:
            logger.info("No significant drift detected")
            return 0
            
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        return 3
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 3


if __name__ == "__main__":
    sys.exit(main())