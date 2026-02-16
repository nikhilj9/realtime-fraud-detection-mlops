"""Production training pipeline with data validation and drift detection."""

import argparse
import sys
from pathlib import Path

import pandas as pd

# --- IMPORTS ---
from src.data.generation import generate, save
from src.features.engineering import run_feature_engineering
from src.models.train import (
    final_evaluation,
    load_split_data,
    retrain_on_train_val,
    save_model,
    train_model,
)
from src.monitoring.drift_config import DriftSeverity
from src.monitoring.drift_detection import detect_drift
from src.monitoring.drift_simulation import split_data_for_drift
from src.utils import Config, get_logger, load_config
from src.validation.expectations import SuiteType, validate_dataframe

logger = get_logger(__name__)


class DataValidationError(Exception):
    """Raised when data validation fails."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        message = f"Data validation failed with {len(errors)} error(s):\n"
        message += "\n".join(f"  - {e}" for e in errors)
        super().__init__(message)


def validate_raw_data(data_path: Path) -> pd.DataFrame:
    """Load and validate processed data before training."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info(f"Loading data from {data_path}")

    # Load based on extension
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")

    logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Validate as PROCESSED data
    logger.info("Running data validation...")
    success, errors = validate_dataframe(df, suite_type=SuiteType.PROCESSED)

    if not success:
        logger.error(f"Validation failed with {len(errors)} error(s)")
        for error in errors:
            logger.error(f"  - {error}")
        raise DataValidationError(errors)

    logger.info("Data validation PASSED")
    return df


def check_data_drift(df: pd.DataFrame) -> None:
    """
    Check for data drift using temporal split.
    Raises DataValidationError on CRITICAL drift.
    """
    logger.info("Running drift detection (Reference: Oldest 50%, Current: Newest 50%)...")

    # Simulate Reference vs Current by splitting time-sorted data
    reference_df, current_df = split_data_for_drift(df, reference_ratio=0.5)

    result = detect_drift(
        reference_df=reference_df,
        current_df=current_df,
        save_report=True
    )

    if result.severity == DriftSeverity.CRITICAL:
        msg = f"CRITICAL DRIFT DETECTED: {result.drift_ratio:.1%} columns drifted. See reports/."
        logger.error(msg)
        # We treat critical drift as a validation failure that stops pipeline
        raise DataValidationError([msg])

    elif result.severity == DriftSeverity.WARNING:
        logger.warning(f"DRIFT WARNING: {result.drift_ratio:.1%} columns drifted. Training continues.")
    else:
        logger.info("Drift check PASSED (No significant drift)")


def run_pipeline(config: Config, skip_drift: bool = False) -> None:
    """Run production training pipeline with validation and drift gates."""

    logger.info("=" * 60)
    logger.info("PRODUCTION TRAINING PIPELINE")
    logger.info("=" * 60)

    # --- Step 0 - Data Enrichment ---
    logger.info("Step 0: Data Enrichment (Generation)")
    # This reads raw CSV (via config.paths.raw_data) and adds IDs/Timestamps
    enriched_df = generate(config)

    # Define where the enriched data goes
    enriched_path = config.paths.processed_data / config.generation.output_filename
    save(enriched_df, enriched_path)
    # ---------------------------------------

    # Step 1: Validate Raw Data (GATE)
    logger.info("Step 1: Data Validation")

    try:
        # Load and validate schema (now pointing to the newly generated file)
        validated_df = validate_raw_data(enriched_path)

        # Step 1.5: Check for Drift (GATE)
        if not skip_drift:
            logger.info("Step 1.5: Drift Detection")
            check_data_drift(validated_df)
        else:
            logger.info("Step 1.5: Skipping Drift Detection (requested via flag)")

    except DataValidationError:
        logger.error("=" * 60)
        logger.error("PIPELINE ABORTED: Data quality/drift check failed")
        logger.error("=" * 60)
        raise  # Re-raise to stop pipeline

    # Step 2: Feature Engineering
    logger.info("Step 2: Feature Engineering")
    paths = run_feature_engineering(
        input_path=enriched_path, # Use the enriched file
        output_dir=config.paths.processed_data,
        train_ratio=config.split.train_ratio,
        val_ratio=config.split.val_ratio,
    )

    # Step 3: Load Data
    logger.info("Step 3: Loading Data")
    X_train, y_train, X_val, y_val, X_test, y_test = load_split_data(
        paths["train"], paths["val"], paths["test"]
    )

    # Step 4: Train on Train, Validate on Val
    logger.info("Step 4: Training Model")
    model, val_metrics = train_model(X_train, y_train, X_val, y_val, config)
    logger.info(f"Validation PR-AUC: {val_metrics['pr_auc']:.4f}")

    # Step 5: Retrain on Train + Val
    logger.info("Step 5: Retraining on Train + Validation")
    final_model = retrain_on_train_val(X_train, y_train, X_val, y_val, config)

    # Step 6: Final Evaluation on Test
    logger.info("Step 6: Final Evaluation on Test")
    test_metrics = final_evaluation(final_model, X_test, y_test, config)

    # Step 7: Save Model
    logger.info("Step 7: Saving Model")
    model_path = config.paths.models / "champion_model.joblib"
    save_model(final_model, model_path)

    # Summary
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE ")
    logger.info(f"Validation PR-AUC: {val_metrics['pr_auc']:.4f}")
    logger.info(f"Test PR-AUC: {test_metrics['pr_auc']:.4f}")
    logger.info(f"Model: {model_path}")
    logger.info("=" * 60)


def main() -> int:
    """Main entry point with proper exit codes."""
    parser = argparse.ArgumentParser(
        description="Run production training pipeline with data validation."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/config/config.yaml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--skip-drift",
        action="store_true",
        help="Skip drift detection step",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        run_pipeline(config, skip_drift=args.skip_drift)
        return 0  # Success
    except DataValidationError:
        return 1  # Validation/Drift failure
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 2  # File error
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 3  # Other error


if __name__ == "__main__":
    sys.exit(main())
