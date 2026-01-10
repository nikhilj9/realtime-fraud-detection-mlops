"""
Fraud Detection Training Pipeline - Prefect Orchestration
End-to-end: ETL → Validation → Drift Check → Feature Engineering → Training → Evaluation
"""

from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact

from src.features.engineering import run_feature_engineering
from src.monitoring.drift_config import DriftSeverity
from src.monitoring.drift_detection import detect_drift
from src.monitoring.drift_simulation import split_data_for_drift
from src.models.train import (
    final_evaluation,
    load_split_data,
    retrain_on_train_val,
    save_model,
    train_model,
)
from src.utils import Config, get_logger, load_config
from src.validation.expectations import SuiteType, validate_dataframe

logger = get_logger(__name__)


# --- NEW TASK: The Missing Link (ETL) ---
@task(name="ETL: Raw to Parquet", retries=2)
def task_etl_process(raw_path: Path, processed_root: Path, config: Config) -> Path:
    """
    Reads RAW csv, runs feature engineering, and saves Versioned Parquet.
    Returns the path to the NEW parquet file.
    """
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found at: {raw_path}")

    # 1. Versioning Logic (YYYY/MM/DD)
    date_str = datetime.now()
    version_dir = processed_root / f"{date_str.year}/{date_str.month:02d}/{date_str.day:02d}"
    version_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = version_dir / config.generation.output_filename
    
    logger.info(f"Running ETL on {raw_path} -> {output_file}")
    
    paths = run_feature_engineering(
        input_path=raw_path,
        output_dir=version_dir,
        train_ratio=config.split.train_ratio,
        val_ratio=config.split.val_ratio,
    )
    
    if not output_file.exists():
        logger.warning(f"Expected output {output_file} not found. Checking directory...")
        found_files = list(version_dir.glob("*.parquet"))
        if found_files:
            output_file = found_files[0]
        else:
            raise FileNotFoundError("ETL failed to produce parquet file.")
            
    return output_file


@task(
    name="Validate Raw Data",
    retries=2,
    retry_delay_seconds=10
)
def task_validate_raw_data(data_path: Path) -> pd.DataFrame:
    """Load and validate data. Fails if expectations are not met."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    elif data_path.suffix == ".csv":
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    success, errors = validate_dataframe(df, suite_type=SuiteType.PROCESSED)
    
    if not success:
        error_msg = f"Data validation failed with {len(errors)} errors: {errors}"
        raise ValueError(error_msg)
        
    return df


@task(name="Check Data Drift")
def task_check_drift(df: pd.DataFrame) -> None:
    """Checks for drift. Fails flow if CRITICAL drift detected."""
    reference_df, current_df = split_data_for_drift(df, reference_ratio=0.5)
    
    result = detect_drift(
        reference_df=reference_df, 
        current_df=current_df, 
        save_report=True
    )
    
    markdown_report = f"""
    # Drift Detection Report
    - **Drift Ratio**: {result.drift_ratio:.1%}
    - **Severity**: {result.severity.value}
    """
    create_markdown_artifact(
        key="drift-report",
        markdown=markdown_report,
        description="Evidently AI Drift Scan"
    )

    if result.severity == DriftSeverity.CRITICAL:
        raise ValueError(f"CRITICAL DRIFT: {result.drift_ratio:.1%}")
    elif result.severity == DriftSeverity.WARNING:
        logger.warning(f"Drift Warning: {result.drift_ratio:.1%}")


@task(
    name="Feature Engineering",
    cache_key_fn=lambda *_: "v1",
    cache_expiration=timedelta(hours=24)
)
def task_feature_engineering(input_path: Path, config: Config) -> dict:
    return run_feature_engineering(
        input_path=input_path,
        output_dir=config.paths.processed_data,
        train_ratio=config.split.train_ratio,
        val_ratio=config.split.val_ratio,
    )


@task(name="Load Split Data")
def task_load_data(paths: dict):
    return load_split_data(paths["train"], paths["val"], paths["test"])


@task(name="Train Initial Model")
def task_train_model(X_train, y_train, X_val, y_val, config: Config):
    model, metrics = train_model(X_train, y_train, X_val, y_val, config)
    logger.info(f"Validation Metrics: {metrics}")
    return model, metrics


@task(name="Retrain Full Model")
def task_retrain(X_train, y_train, X_val, y_val, config: Config):
    return retrain_on_train_val(X_train, y_train, X_val, y_val, config)


@task(name="Evaluate Model")
def task_evaluate(model, X_test, y_test, config: Config):
    metrics = final_evaluation(model, X_test, y_test, config)
    logger.info(f"Test Metrics: {metrics}")
    
    create_markdown_artifact(
        key="model-metrics",
        markdown=f"### Test Results\nPR-AUC: {metrics['pr_auc']:.4f}",
        description="Final Model Performance"
    )
    return metrics


@task(name="Save Model")
def task_save_model(model, config: Config):
    model_path = config.paths.models / "champion_model.joblib"
    save_model(model, model_path)
    return model_path


# --- FLOW ---

@flow(name="Fraud Detection Training (End-to-End)", log_prints=True)
def training_flow(config_path: str = "src/config/config.yaml", skip_drift: bool = False):
    """
    Orchestrates the Fraud Detection Training Pipeline.
    """
    config = load_config(Path(config_path))
    
    # 1. Input is now RAW data
    raw_input = config.paths.raw_data
    
    # 2. Run ETL
    processed_parquet_path = task_etl_process(
        raw_path=raw_input, 
        processed_root=Path("/app/data/processed"), 
        config=config
    )
    
    # 3. Validation
    validated_df = task_validate_raw_data(processed_parquet_path)
    
    # 4. Drift Check
    if not skip_drift:
        task_check_drift(validated_df)
    
    # 5. Feature Engineering
    paths = task_feature_engineering(processed_parquet_path, config)
    
    # 6. Load Data
    X_train, y_train, X_val, y_val, X_test, y_test = task_load_data(paths)
    
    # 7. Train
    _, val_metrics = task_train_model(X_train, y_train, X_val, y_val, config)
    
    # 8. Retrain
    final_model = task_retrain(X_train, y_train, X_val, y_val, config)
    
    # 9. Evaluate
    test_metrics = task_evaluate(final_model, X_test, y_test, config)
    
    # 10. Save
    saved_path = task_save_model(final_model, config)
    
    logger.info(f"Flow complete. Model saved to {saved_path}")


if __name__ == "__main__":
    training_flow()