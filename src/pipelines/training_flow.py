from datetime import timedelta
from pathlib import Path
import pandas as pd
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact

# Import your existing business logic
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

# Initialize Logger (Prefect captures this automatically if log_prints=True)
logger = get_logger(__name__)

# --- TASKS (The Steps) ---

@task(
    name="Validate Raw Data",
    retries=2,                    # ← retry twice on failure
    retry_delay_seconds=10
)
def task_validate_raw_data(data_path: Path) -> pd.DataFrame:
    """Load and validate data. Fails if expectations are not met."""
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    # Load
    if data_path.suffix == ".parquet":
        df = pd.read_parquet(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    # Validate
    success, errors = validate_dataframe(df, suite_type=SuiteType.PROCESSED)
    
    if not success:
        error_msg = f"Data validation failed with {len(errors)} errors: {errors}"
        raise ValueError(error_msg)  # Raising exception fails the task
        
    return df

@task(name="Check Data Drift")
def task_check_drift(df: pd.DataFrame) -> None:
    """Checks for drift. Fails flow if CRITICAL drift detected."""
    # Simulate Reference vs Current
    reference_df, current_df = split_data_for_drift(df, reference_ratio=0.5)
    
    result = detect_drift(
        reference_df=reference_df, 
        current_df=current_df, 
        save_report=True
    )
    
    # Create an artifact in the UI (A nice report card)
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
    cache_key_fn=lambda *_: "v1",  # ← simple versioned cache
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
    
    # Artifact for Metrics
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

# --- FLOW (The Manager) ---

@flow(name="Fraud Detection Training", log_prints=True)
def training_flow(config_path: str = "src/config/config.yaml", skip_drift: bool = False):
    """
    Orchestrates the Fraud Detection Training Pipeline.
    """
    # Setup
    config = load_config(Path(config_path))
    input_path = config.paths.processed_data / config.generation.output_filename
    
    # 1. Validation Gates
    validated_df = task_validate_raw_data(input_path)
    
    if not skip_drift:
        task_check_drift(validated_df)
    
    # 2. Feature Engineering
    paths = task_feature_engineering(input_path, config)
    
    # 3. Load Data
    # Note: Tasks return "Futures" (promises of data), but when passed to other tasks, 
    # Prefect resolves them automatically.
    X_train, y_train, X_val, y_val, X_test, y_test = task_load_data(paths)
    
    # 4. Train
    _, val_metrics = task_train_model(X_train, y_train, X_val, y_val, config)
    
    # 5. Retrain
    final_model = task_retrain(X_train, y_train, X_val, y_val, config)
    
    # 6. Evaluate
    test_metrics = task_evaluate(final_model, X_test, y_test, config)
    
    # 7. Save
    saved_path = task_save_model(final_model, config)
    
    logger.info(f"Flow complete. Model saved to {saved_path}")

if __name__ == "__main__":
    # Allow running this file directly for debugging
    training_flow()