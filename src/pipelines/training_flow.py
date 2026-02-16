"""
Fraud Detection Training Pipeline - Prefect Orchestration
End-to-end: ETL → Validation → Drift Check → Feature Engineering → Training → Evaluation
INTEGRATED WITH: MLflow (Experiment Tracking & Model Registry)
"""

import gc
import os
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv
from prefect import flow, task
from prefect.artifacts import create_markdown_artifact

from src.data.generation import generate, save
from src.features.engineering import run_feature_engineering
from src.models.evaluate import plot_confusion_matrix
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

load_dotenv()

logger = get_logger(__name__)

# =============================================================================
# MLFLOW CONFIGURATION
# =============================================================================

def configure_mlflow_tracking() -> str:
    """
    Configure MLflow to use the correct tracking server.

    Priority:
    1. MLFLOW_TRACKING_URI environment variable (if set)
    2. Default to localhost:5000 (for local development with port-forwarding)

    In Kubernetes, the prefect-worker deployment sets MLFLOW_TRACKING_URI=http://mlflow:5000
    Locally, you must either:
      - Set MLFLOW_TRACKING_URI=http://localhost:5000, OR
      - Run: kubectl port-forward -n fraud-detection svc/mlflow 5000:5000
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {tracking_uri}")
    return tracking_uri


# =============================================================================
# S3/MINIO HELPERS
# =============================================================================

def download_from_s3_if_needed(input_path: str) -> Path:
    """
    If input_path is an S3 URL, download it to a temp location.
    Otherwise, return the path as-is.
    """
    if str(input_path).startswith("s3://"):
        import boto3

        # Parse S3 path: s3://bucket/key
        s3_path = str(input_path).replace("s3://", "")
        bucket, key = s3_path.split("/", 1)

        local_path = Path(tempfile.gettempdir()) / Path(key).name

        logger.info(f"Downloading {input_path} → {local_path}")

        s3_client = boto3.client(
            's3',
            endpoint_url=os.getenv('MLFLOW_S3_ENDPOINT_URL'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )

        s3_client.download_file(bucket, key, str(local_path))
        logger.info(f"Download complete: {local_path}")

        return local_path

    return Path(input_path)


# =============================================================================
# TASKS
# =============================================================================

@task(name="ETL: Raw to Parquet", retries=2)
def task_etl_process(raw_path: Path, processed_root: Path, config: Config) -> dict:
    """
    1. Reads RAW csv (Kaggle data).
    2. Runs GENERATION to add IDs, Timestamps, and extra columns.
    3. Runs FEATURE ENGINEERING on the enriched data.
    4. Saves Versioned Parquet.

    Returns:
        dict with keys: 'enriched_path', 'train', 'val', 'test'
    """
    if not str(raw_path).startswith("s3:/"):
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data not found at: {raw_path}")

    # Versioning: YYYY/MM/DD
    date_str = datetime.now()
    version_dir = processed_root / f"{date_str.year}/{date_str.month:02d}/{date_str.day:02d}"
    version_dir.mkdir(parents=True, exist_ok=True)

    config.paths.raw_data = raw_path

    # Step 1: Enrich raw data (31 cols → 46 cols)
    logger.info(f"Step 1: Enriching raw data from {raw_path}")
    enriched_df = generate(config)

    # Step 2: Save enriched data
    output_file = version_dir / config.generation.output_filename
    save(enriched_df, output_file)

    # Step 3: Feature engineering on enriched data
    # NOW we capture the returned paths instead of discarding them
    logger.info(f"Step 2: Running feature engineering on {output_file}")
    fe_paths = run_feature_engineering(
        input_path=output_file,
        output_dir=version_dir,
        train_ratio=config.split.train_ratio,
        val_ratio=config.split.val_ratio,
    )

    # Return both the enriched parquet path AND the train/val/test paths
    return {
        "enriched_path": output_file,
        "train": fe_paths["train"],
        "val": fe_paths["val"],
        "test": fe_paths["test"],
        "encoder": fe_paths["encoder"],
        "features": fe_paths["features"],
    }


@task(name="Validate Raw Data", retries=0, retry_delay_seconds=10)
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


@task(name="Check Data Drift", persist_result=False)
def task_check_drift(df: pd.DataFrame) -> None:
    """Checks for drift. Fails flow if CRITICAL drift detected."""
    reference_df, current_df = split_data_for_drift(df, reference_ratio=0.5)

    # Delete the original df - we only need the two halves now
    del df
    gc.collect()

    result = detect_drift(
        reference_df=reference_df,
        current_df=current_df,
        save_report=True
    )

    # Free drift dataframes immediately after analysis
    del reference_df, current_df
    gc.collect()

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


# REMOVED: task_feature_engineering - it was causing duplicate FE runs
# Feature engineering is now done inside task_etl_process


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


# =============================================================================
# MLFLOW EXPERIMENT TASK
# =============================================================================

@task(name="MLflow Training & Logging")
def task_mlflow_training(
    X_train, y_train, X_val, y_val, X_test, y_test, config: Config,
    encoder_path: str = None, features_path: str = None
) -> Path:
    """
    Runs training inside an MLflow run context.

    Separated as a task to ensure clean MLflow state management.
    Each task runs in isolation, preventing run state from leaking.
    """
    run_name = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")

        # Log hyperparameters
        mlflow.log_params({
            "model_type": config.model.model_type,
            "n_estimators": config.model.n_estimators,
            "max_depth": config.model.max_depth,
            "learning_rate": config.model.learning_rate,
            "smote_ratio": config.model.smote_ratio,
            "train_ratio": config.split.train_ratio
        })

        # Convert integer columns to float64 to satisfy MLflow signature requirements
        int_cols = X_train.select_dtypes(include=["int8", "int16", "int32", "int64"]).columns
        if not int_cols.empty:
            X_train[int_cols] = X_train[int_cols].astype("float64")
            X_val[int_cols] = X_val[int_cols].astype("float64")
            X_test[int_cols] = X_test[int_cols].astype("float64")
            logger.info(f"Cast {len(int_cols)} int columns to float64 for MLflow signature.")

        # Train initial model
        _, val_metrics = train_model(X_train, y_train, X_val, y_val, config)

        mlflow.log_metrics({
            "val_pr_auc": val_metrics["pr_auc"],
            "val_f1": val_metrics["f1_score"],
            "val_roc_auc": val_metrics["roc_auc"]
        })

        # Retrain on train+val
        final_model = retrain_on_train_val(X_train, y_train, X_val, y_val, config)

        # Final evaluation on test set
        test_metrics = final_evaluation(final_model, X_test, y_test, config)

        mlflow.log_metrics({
            "test_pr_auc": test_metrics["pr_auc"],
            "test_f1": test_metrics["f1_score"],
            "test_roc_auc": test_metrics["roc_auc"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"]
        })

        # Generate and log confusion matrix artifact
        y_pred = final_model.predict(X_test)
        fig = plot_confusion_matrix(y_test.values, y_pred, title="Test Confusion Matrix")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)
        logger.info("Logged confusion_matrix.png to MLflow")

        # Log preprocessing artifacts to MLflow
        if encoder_path:
            mlflow.log_artifact(str(encoder_path), "preprocessing")
            logger.info("Logged target_encoder.joblib to MLflow")
        if features_path:
            mlflow.log_artifact(str(features_path), "preprocessing")
            logger.info("Logged feature_columns.joblib to MLflow")

        # Save model locally (for API to load from PVC)
        model_path = config.paths.models / "champion_model.joblib"

        # Convert category columns to string for MLflow schema inference
        input_example = X_test.head(1).copy()
        for col in input_example.select_dtypes(include=["category"]).columns:
            input_example[col] = input_example[col].astype(str)
        for col in input_example.select_dtypes(include=["int8", "int16", "int32", "int64"]).columns:
            input_example[col] = input_example[col].astype("float64")

        # Register model in MLflow Model Registry
        mlflow.sklearn.log_model(
            sk_model=final_model,
            name="model",
            registered_model_name="fraud-detection-model",
            input_example=input_example,
        )

        logger.info(f"Model saved to {model_path} and registered in MLflow")

        return model_path


# =============================================================================
# MAIN FLOW
# =============================================================================

@flow(name="Fraud Detection Training (End-to-End)", log_prints=True)
def training_flow(config_path: str = "src/config/config.yaml", skip_drift: bool = False):
    """
    Orchestrates the Fraud Detection Training Pipeline.
    Logs experiments to MLflow.
    """
    config = load_config(Path(config_path))

    # Step 0: Configure MLflow tracking (THE FIX)
    # This MUST happen before any MLflow operations
    configure_mlflow_tracking()

    # Set experiment (now it talks to PostgreSQL, not local filesystem)
    experiment_name = config.mlflow.experiment_name if hasattr(config, "mlflow") else "fraud-detection"
    mlflow.set_experiment(experiment_name)
    logger.info(f"MLflow experiment: {experiment_name}")

    # Step 1: Download raw data from MinIO if needed
    raw_input = download_from_s3_if_needed(config.paths.raw_data)

    # Step 2: ETL (enrich + feature engineering)
    # Now returns dict with enriched_path AND train/val/test paths
    logger.info("Starting ETL: enrich raw data + feature engineering...")
    etl_output = task_etl_process(
        raw_path=raw_input,
        processed_root=Path("/app/data/processed"),
        config=config
    )
    gc.collect()
    logger.info(f"ETL complete. Output paths: train={etl_output['train']}, val={etl_output['val']}, test={etl_output['test']}")

    # Step 3: Validation (on enriched parquet)
    validated_df = task_validate_raw_data(etl_output["enriched_path"])
    logger.info(f"Validation passed. Shape: {validated_df.shape}")

    # Step 4: Drift Check
    if not skip_drift:
        task_check_drift(validated_df)

    del validated_df
    gc.collect()

    # Step 5: Load train/val/test splits (using paths from ETL)
    X_train, y_train, X_val, y_val, X_test, y_test = task_load_data(etl_output)

    # Step 6-9: Training, Evaluation, Saving (all inside MLflow context)
    saved_path = task_mlflow_training(
        X_train, y_train, X_val, y_val, X_test, y_test, config,
        encoder_path=etl_output.get("encoder"),
        features_path=etl_output.get("features"),
    )

    logger.info(f"Pipeline complete. Model saved to: {saved_path}")


if __name__ == "__main__":
    training_flow()
