"""FastAPI application for fraud detection."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import mlflow
from fastapi import FastAPI
from mlflow.tracking import MlflowClient
from prometheus_fastapi_instrumentator import Instrumentator

from src.api.routes.predict import router as predict_router
from src.utils.logger import get_logger, log_session_end, log_session_start

logger = get_logger(__name__)

# MLflow model name (must match training_flow.py)
MODEL_NAME = "fraud-detection-model"

# Fallback PVC paths (used when MLflow is unavailable)
PVC_MODEL_PATH = Path("/app/models/champion_model.joblib")
PVC_ENCODER_PATH = Path("/app/data/processed/target_encoder.joblib")
PVC_FEATURE_COLUMNS_PATH = Path("/app/data/processed/feature_columns.joblib")


# =============================================================================
# ARTIFACT LOADING (MLflow first, PVC fallback)
# =============================================================================

def load_from_mlflow(tracking_uri: str):
    """Load all 3 artifacts from MLflow Model Registry."""
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    # Get latest model version
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise ValueError(f"No model versions found for '{MODEL_NAME}'")

    latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
    run_id = latest.run_id

    logger.info(f"MLflow: model={MODEL_NAME} v{latest.version}, run={run_id}")

    # Load model from registry
    model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{latest.version}")
    logger.info("Loaded model from MLflow Registry")

    # Load preprocessing artifacts from run
    artifact_dir = client.download_artifacts(run_id, "preprocessing")
    encoder = joblib.load(Path(artifact_dir) / "target_encoder.joblib")
    feature_columns = joblib.load(Path(artifact_dir) / "feature_columns.joblib")
    logger.info("Loaded encoder & feature_columns from MLflow artifacts")

    return model, encoder, feature_columns


def load_from_pvc():
    """Fallback: Load from PVC-mounted files."""
    logger.info("Loading artifacts from PVC files (fallback)...")
    model = joblib.load(PVC_MODEL_PATH)
    logger.info("Loaded model from PVC")
    encoder = joblib.load(PVC_ENCODER_PATH)
    logger.info("Loaded encoder from PVC")
    feature_columns = joblib.load(PVC_FEATURE_COLUMNS_PATH)
    logger.info(f"Loaded {len(feature_columns)} feature columns from PVC")
    return model, encoder, feature_columns


def load_artifacts():
    """
    Load model artifacts with priority:
    1. MLflow Model Registry (industry standard)
    2. PVC files (fallback for first deployment)
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")

    if tracking_uri:
        try:
            return load_from_mlflow(tracking_uri)
        except Exception as e:
            logger.warning(f"MLflow loading failed: {e}. Falling back to PVC files.")

    return load_from_pvc()


# =============================================================================
# APP LIFECYCLE
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown."""

    log_session_start(logger)
    logger.info("Starting up - loading model artifacts...")

    try:
        model, encoder, feature_columns = load_artifacts()
        app.state.model = model
        app.state.encoder = encoder
        app.state.feature_columns = feature_columns
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")
        raise

    yield

    logger.info("Shutting down...")
    log_session_end(logger)


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection for transactions using XGBoost",
    version="1.0.0",
    lifespan=lifespan,
)

# Expose /metrics endpoint and instrument standard HTTP metrics
instrumentator = Instrumentator().instrument(app).expose(app)


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


app.include_router(predict_router)
