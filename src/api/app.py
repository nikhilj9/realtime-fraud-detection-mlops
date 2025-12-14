"""FastAPI application for fraud detection."""

import joblib
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Paths to artifacts (relative to project root)
MODEL_PATH = Path("models/champion_model.joblib")
ENCODER_PATH = Path("data/processed/target_encoder.joblib")
FEATURE_COLUMNS_PATH = Path("data/processed/feature_columns.joblib")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artifacts at startup."""
    logger.info("Starting up - loading model artifacts...")
    
    # Load model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    app.state.model = joblib.load(MODEL_PATH)
    logger.info(f"Loaded model from {MODEL_PATH}")
    
    # Load target encoder
    if ENCODER_PATH.exists():
        app.state.encoder = joblib.load(ENCODER_PATH)
        logger.info(f"Loaded encoder from {ENCODER_PATH}")
    else:
        app.state.encoder = None
        logger.warning("Target encoder not found - will skip target encoding")
    
    # Load feature columns
    if not FEATURE_COLUMNS_PATH.exists():
        raise FileNotFoundError(f"Feature columns not found: {FEATURE_COLUMNS_PATH}")
    app.state.feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    logger.info(f"Loaded {len(app.state.feature_columns)} feature columns")
    
    yield
    
    logger.info("Shutting down...")


app = FastAPI(
    title="Fraud Detection API",
    description="Real-time fraud detection for transactions using XGBoost",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# Include prediction router
from src.api.routes.predict import router as predict_router
app.include_router(predict_router)