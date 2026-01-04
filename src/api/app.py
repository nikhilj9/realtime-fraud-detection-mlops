"""FastAPI application for fraud detection."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
import joblib
from pathlib import Path
from prometheus_fastapi_instrumentator import Instrumentator

from src.utils.logger import get_logger, log_session_start, log_session_end
from src.api.routes.predict import router as predict_router

logger = get_logger(__name__)

# Paths to artifacts
MODEL_PATH = Path("models/champion_model.joblib")
ENCODER_PATH = Path("data/processed/target_encoder.joblib")
FEATURE_COLUMNS_PATH = Path("data/processed/feature_columns.joblib")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown."""
    
    # Startup
    log_session_start(logger)
    logger.info("Starting up - loading model artifacts...")
    
    try:
        app.state.model = joblib.load(MODEL_PATH)
        logger.info("Loaded model from models/champion_model.joblib")
        
        app.state.encoder = joblib.load(ENCODER_PATH)
        logger.info("Loaded encoder from data/processed/target_encoder.joblib")
        
        app.state.feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
        logger.info(f"Loaded {len(app.state.feature_columns)} feature columns")
        
    except Exception as e:
        logger.error(f"Failed to load artifacts: {e}")
        raise
    
    yield
    
    # Shutdown
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
# --------------------------------------------

@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

app.include_router(predict_router)