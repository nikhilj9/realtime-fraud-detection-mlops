"""Model prediction utilities."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib

from src.utils import get_logger
from src.utils.exceptions import ModelPredictionError

logger = get_logger(__name__)


def load_model(model_path: Path) -> Any:
    """Load model from disk."""
    if not model_path.exists():
        raise ModelPredictionError(f"Model not found: {model_path}")
    
    model = joblib.load(model_path)
    logger.info(f"Loaded: {model_path}")
    return model


def predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Make binary predictions."""
    try:
        return model.predict(X)
    except Exception as e:
        raise ModelPredictionError(f"Prediction failed: {e}") from e


def predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Get fraud probabilities."""
    try:
        return model.predict_proba(X)[:, 1]
    except Exception as e:
        raise ModelPredictionError(f"Prediction failed: {e}") from e


def predict_with_threshold(model: Any, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
    """Predict with custom threshold."""
    probas = predict_proba(model, X)
    return (probas >= threshold).astype(int)


def batch_predict(
    model: Any,
    input_path: Path,
    output_path: Path,
    threshold: float = 0.5
) -> None:
    """Batch prediction on file."""
    
    df = pd.read_parquet(input_path)
    
    if "is_fraud" in df.columns:
        df = df.drop(columns=["is_fraud"])
    
    df["fraud_probability"] = predict_proba(model, df)
    df["is_fraud_predicted"] = (df["fraud_probability"] >= threshold).astype(int)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    logger.info(f"Predictions saved: {output_path}")
    logger.info(f"Fraud detected: {df['is_fraud_predicted'].sum():,} / {len(df):,}")