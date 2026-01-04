"""Prediction endpoint for fraud detection."""

import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, Request

from src.api.schemas.transaction import TransactionRequest, PredictionResponse
from src.utils.logger import get_logger
from src.monitoring.metrics import PREDICTION_TOTAL, PREDICTION_LATENCY

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Predictions"])


def engineer_features(transaction: TransactionRequest, encoder) -> pd.DataFrame:
    """Apply same feature engineering as training."""
    
    data = transaction.model_dump()
    df = pd.DataFrame([data])
    
    # Amount features
    df["amount_inr"] = df["amount"]
    df["log_amount"] = np.log1p(df["amount"])
    
    # Time features
    df["hour"] = (df["time"] / 3600).astype(int) % 24
    df["day_of_week"] = ((df["time"] / 3600) / 24).astype(int) % 7
    df["is_night"] = df["hour"].apply(lambda h: 1 if h <= 6 or h >= 22 else 0)
    df["is_weekend"] = df["day_of_week"].apply(lambda d: 1 if d >= 5 else 0)
    
    # Interaction features
    df["intl_online"] = (df["is_international"] == 1) & (df["transaction_channel"] == 1)
    df["intl_online"] = df["intl_online"].astype(int)
    
    high_amount_threshold = 5000
    df["night_high_amount"] = (df["is_night"] == 1) & (df["amount"] > high_amount_threshold)
    df["night_high_amount"] = df["night_high_amount"].astype(int)
    
    # Target encoding
    if encoder is not None and "merchant_category" in df.columns:
        try:
            df = encoder.transform(df)
        except Exception as e:
            logger.warning(f"Encoding failed for merchant_category: {e}, using 0")
            df["merchant_category_encoded"] = 0.0
    else:
        df["merchant_category_encoded"] = 0.0
    
    return df


def get_risk_level(probability: float) -> str:
    """Convert probability to risk level."""
    if probability < 0.3:
        return "low"
    elif probability < 0.7:
        return "medium"
    else:
        return "high"


@router.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionRequest, request: Request):
    """Predict if a transaction is fraudulent."""
    
    # Start Latency Timer
    with PREDICTION_LATENCY.time():
        try:
            model = request.app.state.model
            encoder = request.app.state.encoder
            feature_columns = request.app.state.feature_columns
            
            df = engineer_features(transaction, encoder)
            
            missing_cols = set(feature_columns) - set(df.columns)
            if missing_cols:
                logger.warning(f"Missing columns (setting to 0): {missing_cols}")
                for col in missing_cols:
                    df[col] = 0
            
            X = df[feature_columns]
            
            logger.debug(f"Feature columns: {list(X.columns)}")
            
            fraud_probability = float(model.predict_proba(X)[0][1])
            is_fraud = fraud_probability >= 0.5
            risk_level = get_risk_level(fraud_probability)
            
            logger.info(f"Prediction: fraud_prob={fraud_probability:.4f}, is_fraud={is_fraud}")
            
            # Record Business Metric
            status_label = "fraud" if is_fraud else "legit"
            PREDICTION_TOTAL.labels(status=status_label, risk_level=risk_level).inc()
            
            return PredictionResponse(
                is_fraud=is_fraud,
                fraud_probability=round(fraud_probability, 4),
                risk_level=risk_level,
            )
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/debug", include_in_schema=False)
async def debug_features(transaction: TransactionRequest, request: Request):
    """Debug endpoint to see generated features."""
    encoder = request.app.state.encoder
    feature_columns = request.app.state.feature_columns
    
    df = engineer_features(transaction, encoder)
    
    dtypes_info = {
        "all_columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "expected_features": feature_columns,
        "missing_features": list(set(feature_columns) - set(df.columns)),
        "extra_features": list(set(df.columns) - set(feature_columns)),
    }
    
    string_columns = [col for col in df.columns if df[col].dtype == "object"]
    dtypes_info["string_columns"] = string_columns
    
    X = df[feature_columns] if all(col in df.columns for col in feature_columns) else None
    if X is not None:
        dtypes_info["X_dtypes"] = {col: str(dtype) for col, dtype in X.dtypes.items()}
        dtypes_info["X_has_strings"] = any(X[col].dtype == "object" for col in X.columns)
    
    return dtypes_info