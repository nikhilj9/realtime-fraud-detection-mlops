"""Model training - Production version (single model).

This module is designed to be called by an orchestrator (training_flow.py).
"""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import xgboost as xgb
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.models.evaluate import evaluate_model
from src.utils import Config, get_logger

logger = get_logger(__name__)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_split_data(
    train_path: Path,
    val_path: Path,
    test_path: Path
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load train, validation, and test data from parquet files."""

    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)

    target = "is_fraud"
    features = [c for c in train_df.columns if c != target]

    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]
    X_test, y_test = test_df[features], test_df[target]

    # Cast integer columns to float64 to handle missing values at inference
    int_cols = X_train.select_dtypes(include=['int64', 'int32', 'int16', 'int8']).columns
    X_train.loc[:, int_cols] = X_train[int_cols].astype('float64')
    X_val.loc[:, int_cols] = X_val[int_cols].astype('float64')
    X_test.loc[:, int_cols] = X_test[int_cols].astype('float64')

    logger.info(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    logger.info(f"Converted {len(int_cols)} integer columns to float64")

    return X_train, y_train, X_val, y_val, X_test, y_test


# =============================================================================
# PREPROCESSING
# =============================================================================

def create_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing pipeline with categorical encoding and numeric scaling."""

    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", RobustScaler(), num_cols)
        ],
        remainder="passthrough"
    )


# =============================================================================
# TRAINING
# =============================================================================

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: Config
) -> tuple[Any, dict[str, float]]:
    """
    Train the production model (XGBoost with class weights).

    Returns:
        Tuple of (trained_pipeline, validation_metrics)
    """
    logger.info("Training XGBoost with class weights")

    # Handle class imbalance via scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info(f"scale_pos_weight: {scale_pos_weight:.2f}")

    preprocessor = create_preprocessor(X_train)

    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("classifier", xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            n_estimators=config.model.n_estimators,
            max_depth=config.model.max_depth,
            learning_rate=config.model.learning_rate,
            subsample=config.model.subsample,
            colsample_bytree=config.model.colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            random_state=config.model.random_state,
            n_jobs=-1,
            verbosity=0
        ))
    ])

    pipeline.fit(X_train, y_train)

    val_metrics, _, _ = evaluate_model(pipeline, X_val, y_val, "Validation")

    return pipeline, val_metrics


def retrain_on_train_val(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: Config
) -> Any:
    """
    Retrain on combined train + validation data for final model.

    This is a common practice to maximize training data before final deployment.
    """
    logger.info("Retraining on train + validation")

    X_combined = pd.concat([X_train, X_val], ignore_index=True)
    y_combined = pd.concat([y_train, y_val], ignore_index=True)

    scale_pos_weight = (y_combined == 0).sum() / (y_combined == 1).sum()

    preprocessor = create_preprocessor(X_combined)

    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("classifier", xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            n_estimators=config.model.n_estimators,
            max_depth=config.model.max_depth,
            learning_rate=config.model.learning_rate,
            subsample=config.model.subsample,
            colsample_bytree=config.model.colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            random_state=config.model.random_state,
            n_jobs=-1,
            verbosity=0
        ))
    ])

    pipeline.fit(X_combined, y_combined)

    return pipeline


# =============================================================================
# EVALUATION
# =============================================================================

def final_evaluation(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: Config
) -> dict[str, float]:
    """
    Final evaluation on test set.

    Note: This function only computes metrics. MLflow logging is handled by the orchestrator.
    """
    test_metrics, _, _ = evaluate_model(model, X_test, y_test, "TEST")
    return test_metrics


# =============================================================================
# MODEL PERSISTENCE
# =============================================================================

def save_model(model: Any, output_path: Path) -> None:
    """Save model to disk using joblib."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    logger.info(f"Saved: {output_path}")
