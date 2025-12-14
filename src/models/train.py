"""Model training - Production version (single model)."""

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt

from src.utils import Config, get_logger
from src.utils.exceptions import ModelTrainingError
from src.models.evaluate import evaluate_model, plot_confusion_matrix

logger = get_logger(__name__)


def load_split_data(
    train_path: Path,
    val_path: Path,
    test_path: Path
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load train, validation, and test data."""
    
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    
    target = "is_fraud"
    features = [c for c in train_df.columns if c != target]
    
    X_train, y_train = train_df[features], train_df[target]
    X_val, y_val = val_df[features], val_df[target]
    X_test, y_test = test_df[features], test_df[target]
    
    # Cast integer columns to float64 to handle missing values at inference
    int_cols = X_train.select_dtypes(include=['int64', 'int32']).columns
    X_train[int_cols] = X_train[int_cols].astype('float64')
    X_val[int_cols] = X_val[int_cols].astype('float64')
    X_test[int_cols] = X_test[int_cols].astype('float64')
    
    logger.info(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    logger.info(f"Converted {len(int_cols)} integer columns to float64")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def create_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing pipeline."""
    
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", RobustScaler(), num_cols)
        ],
        remainder="passthrough"
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: Config
) -> Tuple[Any, Dict[str, float]]:
    """Train the production model (XGBoost with class weights)."""
    
    logger.info("Training XGBoost with class weights")
    
    # Calculate class weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    logger.info(f"scale_pos_weight: {scale_pos_weight:.2f}")
    
    # Create pipeline
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
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Evaluate on validation
    val_metrics, _, _ = evaluate_model(pipeline, X_val, y_val, "Validation")
    
    return pipeline, val_metrics


def retrain_on_train_val(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    config: Config
) -> Any:
    """Retrain on train + validation for final model."""
    
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


def final_evaluation(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: Config
) -> Dict[str, float]:
    """Final evaluation on test set with MLflow logging."""
    
    mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    mlflow.set_experiment(config.mlflow.experiment_name)
    
    with mlflow.start_run(run_name="Production_XGBoost") as run:
        
        # Log parameters
        mlflow.log_params({
            "model_type": "XGBoost_Weighted",
            "n_estimators": config.model.n_estimators,
            "max_depth": config.model.max_depth,
            "learning_rate": config.model.learning_rate
        })
        
        # Evaluate
        test_metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test, "TEST")
        
        # Log metrics
        mlflow.log_metrics(test_metrics)
        
        # Log confusion matrix
        fig = plot_confusion_matrix(y_test.values, y_pred, title="Test Confusion Matrix")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)
        
        # Log model
        signature = infer_signature(X_test.iloc[:5], model.predict(X_test.iloc[:5]))
        mlflow.sklearn.log_model(model, name="model", signature=signature)
        
        logger.info(f"Run ID: {run.info.run_id}")
        
        return test_metrics


def save_model(model: Any, output_path: Path) -> None:
    """Save model to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    logger.info(f"Saved: {output_path}")