# tests/integration/test_pipeline_e2e.py
"""End-to-end integration tests for training pipeline."""

import pytest
import pandas as pd
import numpy as np
import joblib
from unittest.mock import patch

from src.features.engineering import run_feature_engineering
from src.models.train import load_split_data, train_model, retrain_on_train_val, save_model
from src.models.predict import load_model, predict_proba


@pytest.mark.integration
class TestFeatureEngineeringE2E:
    """End-to-end tests for feature engineering stage."""

    def test_feature_engineering_creates_all_artifacts(self, e2e_raw_data_path, tmp_path):
        """Verify feature engineering creates train, val, test, and encoder files."""
        output_dir = tmp_path / "processed"
        paths = run_feature_engineering(e2e_raw_data_path, output_dir)
        
        assert paths["train"].exists(), "Train parquet must exist"
        assert paths["val"].exists(), "Val parquet must exist"
        assert paths["test"].exists(), "Test parquet must exist"
        assert paths["encoder"].exists(), "Encoder must exist"

    def test_feature_engineering_preserves_total_rows(self, e2e_raw_data_path, tmp_path):
        """Verify no data loss across train/val/test splits."""
        original = pd.read_parquet(e2e_raw_data_path)
        output_dir = tmp_path / "processed"
        paths = run_feature_engineering(e2e_raw_data_path, output_dir)
        
        train = pd.read_parquet(paths["train"])
        val = pd.read_parquet(paths["val"])
        test = pd.read_parquet(paths["test"])
        total = len(train) + len(val) + len(test)
        
        assert total == len(original), f"Expected {len(original)} rows, got {total}"

    def test_splits_have_consistent_schema(self, e2e_raw_data_path, tmp_path):
        """Verify all splits have identical column schema."""
        output_dir = tmp_path / "processed"
        paths = run_feature_engineering(e2e_raw_data_path, output_dir)
        
        train = pd.read_parquet(paths["train"])
        val = pd.read_parquet(paths["val"])
        test = pd.read_parquet(paths["test"])
        
        assert list(train.columns) == list(val.columns), "Train/val schema mismatch"
        assert list(val.columns) == list(test.columns), "Val/test schema mismatch"


@pytest.mark.integration
class TestModelTrainingE2E:
    """End-to-end tests for model training stage."""

    def test_train_model_produces_usable_model(
        self, e2e_processed_splits, mock_config
    ):
        """Verify trained model can make predictions."""
        train_path, val_path, test_path = e2e_processed_splits
        X_train, y_train, X_val, y_val, X_test, y_test = load_split_data(
            train_path, val_path, test_path
        )
        
        with patch("src.models.train.evaluate_model") as mock_eval:
            mock_eval.return_value = ({"pr_auc": 0.8}, np.zeros(len(y_val)), np.zeros(len(y_val)))
        
            model, *_ = train_model(X_train, y_train, X_val, y_val, mock_config)
        
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test), "Predictions must match test size"
        assert predictions.shape[1:] == (1,) or predictions.ndim == 1
        assert hasattr(model, "predict") ,"It's a valid model type"

    def test_retrained_model_uses_more_data(
        self, e2e_processed_splits, mock_config
    ):
        """Verify retrained model can predict after using train+val."""
        train_path, val_path, test_path = e2e_processed_splits
        X_train, y_train, X_val, y_val, X_test, y_test = load_split_data(
            train_path, val_path, test_path
        )
        
        model = retrain_on_train_val(X_train, y_train, X_val, y_val, mock_config)
        probs = model.predict_proba(X_test)[:, 1]
        
        assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities must be in [0,1]"


@pytest.mark.integration
class TestFullPipelineE2E:
    """End-to-end tests for complete pipeline flow."""

    def test_saved_model_can_predict_on_new_data(
        self, e2e_processed_splits, mock_config, tmp_path
    ):
        """Verify saved model loads and predicts correctly."""
        train_path, val_path, test_path = e2e_processed_splits
        X_train, y_train, X_val, y_val, X_test, y_test = load_split_data(
            train_path, val_path, test_path
        )
        
        model = retrain_on_train_val(X_train, y_train, X_val, y_val, mock_config)
        model_path = tmp_path / "model.joblib"
        save_model(model, model_path)
        
        loaded = load_model(model_path)
        probs = predict_proba(loaded, X_test)
        
        assert len(probs) == len(X_test), "Loaded model must predict all samples"
        assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities must be valid"

    def test_pipeline_output_format_matches_serving_input(
        self, e2e_processed_splits, mock_config, tmp_path
    ):
        """Verify pipeline output is compatible with serving layer."""
        train_path, val_path, test_path = e2e_processed_splits
        X_train, y_train, X_val, y_val, X_test, y_test = load_split_data(
            train_path, val_path, test_path
        )
        
        model = retrain_on_train_val(X_train, y_train, X_val, y_val, mock_config)
        model_path = tmp_path / "model.joblib"
        save_model(model, model_path)
        
        loaded = joblib.load(model_path)
        assert hasattr(loaded, "predict"), "Model must have predict method"
        assert hasattr(loaded, "predict_proba"), "Model must have predict_proba method"
        assert hasattr(loaded, "named_steps"), "Model must be a pipeline"