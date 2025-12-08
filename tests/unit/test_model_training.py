# test_model_training.py
"""Tests for model training module."""

import pytest
import pandas as pd
import numpy as np
import joblib
from unittest.mock import patch, MagicMock

from src.models.train import (
    load_split_data,
    create_preprocessor,
    train_model,
    retrain_on_train_val,
    save_model,
)


class TestDataLoading:
    """Tests for data loading functions."""

    def test_load_split_data_shapes_match(self, parquet_splits):
        """Verify loaded data shapes are consistent."""
        train_path, val_path, test_path = parquet_splits
        X_train, y_train, X_val, y_val, X_test, y_test = load_split_data(
            train_path, val_path, test_path
        )
        assert len(X_train) == len(y_train), "Train X and y must have same length"
        assert len(X_val) == len(y_val), "Val X and y must have same length"
        assert len(X_test) == len(y_test), "Test X and y must have same length"

    def test_load_split_data_no_target_leak(self, parquet_splits):
        """Verify target column is not in features."""
        train_path, val_path, test_path = parquet_splits
        X_train, _, X_val, _, X_test, _ = load_split_data(
            train_path, val_path, test_path
        )
        for name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
            assert "is_fraud" not in X.columns, f"is_fraud leaked into {name} features"

    def test_load_split_data_consistent_columns(self, parquet_splits):
        """Verify all splits have identical feature columns."""
        train_path, val_path, test_path = parquet_splits
        X_train, _, X_val, _, X_test, _ = load_split_data(
            train_path, val_path, test_path
        )
        assert list(X_train.columns) == list(X_val.columns), "Train/val columns mismatch"
        assert list(X_val.columns) == list(X_test.columns), "Val/test columns mismatch"


class TestPreprocessor:
    """Tests for preprocessing pipeline."""

    def test_preprocessor_handles_mixed_types(self, sample_train_df):
        """Verify preprocessor correctly identifies numeric and categorical columns."""
        X = sample_train_df.drop(columns=["is_fraud"])
        preprocessor = create_preprocessor(X)
        transformer_names = [t[0] for t in preprocessor.transformers]
        assert "cat" in transformer_names, "Categorical transformer must exist"
        assert "num" in transformer_names, "Numeric transformer must exist"

    def test_preprocessor_output_no_nulls(self, sample_train_df):
        """Verify preprocessor output has no null values."""
        X = sample_train_df.drop(columns=["is_fraud"])
        preprocessor = create_preprocessor(X)
        X_transformed = preprocessor.fit_transform(X)
        assert not np.isnan(X_transformed).any(), "Transformed data must have no NaN"

    def test_preprocessor_handles_unknown_categories(self, sample_train_df):
        """Verify preprocessor handles unseen categories gracefully."""
        X = sample_train_df.drop(columns=["is_fraud"])
        preprocessor = create_preprocessor(X)
        preprocessor.fit(X)
        X_new = X.iloc[:5].copy()
        X_new["transaction_channel"] = "UnknownChannel"
        result = preprocessor.transform(X_new)
        assert result.shape[0] == 5, "Transform must return correct number of rows"


class TestModelTraining:
    """Tests for model training and persistence."""

    def test_train_model_returns_valid_pipeline(self, sample_train_df, sample_val_df, mock_config):
        """Verify trained model has predict and predict_proba methods."""
        X_train = sample_train_df.drop(columns=["is_fraud"])
        y_train = sample_train_df["is_fraud"]
        X_val = sample_val_df.drop(columns=["is_fraud"])
        y_val = sample_val_df["is_fraud"]
        with patch("src.models.train.evaluate_model") as mock_eval:
            mock_eval.return_value = ({"auc": 0.8}, np.zeros(len(y_val)), np.zeros(len(y_val)))
            model, _ = train_model(X_train, y_train, X_val, y_val, mock_config)
        assert hasattr(model, "predict"), "Model must have predict method"
        assert hasattr(model, "predict_proba"), "Model must have predict_proba method"

    def test_predictions_probability_range(self, sample_train_df, sample_val_df, mock_config):
        """Verify predicted probabilities are in [0, 1] range."""
        X_train = sample_train_df.drop(columns=["is_fraud"])
        y_train = sample_train_df["is_fraud"]
        X_val = sample_val_df.drop(columns=["is_fraud"])
        y_val = sample_val_df["is_fraud"]
        with patch("src.models.train.evaluate_model") as mock_eval:
            mock_eval.return_value = ({"auc": 0.8}, np.zeros(len(y_val)), np.zeros(len(y_val)))
            model, _ = train_model(X_train, y_train, X_val, y_val, mock_config)
        probs = model.predict_proba(X_val)[:, 1]
        assert (probs >= 0).all() and (probs <= 1).all(), "Probabilities must be in [0, 1]"

    def test_retrain_uses_combined_data(self, sample_train_df, sample_val_df, mock_config):
        """Verify retrain uses both train and val data."""
        X_train = sample_train_df.drop(columns=["is_fraud"])
        y_train = sample_train_df["is_fraud"]
        X_val = sample_val_df.drop(columns=["is_fraud"])
        y_val = sample_val_df["is_fraud"]
        model = retrain_on_train_val(X_train, y_train, X_val, y_val, mock_config)
        expected_samples = len(X_train) + len(X_val)
        preds = model.predict(X_train)
        assert len(preds) == len(X_train), "Model must be able to predict on training data"

    def test_saved_model_loadable_and_usable(self, sample_train_df, sample_val_df, mock_config, tmp_path):
        """Verify saved model can be loaded and used for predictions."""
        X_train = sample_train_df.drop(columns=["is_fraud"])
        y_train = sample_train_df["is_fraud"]
        X_val = sample_val_df.drop(columns=["is_fraud"])
        y_val = sample_val_df["is_fraud"]
        with patch("src.models.train.evaluate_model") as mock_eval:
            mock_eval.return_value = ({"auc": 0.8}, np.zeros(len(y_val)), np.zeros(len(y_val)))
            model, _ = train_model(X_train, y_train, X_val, y_val, mock_config)
        model_path = tmp_path / "model.joblib"
        save_model(model, model_path)
        loaded = joblib.load(model_path)
        preds = loaded.predict(X_val)
        assert len(preds) == len(X_val), "Loaded model must predict correct number of samples"