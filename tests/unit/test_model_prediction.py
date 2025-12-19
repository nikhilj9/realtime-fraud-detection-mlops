# tests/unit/test_model_prediction.py
"""Tests for model prediction utilities."""

import pytest
import pandas as pd
import numpy as np

from src.models.predict import (
    load_model,
    predict,
    predict_proba,
    predict_with_threshold,
    batch_predict,
)
from src.utils.exceptions import ModelPredictionError


class TestModelLoading:
    """Tests for model loading functionality."""

    def test_load_model_missing_raises_error(self, tmp_path):
        """Verify missing model path raises ModelPredictionError."""
        fake_path = tmp_path / "nonexistent.joblib"
        with pytest.raises(ModelPredictionError, match="Model not found"):
            load_model(fake_path)

    def test_load_model_returns_usable_model(self, saved_model_path, numeric_feature_df):
        """Verify loaded model has required prediction methods."""
        model = load_model(saved_model_path)
        assert hasattr(model, "predict"), "Loaded model must have predict method"
        assert hasattr(model, "predict_proba"), "Loaded model must have predict_proba method"
        preds = model.predict(numeric_feature_df)
        assert len(preds) == len(numeric_feature_df), "Predictions must match input length"


class TestPrediction:
    """Tests for prediction functions."""

    def test_predict_returns_correct_shape(self, trained_sklearn_model, numeric_feature_df):
        """Verify predict returns array with correct length."""
        preds = predict(trained_sklearn_model, numeric_feature_df)
        assert len(preds) == len(numeric_feature_df), "Prediction length must match input"

    def test_predict_proba_in_valid_range(self, trained_sklearn_model, numeric_feature_df):
        """Verify probabilities are in [0, 1] range."""
        probas = predict_proba(trained_sklearn_model, numeric_feature_df)
        assert (probas >= 0).all() and (probas <= 1).all(), "Probabilities must be in [0, 1]"

    def test_predict_proba_raises_on_invalid_model(self, numeric_feature_df):
        """Verify ModelPredictionError raised for broken model."""
        broken_model = object()
        with pytest.raises(ModelPredictionError, match="Prediction failed"):
            predict_proba(broken_model, numeric_feature_df)

    def test_predict_with_threshold_returns_binary(self, trained_sklearn_model, numeric_feature_df):
        """Verify threshold prediction returns only 0 or 1."""
        preds = predict_with_threshold(trained_sklearn_model, numeric_feature_df, threshold=0.5)
        assert set(np.unique(preds)).issubset({0, 1}), "Predictions must be binary"

    def test_predict_with_threshold_respects_boundary(self, trained_sklearn_model, numeric_feature_df):
        """Verify threshold=0 predicts all fraud, threshold=1 predicts none."""
        preds_low = predict_with_threshold(trained_sklearn_model, numeric_feature_df, threshold=0.0)
        preds_high = predict_with_threshold(trained_sklearn_model, numeric_feature_df, threshold=1.0)
        assert preds_low.sum() == len(preds_low), "Threshold=0 should predict all positive"
        assert preds_high.sum() == 0, "Threshold=1 should predict all negative"


class TestBatchPrediction:
    """Tests for batch prediction functionality."""

    def test_batch_predict_creates_output_file(
        self, trained_sklearn_model, prediction_input_parquet, tmp_path
    ):
        """Verify batch predict creates output parquet file."""
        output_path = tmp_path / "output" / "predictions.parquet"
        batch_predict(trained_sklearn_model, prediction_input_parquet, output_path)
        assert output_path.exists(), "Output file must be created"

    def test_batch_predict_adds_required_columns(
        self, trained_sklearn_model, prediction_input_parquet, tmp_path
    ):
        """Verify output contains probability and prediction columns."""
        output_path = tmp_path / "predictions.parquet"
        batch_predict(trained_sklearn_model, prediction_input_parquet, output_path)
        result = pd.read_parquet(output_path)
        assert "fraud_probability" in result.columns, "Must contain fraud_probability"
        assert "is_fraud_predicted" in result.columns, "Must contain is_fraud_predicted"

    def test_batch_predict_no_data_loss(
        self, trained_sklearn_model, prediction_input_parquet, tmp_path
    ):
        """Verify row count preserved after batch prediction."""
        input_df = pd.read_parquet(prediction_input_parquet)
        output_path = tmp_path / "predictions.parquet"
        batch_predict(trained_sklearn_model, prediction_input_parquet, output_path)
        result = pd.read_parquet(output_path)
        assert len(result) == len(input_df), "Row count must be preserved"

    def test_batch_predict_removes_target_column(
        self, trained_sklearn_model, prediction_input_parquet, tmp_path
    ):
        """Verify is_fraud column is dropped to prevent leakage."""
        output_path = tmp_path / "predictions.parquet"
        batch_predict(trained_sklearn_model, prediction_input_parquet, output_path)
        result = pd.read_parquet(output_path)
        assert "is_fraud" not in result.columns, "Original target must be removed"