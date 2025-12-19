# tests/unit/test_model_evaluation.py
"""Tests for model evaluation utilities."""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.models.evaluate import (
    calculate_metrics,
    evaluate_model,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_feature_importance,
)


class TestCalculateMetrics:
    """Tests for metric calculation functions."""

    def test_metrics_in_valid_range(self, binary_classification_data):
        """Verify all metrics are within [0, 1] domain."""
        y_true, y_pred, y_prob = binary_classification_data
        metrics = calculate_metrics(y_true, y_pred, y_prob)
        for name, value in metrics.items():
            assert 0 <= value <= 1, f"{name} must be in [0, 1], got {value}"

    def test_metrics_returns_required_keys(self, binary_classification_data):
        """Verify all expected metric keys are present."""
        y_true, y_pred, y_prob = binary_classification_data
        metrics = calculate_metrics(y_true, y_pred, y_prob)
        required = {"precision", "recall", "f1_score", "roc_auc", "pr_auc"}
        assert required.issubset(metrics.keys()), f"Missing keys: {required - metrics.keys()}"

    def test_metrics_handles_all_negative(self, all_negative_data):
        """Verify metrics handle edge case with no positive samples gracefully."""
        y_true, y_pred, y_prob = all_negative_data
        metrics = calculate_metrics(y_true, y_pred, y_prob)
        assert metrics["precision"] == 0, "Precision should be 0 with zero_division=0"
        assert metrics["recall"] == 0, "Recall should be 0 with no positives predicted"


class TestEvaluateModel:
    """Tests for model evaluation function."""

    def test_evaluate_returns_correct_shapes(self, trained_sklearn_model, numeric_feature_df):
        """Verify evaluate_model returns arrays matching input length."""
        y = pd.Series(np.random.choice([0, 1], len(numeric_feature_df), p=[0.9, 0.1]))
        metrics, y_pred, y_prob = evaluate_model(trained_sklearn_model, numeric_feature_df, y)
        assert len(y_pred) == len(numeric_feature_df), "y_pred length must match input"
        assert len(y_prob) == len(numeric_feature_df), "y_prob length must match input"

    def test_evaluate_predictions_are_binary(self, trained_sklearn_model, numeric_feature_df):
        """Verify predictions are binary values."""
        y = pd.Series(np.random.choice([0, 1], len(numeric_feature_df), p=[0.9, 0.1]))
        _, y_pred, _ = evaluate_model(trained_sklearn_model, numeric_feature_df, y)
        assert set(np.unique(y_pred)).issubset({0, 1}), "Predictions must be binary"

    def test_evaluate_probabilities_in_range(self, trained_sklearn_model, numeric_feature_df):
        """Verify probabilities are in [0, 1] range."""
        y = pd.Series(np.random.choice([0, 1], len(numeric_feature_df), p=[0.9, 0.1]))
        _, _, y_prob = evaluate_model(trained_sklearn_model, numeric_feature_df, y)
        assert (y_prob >= 0).all() and (y_prob <= 1).all(), "Probabilities must be in [0, 1]"


class TestPlotFunctions:
    """Tests for visualization functions."""

    def test_confusion_matrix_returns_figure(self, binary_classification_data):
        """Verify plot_confusion_matrix returns matplotlib Figure."""
        y_true, y_pred, _ = binary_classification_data
        fig = plot_confusion_matrix(y_true, y_pred)
        assert isinstance(fig, plt.Figure), "Must return matplotlib Figure"
        plt.close(fig)

    def test_confusion_matrix_saves_to_path(self, binary_classification_data, tmp_path):
        """Verify confusion matrix can be saved and file exists."""
        y_true, y_pred, _ = binary_classification_data
        save_path = tmp_path / "cm.png"
        plot_confusion_matrix(y_true, y_pred, save_path=save_path)
        assert save_path.exists(), "Saved figure file must exist"
        assert save_path.stat().st_size > 0, "Saved file must not be empty"

    def test_pr_curve_returns_figure(self, binary_classification_data):
        """Verify plot_precision_recall_curve returns matplotlib Figure."""
        y_true, _, y_prob = binary_classification_data
        fig = plot_precision_recall_curve(y_true, y_prob)
        assert isinstance(fig, plt.Figure), "Must return matplotlib Figure"
        plt.close(fig)

    def test_feature_importance_with_valid_model(
        self, model_with_feature_importance, numeric_feature_df
    ):
        """Verify feature importance plot works with compatible model."""
        feature_names = numeric_feature_df.columns.tolist()
        fig = plot_feature_importance(model_with_feature_importance, feature_names)
        assert isinstance(fig, plt.Figure), "Must return matplotlib Figure"
        plt.close(fig)

    def test_feature_importance_returns_none_for_incompatible(
        self, model_without_feature_importance, numeric_feature_df
    ):
        """Verify feature importance returns None for models without importances."""
        feature_names = numeric_feature_df.columns.tolist()
        result = plot_feature_importance(model_without_feature_importance, feature_names)
        assert result is None, "Must return None for incompatible model"