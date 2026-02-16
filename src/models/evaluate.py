"""Model evaluation utilities."""

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.utils import get_logger

logger = get_logger(__name__)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Calculate all evaluation metrics."""
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob)
    }


def evaluate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    dataset_name: str = "dataset"
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate model and return metrics."""

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    metrics = calculate_metrics(y.values, y_pred, y_prob)

    logger.info(f"{dataset_name} metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    return metrics, y_pred, y_prob


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: Path = None
) -> plt.Figure:
    """Plot confusion matrix."""

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    title: str = "Precision-Recall Curve",
    save_path: Path = None
) -> plt.Figure:
    """Plot precision-recall curve."""

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, label=f"PR-AUC: {pr_auc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_feature_importance(
    model: Any,
    feature_names: list,
    top_n: int = 15,
    title: str = "Feature Importance",
    save_path: Path = None
) -> plt.Figure:
    """Plot feature importance."""

    if hasattr(model, "named_steps"):
        classifier = model.named_steps.get("classifier")
    else:
        classifier = model

    if not hasattr(classifier, "feature_importances_"):
        logger.warning("Model doesn't have feature_importances_")
        return None

    importances = classifier.feature_importances_

    if len(importances) != len(feature_names):
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    feat_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(data=feat_df, x="importance", y="feature", palette="viridis", ax=ax)
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    return fig
