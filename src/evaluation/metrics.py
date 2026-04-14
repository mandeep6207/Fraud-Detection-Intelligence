"""Fraud model evaluation metrics and threshold tuning."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def tune_threshold(y_true, y_prob: np.ndarray) -> float:
    """Find threshold that maximizes F1 score on validation data."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    if len(thresholds) == 0:
        return 0.5

    f1_scores = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    best_index = int(np.argmax(f1_scores))
    return float(thresholds[best_index])


def evaluate_binary_classification(
    y_true,
    y_prob: np.ndarray,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Evaluate binary classifier with ranking and threshold metrics."""
    tuned_threshold = tune_threshold(y_true, y_prob) if threshold is None else float(threshold)
    y_pred = (y_prob >= tuned_threshold).astype(int)

    metrics = {
        "threshold": tuned_threshold,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    return metrics


def summarize_model_metrics(metrics_by_model: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Extract numeric model-level metrics for model selection."""
    summary: dict[str, dict[str, float]] = {}
    for model_name, metrics in metrics_by_model.items():
        summary[model_name] = {
            "f1": float(metrics["f1"]),
            "roc_auc": float(metrics["roc_auc"]),
        }
    return summary
