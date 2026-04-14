"""Evaluation helpers for fraud detection classifiers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


MetricResult = dict[str, float | np.ndarray]


def _metric_average(y_true: pd.Series) -> str:
    """Use binary metrics for binary tasks and weighted for multiclass tasks."""
    return "binary" if y_true.nunique(dropna=False) == 2 else "weighted"


def evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series) -> MetricResult:
    """Evaluate a trained model and return key classification metrics."""
    y_pred = model.predict(x_test)
    average = _metric_average(y_test)

    metrics: MetricResult = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average=average, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
    return metrics


def print_evaluation(name: str, metrics: MetricResult) -> None:
    """Pretty-print model metrics and confusion matrix."""
    print(f"\n=== {name} ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1 Score : {metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
