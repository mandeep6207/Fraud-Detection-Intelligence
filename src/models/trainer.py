"""Model training and model selection helpers."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def build_candidate_models(random_state: int = 42) -> dict[str, Any]:
    """Create candidate classifiers for fraud detection."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def train_candidate_models(
    x_train_transformed,
    y_train,
    random_state: int = 42,
) -> dict[str, Any]:
    """Fit all candidate models on preprocessed training data."""
    trained: dict[str, Any] = {}
    for name, model in build_candidate_models(random_state=random_state).items():
        model.fit(x_train_transformed, y_train)
        trained[name] = model
    return trained


def select_best_model(
    metrics_by_model: dict[str, dict[str, float]],
    selection_metric: str = "f1",
) -> str:
    """Select best model based on configured metric."""
    valid_metrics = {"f1", "roc_auc"}
    if selection_metric not in valid_metrics:
        raise ValueError(f"selection_metric must be one of {valid_metrics}")

    best_name = max(
        metrics_by_model,
        key=lambda model_name: metrics_by_model[model_name][selection_metric],
    )
    return best_name
