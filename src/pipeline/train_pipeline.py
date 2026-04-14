"""End-to-end training pipeline orchestration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from src.data.io import ensure_binary_target, load_dataset, split_features_target
from src.evaluation.metrics import evaluate_binary_classification, summarize_model_metrics
from src.features.preprocessing import build_preprocessing_pipeline
from src.models.trainer import select_best_model, train_candidate_models


ARTIFACT_MODEL = "model.pkl"
ARTIFACT_PREPROCESSOR = "preprocessing_pipeline.pkl"
ARTIFACT_METRICS = "metrics.json"


def train_pipeline(
    data_path: str | Path,
    target_column: str,
    artifacts_dir: str | Path = "models",
    test_size: float = 0.2,
    random_state: int = 42,
    selection_metric: str = "f1",
) -> dict[str, Any]:
    """Run full pipeline: data -> preprocessing -> training -> evaluation -> save artifacts."""
    df = load_dataset(data_path)
    x, y = split_features_target(df, target_column=target_column)
    ensure_binary_target(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor, numeric_cols, categorical_cols = build_preprocessing_pipeline(x_train)
    x_train_t = preprocessor.fit_transform(x_train)
    x_test_t = preprocessor.transform(x_test)

    trained_models = train_candidate_models(x_train_t, y_train, random_state=random_state)

    metrics_by_model: dict[str, dict[str, Any]] = {}
    for model_name, model in trained_models.items():
        y_prob = model.predict_proba(x_test_t)[:, 1]
        metrics_by_model[model_name] = evaluate_binary_classification(y_test, y_prob)

    summary = summarize_model_metrics(metrics_by_model)
    best_model_name = select_best_model(summary, selection_metric=selection_metric)
    best_model = trained_models[best_model_name]
    best_threshold = float(metrics_by_model[best_model_name]["threshold"])

    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, artifacts_path / ARTIFACT_MODEL)
    joblib.dump(preprocessor, artifacts_path / ARTIFACT_PREPROCESSOR)

    report = {
        "selected_model": best_model_name,
        "selection_metric": selection_metric,
        "selected_threshold": best_threshold,
        "features": {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
        },
        "metrics_by_model": metrics_by_model,
    }

    with (artifacts_path / ARTIFACT_METRICS).open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


def evaluate_saved_model(
    data_path: str | Path,
    target_column: str,
    artifacts_dir: str | Path = "models",
) -> dict[str, Any]:
    """Evaluate saved model artifacts on a labeled dataset."""
    artifacts_path = Path(artifacts_dir)
    model = joblib.load(artifacts_path / ARTIFACT_MODEL)
    preprocessor = joblib.load(artifacts_path / ARTIFACT_PREPROCESSOR)

    with (artifacts_path / ARTIFACT_METRICS).open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    threshold = float(metadata.get("selected_threshold", 0.5))

    df = load_dataset(data_path)
    x, y = split_features_target(df, target_column=target_column)
    ensure_binary_target(y)

    x_t = preprocessor.transform(x)
    y_prob = model.predict_proba(x_t)[:, 1]
    return evaluate_binary_classification(y, y_prob, threshold=threshold)


def _to_numpy_row(transformed) -> np.ndarray:
    """Convert transformed matrix row to dense ndarray."""
    if hasattr(transformed, "toarray"):
        return transformed.toarray()[0]
    return np.asarray(transformed)[0]


def explain_with_artifacts(
    transaction: dict[str, Any],
    artifacts_dir: str | Path = "models",
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Explain one prediction using trained model and fitted preprocessor artifacts."""
    artifacts_path = Path(artifacts_dir)
    model = joblib.load(artifacts_path / ARTIFACT_MODEL)
    preprocessor = joblib.load(artifacts_path / ARTIFACT_PREPROCESSOR)

    import pandas as pd

    input_df = pd.DataFrame([transaction])
    transformed = preprocessor.transform(input_df)
    x_row = _to_numpy_row(transformed)

    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [f"feature_{i}" for i in range(len(x_row))]

    if hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0] * x_row)
    elif hasattr(model, "feature_importances_"):
        importance = np.abs(model.feature_importances_ * x_row)
    else:
        importance = np.abs(x_row)

    order = np.argsort(importance)[::-1][:top_n]
    explanation = [
        {
            "feature": feature_names[int(idx)],
            "transformed_value": float(x_row[int(idx)]),
            "importance": float(importance[int(idx)]),
        }
        for idx in order
    ]
    return explanation
