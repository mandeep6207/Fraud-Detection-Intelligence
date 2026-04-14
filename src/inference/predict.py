"""Prediction utilities using persisted artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.pipeline.train_pipeline import ARTIFACT_METRICS, ARTIFACT_MODEL, ARTIFACT_PREPROCESSOR


def load_artifacts(artifacts_dir: str | Path = "models") -> tuple[Any, Any, dict[str, Any]]:
    """Load model, preprocessor, and metadata artifacts."""
    artifacts_path = Path(artifacts_dir)
    model_path = artifacts_path / ARTIFACT_MODEL
    preprocessor_path = artifacts_path / ARTIFACT_PREPROCESSOR
    metrics_path = artifacts_path / ARTIFACT_METRICS

    if not model_path.exists() or not preprocessor_path.exists() or not metrics_path.exists():
        raise FileNotFoundError("Missing model artifacts. Train the pipeline first.")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    with metrics_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, preprocessor, metadata


def predict_transaction(transaction: dict[str, Any], artifacts_dir: str | Path = "models") -> dict[str, Any]:
    """Predict fraud probability and label for one transaction."""
    model, preprocessor, metadata = load_artifacts(artifacts_dir=artifacts_dir)
    threshold = float(metadata.get("selected_threshold", 0.5))

    input_df = pd.DataFrame([transaction])
    transformed = preprocessor.transform(input_df)
    fraud_probability = float(model.predict_proba(transformed)[:, 1][0])
    label = int(fraud_probability >= threshold)

    return {
        "fraud_probability": fraud_probability,
        "label": label,
        "threshold": threshold,
        "model": metadata.get("selected_model"),
    }
