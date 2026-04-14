"""Model training script for fraud detection project."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_loader import load_csv_data
from evaluate import evaluate_classification_model, print_metrics
from preprocessing import build_preprocessing_pipeline, split_features_target


def build_models(random_state: int = 42) -> dict[str, Any]:
    """Create model instances for fraud detection."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
    }


def train_models(
    data_path: str | Path,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[dict[str, Pipeline], dict[str, dict[str, Any]]]:
    """Train logistic regression and random forest models with preprocessing."""
    df = load_csv_data(data_path)
    x, y = split_features_target(df, target_column)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor, _, _ = build_preprocessing_pipeline(x_train)
    models = build_models(random_state=random_state)

    trained_pipelines: dict[str, Pipeline] = {}
    metrics_results: dict[str, dict[str, Any]] = {}

    for model_name, estimator in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessing", preprocessor),
                ("classifier", estimator),
            ]
        )
        pipeline.fit(x_train, y_train)

        predictions = pipeline.predict(x_test)
        metrics = evaluate_classification_model(y_test, predictions)

        trained_pipelines[model_name] = pipeline
        metrics_results[model_name] = metrics

    return trained_pipelines, metrics_results


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train fraud detection models.")
    parser.add_argument("--data-path", required=True, help="Path to CSV dataset")
    parser.add_argument("--target-column", required=True, help="Target column name")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for training."""
    args = parse_args()
    _, metrics_results = train_models(
        data_path=args.data_path,
        target_column=args.target_column,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    for model_name, metrics in metrics_results.items():
        print_metrics(model_name, metrics)


if __name__ == "__main__":
    main()
