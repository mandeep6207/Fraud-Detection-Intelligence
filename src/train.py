"""Training entry point for fraud detection models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from evaluate import evaluate_model, print_evaluation
from model import build_model_pipelines
from preprocessing import build_preprocessor, split_features_target, split_train_test


TrainOutput = tuple[dict[str, Any], dict[str, Any]]


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """Load dataset from a CSV path."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path)


def train_models(
    data_path: str | Path,
    target_column: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainOutput:
    """Train and evaluate logistic regression and random forest models."""
    df = load_dataset(data_path)
    x, y = split_features_target(df, target_column=target_column)
    x_train, x_test, y_train, y_test = split_train_test(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(x_train)
    print(f"Using {len(numeric_cols)} numeric columns and {len(categorical_cols)} categorical columns.")

    pipelines = build_model_pipelines(preprocessor=preprocessor, random_state=random_state)

    trained_models: dict[str, Any] = {}
    results: dict[str, Any] = {}

    for name, pipeline in pipelines.items():
        pipeline.fit(x_train, y_train)
        trained_models[name] = pipeline
        results[name] = evaluate_model(pipeline, x_test, y_test)

    return trained_models, results


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train fraud detection models.")
    parser.add_argument("--data-path", required=True, help="Path to training CSV")
    parser.add_argument("--target-column", required=True, help="Target column name")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    """Run model training and print evaluation metrics."""
    args = parse_args()

    _, results = train_models(
        data_path=args.data_path,
        target_column=args.target_column,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    for model_name, metrics in results.items():
        print_evaluation(model_name, metrics)


if __name__ == "__main__":
    main()
