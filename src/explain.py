"""Model explanation utilities for fraud detection classifiers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from predict import load_model


def get_feature_names(model, x_sample: pd.DataFrame) -> list[str]:
    """Extract transformed feature names from the preprocessing pipeline."""
    preprocessor = model.named_steps["preprocessing"]
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        transformed = preprocessor.transform(x_sample)
        return [f"feature_{i}" for i in range(transformed.shape[1])]


def feature_importance(model, x_sample: pd.DataFrame, top_n: int = 15) -> pd.DataFrame:
    """Return top feature importances for random forest or logistic regression."""
    model_step = model.named_steps["classifier"]
    names = get_feature_names(model, x_sample)

    if hasattr(model_step, "feature_importances_"):
        importance_values = model_step.feature_importances_
    elif hasattr(model_step, "coef_"):
        importance_values = np.abs(model_step.coef_[0])
    else:
        raise ValueError("Model does not expose feature importances or coefficients.")

    importance_df = pd.DataFrame(
        {
            "feature": names,
            "importance": importance_values,
        }
    ).sort_values(by="importance", ascending=False)

    return importance_df.head(top_n)


def explain_single_prediction(model, x_single: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Approximate contribution explanation for a single prediction."""
    preprocessor = model.named_steps["preprocessing"]
    classifier = model.named_steps["classifier"]

    transformed = preprocessor.transform(x_single)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed_vector = transformed[0]
    feature_names = get_feature_names(model, x_single)

    if hasattr(classifier, "coef_"):
        contribution = transformed_vector * classifier.coef_[0]
    elif hasattr(classifier, "feature_importances_"):
        contribution = transformed_vector * classifier.feature_importances_
    else:
        raise ValueError("Model does not support this explanation method.")

    explanation_df = pd.DataFrame(
        {
            "feature": feature_names,
            "contribution": contribution,
            "abs_contribution": np.abs(contribution),
        }
    ).sort_values(by="abs_contribution", ascending=False)

    return explanation_df.head(top_n)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for explanation workflow."""
    parser = argparse.ArgumentParser(description="Explain fraud model behavior.")
    parser.add_argument("--model-path", required=True, help="Path to saved model (.joblib)")
    parser.add_argument("--input-path", required=True, help="CSV path for sample input data")
    parser.add_argument("--top-n", type=int, default=10, help="Top features to display")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for feature importance and prediction explanation."""
    args = parse_args()
    model = load_model(Path(args.model_path))
    sample_df = pd.read_csv(args.input_path)

    if sample_df.empty:
        raise ValueError("Input sample file is empty.")

    print("=== Feature Importance ===")
    print(feature_importance(model, sample_df.head(1), top_n=args.top_n))

    print("\n=== Single Prediction Explanation ===")
    print(explain_single_prediction(model, sample_df.head(1), top_n=args.top_n))


if __name__ == "__main__":
    main()
