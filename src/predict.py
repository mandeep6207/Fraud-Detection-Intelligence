"""Inference utilities for fraud detection models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def load_model(model_path: str | Path):
    """Load model from joblib file."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def predict_new_data(model, input_df: pd.DataFrame) -> dict[str, Any]:
    """Predict class and fraud probability for input dataframe."""
    predictions = model.predict(input_df)
    result: dict[str, Any] = {"predictions": predictions.tolist()}

    if hasattr(model, "predict_proba"):
        prob_matrix = model.predict_proba(input_df)
        classes = list(model.classes_)

        if 1 in classes:
            fraud_idx = classes.index(1)
        elif "fraud" in classes:
            fraud_idx = classes.index("fraud")
        else:
            fraud_idx = int(prob_matrix.shape[1] - 1)

        result["fraud_probability"] = prob_matrix[:, fraud_idx].tolist()

    return result


def predict_from_dict(model_path: str | Path, transaction: dict[str, Any]) -> dict[str, Any]:
    """Load model and predict fraud on one transaction dictionary."""
    model = load_model(model_path)
    input_df = pd.DataFrame([transaction])
    output = predict_new_data(model, input_df)
    return {
        "prediction": output["predictions"][0],
        "fraud_probability": output.get("fraud_probability", [None])[0],
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI args for prediction workflow."""
    parser = argparse.ArgumentParser(description="Predict fraud for new transaction data.")
    parser.add_argument("--model-path", required=True, help="Path to joblib model")
    parser.add_argument(
        "--input-json",
        required=True,
        help="Input transaction as JSON string",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for single transaction prediction."""
    args = parse_args()
    transaction = json.loads(args.input_json)
    prediction = predict_from_dict(args.model_path, transaction)
    print(prediction)


if __name__ == "__main__":
    main()
