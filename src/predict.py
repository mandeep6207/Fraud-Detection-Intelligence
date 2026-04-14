"""Inference utilities and CLI for fraud prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from utils import load_joblib_object, transaction_to_dataframe


PredictionOutput = dict[str, Any]


def load_model_for_inference(model_path: str | Path):
    """Load a saved model pipeline for inference."""
    return load_joblib_object(model_path)


def predict_transaction(model, transaction: dict[str, Any]) -> PredictionOutput:
    """Predict fraud label and return fraud probability score for one transaction."""
    input_df = transaction_to_dataframe(transaction)
    predicted_label = model.predict(input_df)[0]

    fraud_probability: float | None = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.classes_)

        if 1 in classes:
            fraud_idx = classes.index(1)
            fraud_probability = float(probabilities[fraud_idx])
        elif "fraud" in classes:
            fraud_idx = classes.index("fraud")
            fraud_probability = float(probabilities[fraud_idx])
        else:
            fraud_probability = float(max(probabilities))

    return {
        "prediction": int(predicted_label) if isinstance(predicted_label, (int, bool)) else predicted_label,
        "fraud_probability": fraud_probability,
    }


def parse_args() -> argparse.Namespace:
    """Parse prediction CLI arguments."""
    parser = argparse.ArgumentParser(description="Run fraud prediction for one transaction.")
    parser.add_argument("--model-path", required=True, help="Path to saved .joblib model")
    parser.add_argument(
        "--transaction-json",
        required=True,
        help="Transaction as JSON string, e.g. '{\"amount\": 120.5, \"merchant\": \"A\"}'",
    )
    return parser.parse_args()


def main() -> None:
    """Load model and score one transaction from JSON input."""
    args = parse_args()
    model = load_model_for_inference(args.model_path)
    transaction = json.loads(args.transaction_json)
    result = predict_transaction(model, transaction)

    print("Prediction result:")
    print(result)


if __name__ == "__main__":
    main()
