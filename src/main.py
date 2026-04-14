"""Single entrypoint for training, evaluation, prediction, and explanation."""

from __future__ import annotations

import argparse
import json

from src.inference.explain import explain_prediction
from src.inference.predict import predict_transaction
from src.pipeline.train_pipeline import evaluate_saved_model, train_pipeline


def parse_args() -> argparse.Namespace:
    """Create CLI for end-to-end project workflows."""
    parser = argparse.ArgumentParser(description="Fraud detection project CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and save model artifacts")
    train_parser.add_argument("--data-path", required=True)
    train_parser.add_argument("--target-column", required=True)
    train_parser.add_argument("--artifacts-dir", default="models")
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.add_argument("--selection-metric", choices=["f1", "roc_auc"], default="f1")

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate saved artifacts")
    eval_parser.add_argument("--data-path", required=True)
    eval_parser.add_argument("--target-column", required=True)
    eval_parser.add_argument("--artifacts-dir", default="models")

    predict_parser = subparsers.add_parser("predict", help="Predict fraud for one transaction")
    predict_parser.add_argument("--input-json", required=True)
    predict_parser.add_argument("--artifacts-dir", default="models")

    explain_parser = subparsers.add_parser("explain", help="Explain one prediction")
    explain_parser.add_argument("--input-json", required=True)
    explain_parser.add_argument("--artifacts-dir", default="models")
    explain_parser.add_argument("--top-n", type=int, default=10)

    return parser.parse_args()


def main() -> None:
    """Run selected command."""
    args = parse_args()

    if args.command == "train":
        report = train_pipeline(
            data_path=args.data_path,
            target_column=args.target_column,
            artifacts_dir=args.artifacts_dir,
            test_size=args.test_size,
            random_state=args.random_state,
            selection_metric=args.selection_metric,
        )
        print(json.dumps(report, indent=2))
        return

    if args.command == "evaluate":
        metrics = evaluate_saved_model(
            data_path=args.data_path,
            target_column=args.target_column,
            artifacts_dir=args.artifacts_dir,
        )
        print(json.dumps(metrics, indent=2))
        return

    if args.command == "predict":
        transaction = json.loads(args.input_json)
        output = predict_transaction(transaction, artifacts_dir=args.artifacts_dir)
        print(json.dumps(output, indent=2))
        return

    if args.command == "explain":
        transaction = json.loads(args.input_json)
        output = explain_prediction(
            transaction=transaction,
            artifacts_dir=args.artifacts_dir,
            top_n=args.top_n,
        )
        print(json.dumps(output, indent=2))
        return


if __name__ == "__main__":
    main()
