"""Train fraud detection models and save them as joblib artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from evaluate import print_evaluation
from train import train_models
from utils import save_joblib_object


def train_and_save_models(
    data_path: str | Path,
    target_column: str,
    output_dir: str | Path = "models/artifacts",
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Path]:
    """Train models and save each trained pipeline to disk."""
    trained_models, results = train_models(
        data_path=data_path,
        target_column=target_column,
        test_size=test_size,
        random_state=random_state,
    )

    output_base = Path(output_dir)
    saved_paths: dict[str, Path] = {}

    for model_name, model in trained_models.items():
        model_path = output_base / f"{model_name}.joblib"
        saved_paths[model_name] = save_joblib_object(model, model_path)

    for model_name, metrics in results.items():
        print_evaluation(model_name, metrics)

    return saved_paths


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training and model persistence."""
    parser = argparse.ArgumentParser(description="Train and save fraud detection models.")
    parser.add_argument("--data-path", required=True, help="Path to training CSV")
    parser.add_argument("--target-column", required=True, help="Target column name")
    parser.add_argument(
        "--output-dir",
        default="models/artifacts",
        help="Directory where model files will be saved",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    """Run training and save all model artifacts."""
    args = parse_args()
    saved_paths = train_and_save_models(
        data_path=args.data_path,
        target_column=args.target_column,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print("\nSaved model artifacts:")
    for name, path in saved_paths.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
