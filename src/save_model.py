"""Model persistence utilities for fraud detection."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib

from train import train_models


def save_model(model: Any, file_path: str | Path) -> Path:
    """Save trained model to disk using joblib."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def train_and_save(
    data_path: str | Path,
    target_column: str,
    output_dir: str | Path = "models/artifacts",
) -> dict[str, Path]:
    """Train project models and save each one as a joblib artifact."""
    trained_models, _ = train_models(
        data_path=data_path,
        target_column=target_column,
    )

    output_dir = Path(output_dir)
    saved_paths: dict[str, Path] = {}

    for model_name, model in trained_models.items():
        file_path = output_dir / f"{model_name}.joblib"
        saved_paths[model_name] = save_model(model, file_path)

    return saved_paths


def parse_args() -> argparse.Namespace:
    """Parse CLI args for model save workflow."""
    parser = argparse.ArgumentParser(description="Train and save fraud detection models.")
    parser.add_argument("--data-path", required=True, help="Path to CSV dataset")
    parser.add_argument("--target-column", required=True, help="Target column name")
    parser.add_argument("--output-dir", default="models/artifacts", help="Directory to save models")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for train-and-save workflow."""
    args = parse_args()
    saved = train_and_save(
        data_path=args.data_path,
        target_column=args.target_column,
        output_dir=args.output_dir,
    )
    print("Saved models:")
    for name, path in saved.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
