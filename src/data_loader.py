"""Data loading utilities for the fraud detection project."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_csv_data(file_path: str | Path) -> pd.DataFrame:
    """Load CSV data into a pandas DataFrame."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def preview_data(df: pd.DataFrame, rows: int = 5) -> None:
    """Print the first N rows of a DataFrame."""
    print(df.head(rows))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Load and preview a fraud dataset.")
    parser.add_argument("--data-path", required=True, help="Path to CSV dataset")
    parser.add_argument("--rows", type=int, default=5, help="Rows to preview")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for loading and previewing dataset."""
    args = parse_args()
    dataset = load_csv_data(args.data_path)
    print(f"Loaded dataset with shape: {dataset.shape}")
    preview_data(dataset, rows=args.rows)


if __name__ == "__main__":
    main()
