"""Dataset I/O and validation utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_dataset(data_path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset from disk."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def preview_dataset(df: pd.DataFrame, rows: int = 5) -> pd.DataFrame:
    """Return a preview of the dataset."""
    return df.head(rows)


def split_features_target(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features and target."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' does not exist.")
    return df.drop(columns=[target_column]), df[target_column]


def ensure_binary_target(y: pd.Series) -> None:
    """Ensure fraud target is binary for this project."""
    if y.nunique(dropna=False) != 2:
        raise ValueError("Target must be binary for this fraud detection pipeline.")
