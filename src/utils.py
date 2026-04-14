"""Shared utility helpers for model persistence and inference input handling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def ensure_parent_dir(file_path: str | Path) -> Path:
    """Ensure parent directory exists and return normalized Path."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_joblib_object(obj: Any, file_path: str | Path) -> Path:
    """Persist an object to disk with joblib."""
    path = ensure_parent_dir(file_path)
    joblib.dump(obj, path)
    return path


def load_joblib_object(file_path: str | Path) -> Any:
    """Load an object from a joblib file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def transaction_to_dataframe(transaction: dict[str, Any]) -> pd.DataFrame:
    """Convert one transaction dictionary into a single-row DataFrame."""
    if not transaction:
        raise ValueError("Transaction input cannot be empty.")
    return pd.DataFrame([transaction])
