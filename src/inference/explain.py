"""Prediction explanation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.pipeline.train_pipeline import explain_with_artifacts


def explain_prediction(
    transaction: dict[str, Any],
    artifacts_dir: str | Path = "models",
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Explain actual transaction prediction using saved artifacts."""
    return explain_with_artifacts(
        transaction=transaction,
        artifacts_dir=artifacts_dir,
        top_n=top_n,
    )
