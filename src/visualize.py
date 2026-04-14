"""Visualization utilities for fraud detection datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data_loader import load_csv_data


def plot_fraud_distribution(
    df: pd.DataFrame,
    target_column: str,
    output_path: str | Path | None = None,
):
    """Plot target class distribution for fraud vs non-fraud."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found.")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.countplot(x=target_column, data=df, palette="Set2", ax=ax)
    ax.set_title("Fraud Distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)

    return fig


def plot_correlation_heatmap(
    df: pd.DataFrame,
    output_path: str | Path | None = None,
):
    """Plot a correlation heatmap for numeric columns."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.empty:
        raise ValueError("No numeric columns available for correlation heatmap.")

    corr = numeric_df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Heatmap")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150)

    return fig


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for visualizations."""
    parser = argparse.ArgumentParser(description="Create fraud detection plots.")
    parser.add_argument("--data-path", required=True, help="Path to dataset CSV")
    parser.add_argument("--target-column", required=True, help="Target column name")
    parser.add_argument(
        "--output-dir",
        default="reports",
        help="Directory to save generated plot images",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for fraud plots."""
    args = parse_args()
    dataset = load_csv_data(args.data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_fraud_distribution(
        dataset,
        target_column=args.target_column,
        output_path=output_dir / "fraud_distribution.png",
    )
    plot_correlation_heatmap(
        dataset,
        output_path=output_dir / "correlation_heatmap.png",
    )
    print(f"Saved plots in: {output_dir}")


if __name__ == "__main__":
    main()
