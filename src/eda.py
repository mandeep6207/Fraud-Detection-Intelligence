"""Exploratory Data Analysis utilities for fraud detection datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_loader import load_csv_data


def dataset_info(df: pd.DataFrame) -> dict[str, object]:
    """Return dataset shape, columns, and data types."""
    return {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
    }


def missing_values_report(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing value counts and percentages per column."""
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100
    report = pd.DataFrame(
        {
            "missing_count": missing_count,
            "missing_percentage": missing_pct.round(2),
        }
    )
    return report.sort_values(by="missing_count", ascending=False)


def basic_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for all columns."""
    return df.describe(include="all", datetime_is_numeric=True).transpose()


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for EDA run."""
    parser = argparse.ArgumentParser(description="Run basic EDA for fraud dataset.")
    parser.add_argument("--data-path", required=True, help="Path to dataset CSV")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for EDA summary."""
    args = parse_args()
    dataset = load_csv_data(Path(args.data_path))

    info = dataset_info(dataset)
    print("=== Dataset Info ===")
    print(f"Shape: {info['shape']}")
    print("Columns:")
    print(info["columns"])
    print("Data types:")
    print(info["dtypes"])

    print("\n=== Missing Values ===")
    print(missing_values_report(dataset))

    print("\n=== Basic Statistics ===")
    print(basic_statistics(dataset))


if __name__ == "__main__":
    main()
