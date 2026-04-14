"""Load a CSV file and print the first N rows (default: 5)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Literal

import pandas as pd


def load_csv(file_path: str | Path) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    return pd.read_csv(path)


def print_preview(dataframe: pd.DataFrame, rows: int = 5) -> None:
    """Print the first `rows` rows of a DataFrame."""
    print(dataframe.head(rows))


def display_dataset_info(dataframe: pd.DataFrame) -> None:
    """Display shape, columns, dtypes, and missing values for a DataFrame."""
    print("\n=== Dataset Info ===")
    print(f"Shape: {dataframe.shape}")

    print("\nColumns:")
    print(list(dataframe.columns))

    print("\nData types:")
    print(dataframe.dtypes)

    print("\nMissing values per column:")
    print(dataframe.isnull().sum())


def handle_missing_values(
    dataframe: pd.DataFrame,
    strategy: Literal[
        "drop", "mean", "median", "mode", "ffill", "bfill", "constant"
    ] = "mean",
    fill_value: Any | None = None,
) -> pd.DataFrame:
    """Handle missing values using a basic strategy and return a new DataFrame."""
    df = dataframe.copy()

    if strategy == "drop":
        return df.dropna()

    if strategy == "mean":
        numeric_columns = df.select_dtypes(include="number").columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        return df

    if strategy == "median":
        numeric_columns = df.select_dtypes(include="number").columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
        return df

    if strategy == "mode":
        for column in df.columns:
            mode_values = df[column].mode(dropna=True)
            if not mode_values.empty:
                df[column] = df[column].fillna(mode_values.iloc[0])
        return df

    if strategy == "ffill":
        return df.ffill()

    if strategy == "bfill":
        return df.bfill()

    if strategy == "constant":
        if fill_value is None:
            raise ValueError("`fill_value` must be provided when strategy='constant'.")
        return df.fillna(fill_value)

    raise ValueError(
        "Invalid strategy. Use one of: "
        "'drop', 'mean', 'median', 'mode', 'ffill', 'bfill', 'constant'."
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Load a CSV file and print the first rows."
    )
    parser.add_argument("csv_path", help="Path to the input CSV file")
    parser.add_argument(
        "--rows",
        type=int,
        default=5,
        help="Number of rows to display (default: 5)",
    )
    return parser.parse_args()


def main() -> None:
    """Run the CSV preview workflow."""
    args = parse_args()
    df = load_csv(args.csv_path)
    display_dataset_info(df)
    print("\n=== Preview ===")
    print_preview(df, rows=args.rows)


if __name__ == "__main__":
    main()
