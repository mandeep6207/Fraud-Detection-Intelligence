"""Load a CSV file and print the first N rows (default: 5)."""

from __future__ import annotations

import argparse
from pathlib import Path

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
    print_preview(df, rows=args.rows)


if __name__ == "__main__":
    main()
