"""Preprocessing utilities for fraud detection pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_features_target(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Split input dataframe into features X and target y."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")
    x = df.drop(columns=[target_column])
    y = df[target_column]
    return x, y


def handle_missing_values(
    df: pd.DataFrame,
    strategy: Literal["drop", "mean", "median", "mode", "ffill", "bfill"] = "median",
) -> pd.DataFrame:
    """Handle missing values with simple global strategies."""
    cleaned = df.copy()

    if strategy == "drop":
        return cleaned.dropna()
    if strategy == "mean":
        num_cols = cleaned.select_dtypes(include="number").columns
        cleaned[num_cols] = cleaned[num_cols].fillna(cleaned[num_cols].mean())
        return cleaned
    if strategy == "median":
        num_cols = cleaned.select_dtypes(include="number").columns
        cleaned[num_cols] = cleaned[num_cols].fillna(cleaned[num_cols].median())
        return cleaned
    if strategy == "mode":
        for col in cleaned.columns:
            mode_values = cleaned[col].mode(dropna=True)
            if not mode_values.empty:
                cleaned[col] = cleaned[col].fillna(mode_values.iloc[0])
        return cleaned
    if strategy == "ffill":
        return cleaned.ffill()
    if strategy == "bfill":
        return cleaned.bfill()

    raise ValueError("Unsupported strategy.")


def infer_column_types(
    x: pd.DataFrame,
    numeric_columns: Sequence[str] | None = None,
    categorical_columns: Sequence[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical columns if not explicitly provided."""
    if numeric_columns is not None and categorical_columns is not None:
        return list(numeric_columns), list(categorical_columns)

    numeric_cols = list(x.select_dtypes(include=["number"]).columns)
    categorical_cols = [col for col in x.columns if col not in numeric_cols]
    return numeric_cols, categorical_cols


def build_preprocessing_pipeline(
    x: pd.DataFrame,
    numeric_columns: Sequence[str] | None = None,
    categorical_columns: Sequence[str] | None = None,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build preprocessing pipeline with missing handling, encoding, and scaling."""
    numeric_cols, categorical_cols = infer_column_types(
        x,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_cols, categorical_cols
