"""Preprocessing utilities for fraud detection ML pipelines."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NumericCols = list[str]
CategoricalCols = list[str]


def split_features_target(df: pd.DataFrame, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    """Split a DataFrame into features and target."""
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset.")

    x = df.drop(columns=[target_column])
    y = df[target_column]
    return x, y


def infer_feature_types(
    x: pd.DataFrame,
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
) -> tuple[NumericCols, CategoricalCols]:
    """Infer or validate numeric and categorical feature lists."""
    if numeric_features is not None and categorical_features is not None:
        numeric_cols = list(numeric_features)
        categorical_cols = list(categorical_features)
    else:
        numeric_cols = list(x.select_dtypes(include=["number"]).columns)
        categorical_cols = [col for col in x.columns if col not in numeric_cols]

    if not numeric_cols and not categorical_cols:
        raise ValueError("No usable feature columns found.")

    return numeric_cols, categorical_cols


def build_preprocessor(
    x: pd.DataFrame,
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] | None = None,
) -> tuple[ColumnTransformer, NumericCols, CategoricalCols]:
    """Build a preprocessing pipeline with scaling and encoding."""
    numeric_cols, categorical_cols = infer_feature_types(
        x=x,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_cols:
        transformers.append(("num", numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_transformer, categorical_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return preprocessor, numeric_cols, categorical_cols


def split_train_test(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create train/test split with optional stratification for classification."""
    stratify = y if y.nunique(dropna=False) > 1 else None
    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
