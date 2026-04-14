"""Preprocessing pipeline builders."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def infer_feature_types(
    x: pd.DataFrame,
    numeric_columns: Sequence[str] | None = None,
    categorical_columns: Sequence[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Infer numeric and categorical feature lists if not explicitly provided."""
    if numeric_columns is not None and categorical_columns is not None:
        return list(numeric_columns), list(categorical_columns)

    numeric_cols = list(x.select_dtypes(include=["number"]).columns)
    categorical_cols = [col for col in x.columns if col not in numeric_cols]
    return numeric_cols, categorical_cols


def build_preprocessing_pipeline(
    x_train: pd.DataFrame,
    numeric_columns: Sequence[str] | None = None,
    categorical_columns: Sequence[str] | None = None,
) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build preprocessing pipeline with imputation, encoding, and scaling."""
    numeric_cols, categorical_cols = infer_feature_types(
        x=x_train,
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

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_cols:
        transformers.append(("num", numeric_pipeline, numeric_cols))
    if categorical_cols:
        transformers.append(("cat", categorical_pipeline, categorical_cols))

    if not transformers:
        raise ValueError("No usable features found for preprocessing.")

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor, numeric_cols, categorical_cols
