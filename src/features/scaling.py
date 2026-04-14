"""Reusable feature scaling utilities for fraud detection pipelines."""

from __future__ import annotations

from typing import Sequence

import pandas as pd
from sklearn.preprocessing import StandardScaler


def scale_features_standard(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None = None,
    columns: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, StandardScaler]:
    """Fit StandardScaler on train data and scale train/test DataFrames.

    If `columns` is None, numeric columns from `train_df` are scaled.
    Returns scaled train DataFrame, scaled test DataFrame (if provided), and fitted scaler.
    """
    if train_df.empty:
        raise ValueError("train_df must not be empty.")

    columns_to_scale = list(columns) if columns is not None else list(
        train_df.select_dtypes(include="number").columns
    )

    if not columns_to_scale:
        raise ValueError("No columns available to scale.")

    missing_train_cols = [col for col in columns_to_scale if col not in train_df.columns]
    if missing_train_cols:
        raise ValueError(f"Columns not found in train_df: {missing_train_cols}")

    scaler = StandardScaler()

    train_scaled = train_df.copy()
    train_scaled_values = scaler.fit_transform(train_df[columns_to_scale])
    train_scaled.loc[:, columns_to_scale] = train_scaled_values

    test_scaled: pd.DataFrame | None = None
    if test_df is not None:
        missing_test_cols = [col for col in columns_to_scale if col not in test_df.columns]
        if missing_test_cols:
            raise ValueError(f"Columns not found in test_df: {missing_test_cols}")

        test_scaled = test_df.copy()
        test_scaled_values = scaler.transform(test_df[columns_to_scale])
        test_scaled.loc[:, columns_to_scale] = test_scaled_values

    return train_scaled, test_scaled, scaler


def transform_with_scaler(
    dataframe: pd.DataFrame,
    scaler: StandardScaler,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Transform a DataFrame with a pre-fitted StandardScaler."""
    if dataframe.empty:
        raise ValueError("dataframe must not be empty.")

    missing_cols = [col for col in columns if col not in dataframe.columns]
    if missing_cols:
        raise ValueError(f"Columns not found in dataframe: {missing_cols}")

    transformed_df = dataframe.copy()
    transformed_values = scaler.transform(dataframe[list(columns)])
    transformed_df.loc[:, list(columns)] = transformed_values
    return transformed_df
