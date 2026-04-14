"""Streamlit dashboard for fraud detection model inference and dataset visualizations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_csv_data
from src.predict import predict_from_dict
from src.visualize import plot_correlation_heatmap, plot_fraud_distribution


st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("Fraud Detection Dashboard")

st.sidebar.header("Inputs")
dataset_path = st.sidebar.text_input("Dataset CSV path", "data/processed/fraud.csv")
target_column = st.sidebar.text_input("Target column", "is_fraud")
model_path = st.sidebar.text_input("Model path", "models/artifacts/random_forest.joblib")

left, right = st.columns(2)

with left:
    st.subheader("Dataset Overview")
    try:
        df = load_csv_data(dataset_path)
        st.write("Shape:", df.shape)
        st.dataframe(df.head(10), use_container_width=True)

        fig_dist = plot_fraud_distribution(df, target_column=target_column)
        st.pyplot(fig_dist)
    except Exception as exc:
        st.warning(f"Dataset visualization unavailable: {exc}")

with right:
    st.subheader("Correlation Heatmap")
    try:
        df = load_csv_data(dataset_path)
        fig_corr = plot_correlation_heatmap(df)
        st.pyplot(fig_corr)
    except Exception as exc:
        st.warning(f"Heatmap unavailable: {exc}")

st.markdown("---")
st.subheader("Single Transaction Prediction")
st.caption("Provide one transaction as JSON. Example: {\"amount\": 1200, \"merchant\": \"A\", \"device\": \"mobile\"}")
transaction_json = st.text_area(
    "Transaction JSON",
    value='{"amount": 100.0, "merchant": "StoreA", "transaction_type": "online"}',
    height=120,
)

if st.button("Predict Fraud"):
    try:
        transaction = json.loads(transaction_json)
        result = predict_from_dict(model_path=model_path, transaction=transaction)
        st.success("Prediction complete")
        st.write(result)
    except json.JSONDecodeError:
        st.error("Invalid JSON format for transaction input.")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
