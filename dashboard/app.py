"""Streamlit dashboard that consumes the Fraud Detection FastAPI service."""

from __future__ import annotations

import json

import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("Fraud Detection Dashboard")
st.caption("Dashboard is API-driven and does not load ML artifacts directly.")

st.sidebar.header("API Settings")
api_url = st.sidebar.text_input("Predict endpoint", "http://127.0.0.1:8000/predict")

left, right = st.columns(2)

with left:
    st.subheader("Transaction Prediction")
    transaction_json = st.text_area(
        "Transaction JSON",
        value='{"amount": 100.0, "merchant": "StoreA", "transaction_type": "online"}',
        height=140,
    )

    if st.button("Predict Fraud", type="primary"):
        try:
            transaction = json.loads(transaction_json)
            response = requests.post(
                api_url,
                json={"transaction": transaction},
                timeout=10,
            )
            response.raise_for_status()
            prediction = response.json()
            st.success("Prediction completed")
            st.json(prediction)
        except json.JSONDecodeError:
            st.error("Transaction JSON is invalid.")
        except requests.RequestException as exc:
            st.error(f"API request failed: {exc}")

with right:
    st.subheader("Quick Dataset Snapshot")
    uploaded = st.file_uploader("Upload CSV for quick visualization", type=["csv"])
    target_col = st.text_input("Target column", value="is_fraud")

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Shape:", df.shape)
        st.dataframe(df.head(10), use_container_width=True)

        if target_col in df.columns:
            st.write("Fraud Distribution")
            counts = df[target_col].value_counts(dropna=False)
            st.bar_chart(counts)
        else:
            st.info(f"Column '{target_col}' not found for distribution chart.")
