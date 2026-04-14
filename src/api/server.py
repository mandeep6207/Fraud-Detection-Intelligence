"""Fraud prediction API server."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.inference.predict import predict_transaction


app = FastAPI(title="Fraud Detection API", version="1.0.0")


class PredictRequest(BaseModel):
    transaction: dict[str, Any] = Field(..., description="Transaction payload")


class PredictResponse(BaseModel):
    fraud_probability: float
    label: int
    threshold: float
    model: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        result = predict_transaction(payload.transaction, artifacts_dir="models")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictResponse(**result)
