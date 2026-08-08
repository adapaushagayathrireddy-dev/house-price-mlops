"""FastAPI application serving house price predictions."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import HouseFeatures, PredictionResponse


app = FastAPI(
    title="House Price Prediction API",
    description="Predicts median California house prices from 8 features.",
    version="1.0.0",
)


# Allow frontend/dev browsers to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


model = None

MODEL_PATH = Path(
    os.getenv("MODEL_PATH", "models/best_model.pkl")
)

MODEL_VERSION = os.getenv(
    "MODEL_VERSION",
    "1"
)


@app.on_event("startup")
def load_model() -> None:
    global model

    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Model file not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    print(f"Loaded model from {MODEL_PATH}")


@app.get("/")
def root() -> dict:
    return {
        "service": "house-price-prediction",
        "status": "ok",
        "docs": "/docs",
        "model_version": MODEL_VERSION,
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "healthy",
        "model_loaded": model is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(features: HouseFeatures) -> PredictionResponse:

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded yet"
        )

    df = pd.DataFrame(
        [features.model_dump()]
    )

    try:
        pred_100k = float(
            model.predict(df)[0]
        )

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Prediction failed: {e}"
        )

    return PredictionResponse(
        predicted_price_usd=round(
            pred_100k * 100000,
            2
        ),
        predicted_price_100k=round(
            pred_100k,
            4
        ),
        model_version=MODEL_VERSION,
    )