"""
FastAPI serving layer for the Ames Housing price model.

Run:
    uvicorn main:app --reload

Then open http://127.0.0.1:8000/docs for the interactive Swagger UI —
it's the easiest way to try /predict without writing a client.
"""

import json
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from preprocessing import clean_and_encode, align_to_training_columns

app = FastAPI(
    title="Ames Housing Price Predictor",
    description="Predicts residential sale price from house features, "
                 "trained on the Kaggle Ames Housing dataset.",
    version="1.0.0",
)

MODEL = joblib.load("model.pkl")
FEATURE_COLUMNS = json.load(open("feature_columns.json"))
DEFAULTS = json.load(open("defaults.json"))


class HouseFeatures(BaseModel):
    """Every field is optional — anything you don't provide falls back
    to the training set's median (numeric) or most common value
    (categorical), the same rule used for genuinely-missing data during
    training. Only the fields that matter most for price need to be
    filled in for a meaningful prediction."""

    OverallQual: Optional[int] = Field(None, ge=1, le=10, description="Overall material/finish quality, 1-10")
    OverallCond: Optional[int] = Field(None, ge=1, le=10, description="Overall condition, 1-10")
    GrLivArea: Optional[float] = Field(None, description="Above-grade living area, sq ft")
    TotalBsmtSF: Optional[float] = Field(None, description="Total basement area, sq ft")
    GarageCars: Optional[float] = Field(None, description="Garage capacity, number of cars")
    YearBuilt: Optional[int] = Field(None, description="Original construction year")
    YearRemodAdd: Optional[int] = Field(None, description="Remodel year (= YearBuilt if never remodeled)")
    Neighborhood: Optional[str] = Field(None, description="e.g. 'CollgCr', 'OldTown', 'NridgHt'")
    KitchenQual: Optional[str] = Field(None, description="Ex / Gd / TA / Fa / Po")
    FullBath: Optional[int] = Field(None, description="Full bathrooms above grade")
    BedroomAbvGr: Optional[int] = Field(None, description="Bedrooms above grade")
    LotArea: Optional[float] = Field(None, description="Lot size, sq ft")

    class Config:
        extra = "allow"  # accept any other raw Kaggle column too, if supplied


class PredictionResponse(BaseModel):
    predicted_price: float
    currency: str = "USD"


@app.get("/health")
def health():
    return {"status": "ok", "features_expected": len(FEATURE_COLUMNS)}


@app.post("/predict", response_model=PredictionResponse)
def predict(house: HouseFeatures):
    input_dict = house.dict(exclude_unset=False)

    # fill anything not supplied with the saved training default
    row = {}
    for col, default_val in DEFAULTS.items():
        supplied = input_dict.get(col)
        row[col] = supplied if supplied is not None else default_val

    df = pd.DataFrame([row])

    try:
        df = clean_and_encode(df)
        df = align_to_training_columns(df, FEATURE_COLUMNS)
        pred_log = MODEL.predict(df)
        pred_price = float(np.expm1(pred_log)[0])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

    return PredictionResponse(predicted_price=round(pred_price, 2))
