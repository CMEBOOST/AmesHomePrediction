from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator
from typing import Literal
import pandas as pd
import numpy as np
import joblib
import os
from ml_pipeline import (
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, NEIGHBORHOODS, make_form_matrix,
)
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Ames Housing Price Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, change to the Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_artifacts.joblib")

# Load model globally
try:
    artifacts = joblib.load(MODEL_PATH)
    models = artifacts['models']
    weights = artifacts['weights']
    train_ranges = artifacts['train_ranges']
    print("Model artifacts loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    models = None


def _limits(name):
    _, low, high = NUMERIC_FEATURES[name]
    return dict(ge=low, le=high)


class HomeData(BaseModel):
    # The 15 fields the form asks for. The model is trained on exactly these
    # fields, so every one is required and range-checked.
    OverallQual: int = Field(..., **_limits('OverallQual'))    # overall material & finish (1-10)
    OverallCond: int = Field(..., **_limits('OverallCond'))    # overall condition (1-10)
    GrLivArea: float = Field(..., **_limits('GrLivArea'))      # above-ground living area (sq ft)
    FirstFlrSF: float = Field(..., **_limits('FirstFlrSF'))    # first floor area (sq ft)
    TotalBsmtSF: float = Field(..., **_limits('TotalBsmtSF'))  # total basement area (sq ft)
    BsmtFinSF1: float = Field(..., **_limits('BsmtFinSF1'))    # finished basement area (sq ft)
    LotArea: float = Field(..., **_limits('LotArea'))          # lot size (sq ft)
    YearBuilt: int = Field(..., **_limits('YearBuilt'))
    YearRemodAdd: int = Field(..., **_limits('YearRemodAdd'))  # remodel year (= YearBuilt if none)
    GarageCars: int = Field(..., **_limits('GarageCars'))
    Fireplaces: int = Field(..., **_limits('Fireplaces'))
    FullBath: int = Field(..., **_limits('FullBath'))
    Neighborhood: Literal[tuple(NEIGHBORHOODS)]
    MSZoning: Literal[tuple(CATEGORICAL_FEATURES['MSZoning'])]
    CentralAir: Literal[tuple(CATEGORICAL_FEATURES['CentralAir'])]

    @model_validator(mode='after')
    def check_consistency(self):
        if self.YearRemodAdd < self.YearBuilt:
            raise ValueError("ปีที่รีโนเวตต้องไม่น้อยกว่าปีที่สร้าง (ถ้าไม่เคยรีโนเวต ให้ใช้ปีที่สร้าง)")
        if self.GrLivArea < self.FirstFlrSF:
            raise ValueError("พื้นที่ใช้สอยเหนือพื้นดินต้องไม่น้อยกว่าพื้นที่ชั้น 1")
        return self


@app.get("/")
def read_root():
    return {"message": "Welcome to Ames Home Prediction API"}


@app.get("/features")
def get_features():
    """Fields the model needs, with defaults and allowed values."""
    return {
        "numeric": {k: {"default": v[0], "min": v[1], "max": v[2]} for k, v in NUMERIC_FEATURES.items()},
        "categorical": CATEGORICAL_FEATURES,
    }


@app.post("/predict")
def predict_price(data: HomeData):
    if models is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    df_input = pd.DataFrame([data.model_dump()])
    X = make_form_matrix(df_input)

    # test_preds has shape (1, 4) since we have 4 models (log scale)
    test_preds = np.column_stack([pipe.predict(X) for pipe in models.values()])

    # Calculate final weighted prediction
    final_log_pred = np.dot(test_preds, weights)[0]
    final_price = np.expm1(final_log_pred)

    # Calculate confidence based on Ensemble Agreement
    # We convert log predictions to dollar prices for all 4 models
    dollar_preds = np.expm1(test_preds[0])
    # Calculate standard deviation among the 4 predictions
    std_dev = np.std(dollar_preds)

    # Coefficient of Variation (CV) = Standard Deviation / Mean
    cv = std_dev / final_price

    # Map CV to a confidence score (0 to 100)
    # If standard deviation is 5% of the price (cv = 0.05), confidence is 95%
    confidence_score = max(0, min(99.9, 100 * (1 - cv)))

    # Warn (do not reject) when an input is outside what the model has seen
    warnings = []
    for name, (low, high) in train_ranges.items():
        value = getattr(data, name)
        if value < low or value > high:
            warnings.append(f"{name}={value} is outside the training range ({low:g} to {high:g}); the estimate may be unreliable")

    return {
        "predicted_price": round(float(final_price), 2),
        "confidence_score": round(float(confidence_score), 1),
        "warnings": warnings,
    }
