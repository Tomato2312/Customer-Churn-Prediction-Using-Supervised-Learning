"""FastAPI service exposing the churn model.

Run with:
    uvicorn app.api:app --reload --port 8000

Then:
    POST /predict
    {
      "Gender": "Female", "Senior Citizen": "No", "Partner": "Yes",
      "Dependents": "No", "Tenure Months": 5, "Phone Service": "Yes",
      "Multiple Lines": "No", "Internet Service": "Fiber optic",
      "Online Security": "No", "Online Backup": "No",
      "Device Protection": "No", "Tech Support": "No",
      "Streaming TV": "No", "Streaming Movies": "No",
      "Contract": "Month-to-month", "Paperless Billing": "Yes",
      "Payment Method": "Electronic check", "Monthly Charges": 85.05,
      "Total Charges": 425.25, "CLTV": 3200
    }

Requires models/churn_pipeline.joblib (see `python -m src.train`).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Scores a single telco customer's probability of churn.",
    version="1.0.0",
)

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        if not config.MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model not found at {config.MODEL_PATH}. Run `python -m src.train` first.",
            )
        _pipeline = joblib.load(config.MODEL_PATH)
    return _pipeline


YesNo = Literal["Yes", "No"]


class CustomerProfile(BaseModel):
    Gender: Literal["Male", "Female"]
    Senior_Citizen: YesNo = Field(alias="Senior Citizen")
    Partner: YesNo
    Dependents: YesNo
    Tenure_Months: int = Field(alias="Tenure Months", ge=0, le=100)
    Phone_Service: YesNo = Field(alias="Phone Service")
    Multiple_Lines: Literal["Yes", "No", "No phone service"] = Field(alias="Multiple Lines")
    Internet_Service: Literal["DSL", "Fiber optic", "No"] = Field(alias="Internet Service")
    Online_Security: Literal["Yes", "No", "No internet service"] = Field(alias="Online Security")
    Online_Backup: Literal["Yes", "No", "No internet service"] = Field(alias="Online Backup")
    Device_Protection: Literal["Yes", "No", "No internet service"] = Field(alias="Device Protection")
    Tech_Support: Literal["Yes", "No", "No internet service"] = Field(alias="Tech Support")
    Streaming_TV: Literal["Yes", "No", "No internet service"] = Field(alias="Streaming TV")
    Streaming_Movies: Literal["Yes", "No", "No internet service"] = Field(alias="Streaming Movies")
    Contract: Literal["Month-to-month", "One year", "Two year"]
    Paperless_Billing: YesNo = Field(alias="Paperless Billing")
    Payment_Method: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ] = Field(alias="Payment Method")
    Monthly_Charges: float = Field(alias="Monthly Charges", ge=0)
    Total_Charges: float = Field(alias="Total Charges", ge=0)
    CLTV: int

    class Config:
        populate_by_name = True


class PredictionResponse(BaseModel):
    churn_probability: float
    action: str
    threshold_used: float


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": config.MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictionResponse)
def predict(profile: CustomerProfile, threshold: float = 0.37):
    pipeline = get_pipeline()
    row = pd.DataFrame([profile.model_dump(by_alias=True)])[config.FEATURE_COLUMNS]
    probability = float(pipeline.predict_proba(row)[0, 1])
    action = "Priority retention contact" if probability >= threshold else "Monitor"
    return PredictionResponse(churn_probability=probability, action=action, threshold_used=threshold)
