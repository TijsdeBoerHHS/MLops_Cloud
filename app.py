"""FastAPI deployment voor het XGBoost verkeersvolume model.

Het model wordt geladen vanuit model.json (XGBoost native formaat).
Zo is er geen MLflow server nodig in de cloud.

Start de server met:
    uvicorn app:app --host 0.0.0.0 --port 8000

Of via Docker:
    docker build -f Dockerfile.api -t traffic-api .
    docker run -p 8000:8000 -v $(pwd)/model.json:/app/model.json traffic-api
"""

import os
import xgboost as xgb
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

# Model laden vanuit bestand (werkt lokaal én in de cloud)
MODEL_PATH = os.getenv("MODEL_PATH", "model.json")
model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)

app = FastAPI(
    title="Traffic Volume Prediction API",
    description="Voorspelt het uurlijkse verkeersvolume op de I-94 snelweg.",
    version="1.0.0",
)


class TrafficFeatures(BaseModel):
    holiday: bool = Field(..., description="Is het een feestdag?")
    temp: float = Field(..., description="Temperatuur in Kelvin", ge=200, le=350)
    rain_1h: float = Field(0.0, description="Regen in mm", ge=0)
    snow_1h: float = Field(0.0, description="Sneeuw in mm", ge=0)
    clouds_all: int = Field(..., description="Wolkendekking %", ge=0, le=100)
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=1, le=7)
    month: int = Field(..., ge=1, le=12)
    year: int = Field(..., ge=2000, le=2100)
    weather_main: Optional[str] = None
    weather_description: Optional[str] = None


class PredictionResponse(BaseModel):
    predicted_traffic_volume: float
    model_path: str


@app.get("/health")
def health_check():
    return {"status": "ok", "model_path": MODEL_PATH}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: TrafficFeatures):
    df = pd.DataFrame([features.model_dump()])
    try:
        prediction = model.predict(df)
        return PredictionResponse(
            predicted_traffic_volume=float(prediction[0]),
            model_path=MODEL_PATH,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
