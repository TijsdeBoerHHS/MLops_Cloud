"""FastAPI deployment voor het XGBoost verkeersvolume model."""

import os
import xgboost as xgb
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

MODEL_PATH = os.getenv("MODEL_PATH", "model.json")
model = xgb.XGBRegressor()
model.load_model(MODEL_PATH)

# Exacte feature volgorde zoals het model die verwacht
FEATURE_COLS = model.get_booster().feature_names

# Alle mogelijke weather_main en weather_description waarden
WEATHER_MAIN_VALUES = [
    "Clear", "Clouds", "Drizzle", "Fog", "Haze", "Mist",
    "Rain", "Smoke", "Snow", "Squall", "Thunderstorm"
]
WEATHER_DESC_VALUES = [
    "SQUALLS", "Sky is Clear", "broken clouds", "drizzle", "few clouds",
    "fog", "freezing rain", "haze", "heavy intensity drizzle",
    "heavy intensity rain", "heavy snow", "light intensity drizzle",
    "light intensity shower rain", "light rain", "light rain and snow",
    "light shower snow", "light snow", "mist", "moderate rain",
    "overcast clouds", "proximity shower rain", "proximity thunderstorm",
    "proximity thunderstorm with drizzle", "proximity thunderstorm with rain",
    "scattered clouds", "shower drizzle", "shower snow", "sky is clear",
    "sleet", "smoke", "snow", "thunderstorm", "thunderstorm with drizzle",
    "thunderstorm with heavy rain", "thunderstorm with light drizzle",
    "thunderstorm with light rain", "thunderstorm with rain", "very heavy rain"
]

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
    weather_main: Optional[str] = Field(None, description="Bijv. Clouds, Rain, Clear")
    weather_description: Optional[str] = Field(None, description="Bijv. scattered clouds")


class PredictionResponse(BaseModel):
    predicted_traffic_volume: float
    model_path: str


@app.get("/health")
def health_check():
    return {"status": "ok", "model_path": MODEL_PATH}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: TrafficFeatures):
    # Basisfeatures met hoofdletters zoals het model verwacht
    row = {
        "holiday": float(features.holiday),
        "temp": features.temp,
        "rain_1h": features.rain_1h,
        "snow_1h": features.snow_1h,
        "clouds_all": float(features.clouds_all),
        "Hour": float(features.hour),
        "DayOfWeek": float(features.day_of_week),
        "Month": float(features.month),
        "Year": float(features.year),
    }

    # One-hot encoding weather_main
    for val in WEATHER_MAIN_VALUES:
        row[f"weather_main_{val}"] = float(
            features.weather_main == val if features.weather_main else False
        )

    # One-hot encoding weather_description
    for val in WEATHER_DESC_VALUES:
        row[f"weather_description_{val}"] = float(
            features.weather_description == val if features.weather_description else False
        )

    # DataFrame maken met exacte kolomvolgorde van het model
    df = pd.DataFrame([row])
    if FEATURE_COLS:
        df = df[FEATURE_COLS]

    try:
        prediction = model.predict(df)
        return PredictionResponse(
            predicted_traffic_volume=float(prediction[0]),
            model_path=MODEL_PATH,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc