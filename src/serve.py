"""
House price prediction API.

Loads the XGBoost model and scaler saved by train.py.
"""

import contextlib
import os
import sqlite3
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "model.joblib")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "predictions.db")

model_bundle = None


def load_model():
    global model_bundle
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model not found. Run src/train.py first.")
    model_bundle = joblib.load(MODEL_PATH)
    logger.info("Model loaded.")


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            med_inc         REAL,
            house_age       REAL,
            ave_rooms       REAL,
            ave_bedrms      REAL,
            population      REAL,
            ave_occup       REAL,
            latitude        REAL,
            longitude       REAL,
            predicted_price REAL,
            timestamp       DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def log_prediction(features: dict, price: float):
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """INSERT INTO predictions
               (med_inc, house_age, ave_rooms, ave_bedrms, population,
                ave_occup, latitude, longitude, predicted_price)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (features["MedInc"], features["HouseAge"], features["AveRooms"],
             features["AveBedrms"], features["Population"], features["AveOccup"],
             features["Latitude"], features["Longitude"], price),
        )
        conn.commit()
    finally:
        conn.close()


def scale_and_predict(house_dict: dict) -> tuple[float, float, float]:
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    feature_names = model_bundle["feature_names"]

    df = pd.DataFrame([house_dict])[feature_names]
    scaled = scaler.transform(df)
    price = float(model.predict(scaled)[0])

    # Approximate 95% interval from the model's test RMSE saved at training time
    margin = float(model_bundle["rmse"] * 1.96)

    return round(price, 4), round(max(0, price - margin), 4), round(price + margin, 4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    load_model()
    yield


app = FastAPI(
    title="House Price Prediction API",
    description="Predicts California house prices using XGBoost.",
    version="1.0.0",
    lifespan=lifespan,
)


class HouseFeatures(BaseModel):
    MedInc: float = Field(..., gt=0, description="Median income in block group")
    HouseAge: float = Field(..., ge=0, description="Median house age in block group")
    AveRooms: float = Field(..., gt=0, description="Average number of rooms per household")
    AveBedrms: float = Field(..., ge=0, description="Average number of bedrooms per household")
    Population: float = Field(..., gt=0, description="Block group population")
    AveOccup: float = Field(..., gt=0, description="Average household size")
    Latitude: float = Field(..., ge=-90, le=90, description="Block group latitude")
    Longitude: float = Field(..., ge=-180, le=180, description="Block group longitude")


class PredictionResponse(BaseModel):
    predicted_price: float
    price_low: float
    price_high: float
    unit: str = "100k USD"


class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]
    total: int
    avg_predicted_price: float


@app.get("/")
def root():
    return {"message": "House Price Prediction API is running. Visit /docs for usage."}


@app.get("/health")
def health():
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=PredictionResponse)
def predict(house: HouseFeatures):
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    price, low, high = scale_and_predict(house.model_dump())
    log_prediction(house.model_dump(), price)
    return PredictionResponse(predicted_price=price, price_low=low, price_high=high)


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(houses: list[HouseFeatures]):
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if not houses:
        raise HTTPException(status_code=400, detail="House list cannot be empty.")

    results = []
    for house in houses:
        price, low, high = scale_and_predict(house.model_dump())
        log_prediction(house.model_dump(), price)
        results.append(PredictionResponse(predicted_price=price, price_low=low, price_high=high))

    avg = round(sum(r.predicted_price for r in results) / len(results), 4)
    return BatchPredictionResponse(results=results, total=len(results), avg_predicted_price=avg)


@app.get("/logs")
def get_logs(limit: int = Query(default=20, ge=1, le=500)):
    with contextlib.closing(sqlite3.connect(DB_PATH)) as conn:
        rows = conn.execute(
            """SELECT med_inc, house_age, ave_rooms, ave_bedrms, population,
                      ave_occup, latitude, longitude, predicted_price, timestamp
               FROM predictions ORDER BY timestamp DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    keys = ["med_inc", "house_age", "ave_rooms", "ave_bedrms", "population",
            "ave_occup", "latitude", "longitude", "predicted_price", "timestamp"]
    return [dict(zip(keys, row)) for row in rows]


@app.get("/stats")
def get_stats():
    with contextlib.closing(sqlite3.connect(DB_PATH)) as conn:
        row = conn.execute(
            "SELECT COUNT(*), AVG(predicted_price), MIN(predicted_price), "
            "MAX(predicted_price) FROM predictions"
        ).fetchone()
    return {
        "total_predictions": row[0],
        "avg_price": round(row[1], 4) if row[1] else 0.0,
        "min_price": round(row[2], 4) if row[2] else 0.0,
        "max_price": round(row[3], 4) if row[3] else 0.0,
    }
