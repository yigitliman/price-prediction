"""
House price prediction training script.

Uses the California Housing dataset from scikit-learn.
Trains an XGBoost regressor and logs everything to MLflow.

Usage:
    python src/train.py
    python src/train.py --max-depth 6 --n-estimators 200
"""

import argparse
import os

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "model.joblib")
REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "reference.csv")


def load_data() -> pd.DataFrame:
    dataset = fetch_california_housing(as_frame=True)
    df = dataset.frame
    return df


def train(max_depth: int, n_estimators: int, learning_rate: float) -> None:
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("price-prediction")

    with mlflow.start_run():
        mlflow.log_params({
            "max_depth": max_depth,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "dataset": "california_housing",
        })

        df = load_data()
        X = df.drop(columns=["MedHouseVal"])
        y = df["MedHouseVal"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = XGBRegressor(
            max_depth=max_depth,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=42,
        )
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)

        metrics = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
        }
        mlflow.log_metrics(metrics)
        print(f"Metrics: {metrics}")

        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(
            {
                "model": model,
                "scaler": scaler,
                "feature_names": list(X.columns),
                "rmse": metrics["rmse"],
            },
            MODEL_PATH,
        )
        X.head(200).to_csv(REFERENCE_PATH, index=False)

        print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    args = parser.parse_args()
    train(args.max_depth, args.n_estimators, args.learning_rate)
