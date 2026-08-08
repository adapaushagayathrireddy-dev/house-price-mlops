"""Train multiple regression models and log to MLflow."""

from __future__ import annotations
import joblib
from pathlib import Path
import os
import warnings

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from data_loader import load_data, split_data
from preprocess import build_preprocessor

warnings.filterwarnings("ignore")

EXPERIMENT_NAME = "house-price-prediction"


def build_model_pipeline(model) -> Pipeline:
    """Combine preprocessing + model into a single sklearn Pipeline."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("model", model),
        ]
    )


def evaluate(y_true, y_pred) -> dict[str, float]:
    """Compute regression metrics."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_one(
    name: str,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
) -> None:
    """Train one model and log everything to MLflow."""

    with mlflow.start_run(run_name=name):
        pipeline = build_model_pipeline(model)

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        preds = pipeline.predict(X_test)

        # Evaluate
        metrics = evaluate(y_test, preds)

        # Log model type
        mlflow.log_param("model_type", name)

        # Log model parameters
        for k, v in model.get_params().items():
            if isinstance(v, (int, float, str, bool, type(None))):
                mlflow.log_param(k, v)

        # Log metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Log complete preprocessing + model pipeline
            # Save trained model for FastAPI
    if name == "gradient_boosting":
        Path("models").mkdir(exist_ok=True)

        joblib.dump(
            pipeline,
            "models/best_model.pkl"
        )

        print("Saved best model: models/best_model.pkl")
        

        print(
            f"[{name}] "
            f"MAE={metrics['mae']:.3f} "
            f"RMSE={metrics['rmse']:.3f} "
            f"R²={metrics['r2']:.3f}"
        )


def main() -> None:
    """Train all models."""

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "file:./mlruns",
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load and split data
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # Models to compare
    models = {
        "linear_regression": LinearRegression(),

        "random_forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1,
        ),

        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        ),
    }

    # Train each model
    for name, model in models.items():
        train_one(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        )

    print("\nAll runs logged. View the dashboard with:")
    print("mlflow ui")


if __name__ == "__main__":
    main()