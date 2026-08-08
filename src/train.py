"""Train multiple regression models and log them to MLflow."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline

from data_loader import load_data, split_data
from preprocess import build_preprocessor


warnings.filterwarnings("ignore")

EXPERIMENT_NAME = "house-price-prediction"


def build_model_pipeline(model) -> Pipeline:
    """Combine preprocessing and model into a single sklearn Pipeline."""
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
) -> tuple[Pipeline, dict[str, float]]:
    """Train one model and log it to MLflow."""

    with mlflow.start_run(run_name=name):

        # Build pipeline
        pipeline = build_model_pipeline(model)

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        predictions = pipeline.predict(X_test)

        # Evaluate
        metrics = evaluate(y_test, predictions)

        # Log model type
        mlflow.log_param("model_type", name)

        # Log model parameters
        for key, value in model.get_params().items():
            if isinstance(value, (int, float, str, bool, type(None))):
                mlflow.log_param(key, value)

        # Log metrics
        for key, value in metrics.items():
            mlflow.log_metric(key, value)


            mlflow.sklearn.log_model(
    sk_model=pipeline,
    artifact_path="model",
    registered_model_name=None,
)

        
        

        # Print results
        run_id = mlflow.active_run().info.run_id

        print(f"Model logged successfully for run: {run_id}")
        print(
            f"[{name}] "
            f"MAE={metrics['mae']:.3f} "
            f"RMSE={metrics['rmse']:.3f} "
            f"R²={metrics['r2']:.3f}"
        )

        return pipeline, metrics


def main() -> None:
    """Train all models."""

    # Use SQLite instead of the deprecated filesystem MLflow backend.
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "sqlite:///mlflow.db",
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow experiment: {EXPERIMENT_NAME}")

    # Load data
    df = load_data()

    # Split data
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

    # Store results
    results = {}

    # Train every model
    for name, model in models.items():
        pipeline, metrics = train_one(
            name=name,
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
        )

        results[name] = {
            "pipeline": pipeline,
            "metrics": metrics,
        }

    # Find the model with the lowest RMSE
    best_model_name = min(
        results,
        key=lambda name: results[name]["metrics"]["rmse"],
    )

    best_pipeline = results[best_model_name]["pipeline"]
    best_metrics = results[best_model_name]["metrics"]

    print()
    print(f"Best model: {best_model_name}")
    print(f"Best RMSE: {best_metrics['rmse']:.3f}")

    # Save best model for FastAPI
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "best_model.pkl"

    joblib.dump(best_pipeline, model_path)

    print(f"Saved best model: {model_path}")

    print()
    print("All runs logged successfully.")
    print("View the dashboard with:")
    print("mlflow ui")


if __name__ == "__main__":
    main()