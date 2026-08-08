"""Register the best MLflow model."""

from __future__ import annotations

import os

import mlflow
from mlflow.tracking import MlflowClient


EXPERIMENT_NAME = "house-price-prediction"
REGISTERED_MODEL_NAME = "house-price-model"


def register_best_model() -> None:
    """Find the best run and register its model."""

    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        "file:./mlruns",
    )

    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)

    if experiment is None:
        raise RuntimeError(
            f"Experiment '{EXPERIMENT_NAME}' not found"
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"],
        max_results=1,
    )

    if not runs:
        raise RuntimeError("No runs found")

    best_run = runs[0]

    run_id = best_run.info.run_id
    model_type = best_run.data.params.get("model_type")
    rmse = best_run.data.metrics.get("rmse")

    print(f"Best run: {run_id}")
    print(f"Model type: {model_type}")
    print(f"RMSE: {rmse:.3f}")

    model_uri = f"runs:/{run_id}/model"

    print(f"Model URI: {model_uri}")

    result = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    print()
    print("Model registered successfully!")
    print(f"Name: {result.name}")
    print(f"Version: {result.version}")


if __name__ == "__main__":
    register_best_model()

# Also save a plain pickle for the API

import joblib
from pathlib import Path
import mlflow.sklearn

# Create models directory in the project root
models_dir = Path("../models")
models_dir.mkdir(exist_ok=True)

# Load the registered/best model
model_uri = "runs:/eadf53fb43f343e7b60e45d60c2c528b/model"

loaded = mlflow.sklearn.load_model(model_uri)

# Save as pickle
joblib.dump(loaded, models_dir / "best_model.pkl")

print("Saved models/best_model.pkl for API use")