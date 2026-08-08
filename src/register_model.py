"""Register the best MLflow model."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


EXPERIMENT_NAME = "house-price-prediction"
REGISTERED_MODEL_NAME = "house-price-model"


def register_best_model() -> None:
    """Find the best run, register its model, and save a pickle."""

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

    # Use the model from the BEST RUN
    model_uri = f"runs:/{run_id}/model"

    print(f"Model URI: {model_uri}")

    # Register model in MLflow
    result = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    print()
    print("Model registered successfully!")
    print(f"Name: {result.name}")
    print(f"Version: {result.version}")

    # Save plain pickle for the API
    # register_model.py is inside src/,
    # so ../models points to the project root models/ folder.
    models_dir = Path("../models")
    models_dir.mkdir(parents=True, exist_ok=True)

    loaded_model = mlflow.sklearn.load_model(model_uri)

    pickle_path = models_dir / "best_model.pkl"

    joblib.dump(loaded_model, pickle_path)

    print(f"Saved {pickle_path} for API use")


if __name__ == "__main__":
    register_best_model()