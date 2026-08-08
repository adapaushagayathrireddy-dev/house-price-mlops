
"""Find the best MLflow run based on RMSE (lower is better)."""

from __future__ import annotations

import os

import mlflow
from mlflow.tracking import MlflowClient


EXPERIMENT_NAME = "house-price-prediction"


def find_best_run() -> str:
    """Find and return the run ID with the lowest RMSE."""

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

    best = runs[0]

    print(f"Best run: {best.info.run_id}")
    print(f"  Model type: {best.data.params.get('model_type')}")
    print(f"  RMSE: {best.data.metrics['rmse']:.3f}")
    print(f"  R²:   {best.data.metrics['r2']:.3f}")

    return best.info.run_id


if __name__ == "__main__":
    find_best_run()