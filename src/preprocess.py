
"""Build the preprocessing pipeline."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


NUMERIC_FEATURES = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]


def build_preprocessor() -> ColumnTransformer:
    """All 8 features are numeric — scale them."""

    numeric_pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler())
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES)
        ]
    )