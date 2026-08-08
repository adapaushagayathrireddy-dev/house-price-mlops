"""Preprocessing pipeline for house price data."""

from sklearn.compose import ColumnTransformer
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


def build_preprocessor():
    """
    Creates preprocessing pipeline:
    - Select numeric features
    - Scale them using StandardScaler
    """

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                StandardScaler(),
                NUMERIC_FEATURES
            )
        ]
    )

    return preprocessor