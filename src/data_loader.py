"""Load the California Housing dataset."""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


def load_data() -> pd.DataFrame:
    """Return the California Housing dataset as a DataFrame."""
    bunch = fetch_california_housing(as_frame=True)
    df = bunch.frame
    df = df.rename(columns={"MedHouseVal": "target"})
    return df


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
):
    """Return X_train, X_test, y_train, y_test."""
    X = df.drop(columns=["target"])
    y = df["target"]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )


if __name__ == "__main__":
    df = load_data()

    print(f"Shape: {df.shape}")

    print(df.head())

    print("\nStatistics:")
    print(df.describe().round(2))