"""Preprocessing pipeline tests."""

import numpy as np
import pandas as pd

from preprocess import build_preprocessor, NUMERIC_FEATURES


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "MedInc": [3.5, 5.0, 2.1],
        "HouseAge": [15, 30, 45],
        "AveRooms": [5.0, 6.2, 4.1],
        "AveBedrms": [1.0, 1.1, 0.9],
        "Population": [800, 1200, 500],
        "AveOccup": [3.0, 2.5, 3.8],
        "Latitude": [34.05, 37.77, 32.71],
        "Longitude": [-118.24, -122.42, -117.16],
    })


def test_preprocessor_outputs_correct_shape():
    pre = build_preprocessor()

    df = _sample_df()

    out = pre.fit_transform(df)

    assert out.shape == (3, len(NUMERIC_FEATURES))


def test_preprocessor_scales_to_zero_mean():
    pre = build_preprocessor()

    df = _sample_df()

    out = pre.fit_transform(df)

    assert np.allclose(out.mean(axis=0), 0, atol=1e-6)