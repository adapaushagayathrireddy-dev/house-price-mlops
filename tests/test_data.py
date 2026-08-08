"""Data quality tests."""

from data_loader import load_data


def test_dataset_is_not_empty():
    df = load_data()
    assert len(df) > 0, "Dataset must not be empty"


def test_dataset_has_expected_columns():
    df = load_data()

    expected = {
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
        "target",
    }

    assert expected.issubset(df.columns), (
        f"Missing columns: {expected - set(df.columns)}"
    )


def test_no_missing_values():
    df = load_data()
    assert df.isna().sum().sum() == 0, "Dataset should not have missing values"


def test_target_is_positive():
    df = load_data()
    assert (df["target"] > 0).all(), "House prices must be positive"


def test_reasonable_row_count():
    df = load_data()
    assert 20_000 < len(df) < 21_000