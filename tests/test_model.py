"""Model sanity tests using a tiny quick training run."""

from sklearn.linear_model import LinearRegression

from data_loader import load_data, split_data
from train import build_model_pipeline, evaluate


def test_pipeline_can_fit_and_predict():
    df = load_data().sample(1000, random_state=0)

    X_train, X_test, y_train, y_test = split_data(df)

    pipeline = build_model_pipeline(LinearRegression())

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    assert len(preds) == len(y_test)


def test_pipeline_r2_above_threshold():
    """
    Even on 1k rows, R² should be > 0.4
    """

    df = load_data().sample(1000, random_state=0)

    X_train, X_test, y_train, y_test = split_data(df)

    pipeline = build_model_pipeline(LinearRegression())

    pipeline.fit(X_train, y_train)

    metrics = evaluate(
        y_test,
        pipeline.predict(X_test)
    )

    assert metrics["r2"] > 0.4, f"R² too low: {metrics['r2']}"