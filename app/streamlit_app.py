"""Streamlit UI that calls the FastAPI backend."""

from __future__ import annotations

import os
import requests
import streamlit as st


API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000"
)


st.set_page_config(
    page_title="California House Price Predictor",
    page_icon="🏠"
)


st.title("🏠 California House Price Predictor")

st.caption(f"Backend: `{API_URL}`")


with st.sidebar:

    st.header("Enter house features")

    med_inc = st.slider(
        "Median Income (in tens of thousands $)",
        0.5,
        15.0,
        5.0,
        0.1
    )

    house_age = st.slider(
        "House Age (years)",
        1,
        60,
        20
    )

    ave_rooms = st.slider(
        "Avg Rooms per Household",
        1.0,
        15.0,
        6.0,
        0.1
    )

    ave_bedrms = st.slider(
        "Avg Bedrooms per Household",
        0.5,
        5.0,
        1.0,
        0.1
    )

    population = st.number_input(
        "Block Group Population",
        100,
        40000,
        1200
    )

    ave_occup = st.slider(
        "Avg Occupancy",
        1.0,
        10.0,
        3.0,
        0.1
    )

    latitude = st.slider(
        "Latitude",
        32.0,
        42.0,
        34.05,
        0.01
    )

    longitude = st.slider(
        "Longitude",
        -125.0,
        -114.0,
        -118.24,
        0.01
    )


payload = {
    "MedInc": med_inc,
    "HouseAge": house_age,
    "AveRooms": ave_rooms,
    "AveBedrms": ave_bedrms,
    "Population": population,
    "AveOccup": ave_occup,
    "Latitude": latitude,
    "Longitude": longitude,
}


predict_clicked = st.button(
    "🔮 Predict Price",
    type="primary"
)


if predict_clicked:

    try:

        with st.spinner("Contacting the model..."):

            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=15
            )


        if response.status_code == 200:

            data = response.json()

            st.success("Prediction complete!")

            st.metric(
                "Estimated Price",
                f"${data['predicted_price_usd']:,.0f}"
            )

            st.write(
                f"Model version: `{data['model_version']}`"
            )

        else:

            st.error(
                f"API returned {response.status_code}: {response.text}"
            )


    except requests.RequestException as e:

        st.error(
            f"Could not reach the API: {e}"
        )


with st.expander("How it works"):

    st.markdown(
        """
        This app calls a FastAPI backend that hosts a
        scikit-learn regression model trained on the
        California Housing dataset.

        The full pipeline includes training,
        evaluation, deployment, and MLflow tracking.
        """
    )