"""Pydantic schemas for API requests and responses."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HouseFeatures(BaseModel):
    """
    Input features required for house price prediction.
    """

    MedInc: float = Field(
        ...,
        ge=0,
        description="Median income (in tens of thousands)"
    )

    HouseAge: float = Field(
        ...,
        ge=0,
        le=100
    )

    AveRooms: float = Field(
        ...,
        ge=0
    )

    AveBedrms: float = Field(
        ...,
        ge=0
    )

    Population: float = Field(
        ...,
        ge=0
    )

    AveOccup: float = Field(
        ...,
        ge=0
    )

    Latitude: float = Field(
        ...,
        ge=32.0,
        le=42.0
    )

    Longitude: float = Field(
        ...,
        ge=-125.0,
        le=-114.0
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "MedInc": 5.0,
                "HouseAge": 20,
                "AveRooms": 6.0,
                "AveBedrms": 1.0,
                "Population": 1200,
                "AveOccup": 3.0,
                "Latitude": 34.05,
                "Longitude": -118.24,
            }
        }
    }


class PredictionResponse(BaseModel):
    """
    API prediction output format.
    """

    predicted_price_usd: float
    predicted_price_100k: float
    model_version: str