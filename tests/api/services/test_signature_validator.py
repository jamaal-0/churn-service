import pytest
import pandas as pd

from mlflow.models import ModelSignature
from mlflow.types import Schema, ColSpec

from api.services.signature_validator import (
    SignatureValidator,
)


@pytest.fixture
def signature():

    input_schema = Schema(
        [
            ColSpec("string", "gender"),
            ColSpec("long", "SeniorCitizen"),
            ColSpec("long", "tenure"),
            ColSpec("double", "MonthlyCharges"),
        ]
    )

    return ModelSignature(
        inputs=input_schema,
    )


@pytest.fixture
def validator(signature):

    return SignatureValidator(signature)


def test_valid_input_returns_dataframe(
    validator,
):

    features = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "tenure": 12,
        "MonthlyCharges": 70.5,
    }

    result = validator.validate(features)

    assert isinstance(result, pd.DataFrame)

    assert list(result.columns) == [
        "gender",
        "SeniorCitizen",
        "tenure",
        "MonthlyCharges",
    ]

    assert result.iloc[0]["gender"] == "Female"
    assert result.iloc[0]["SeniorCitizen"] == 0
    assert result.iloc[0]["tenure"] == 12
    assert result.iloc[0]["MonthlyCharges"] == 70.5


def test_missing_feature_raises_error(
    validator,
):

    features = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "tenure": 12,
    }

    with pytest.raises(
        ValueError,
        match="Missing required features",
    ):
        validator.validate(features)


def test_extra_feature_raises_error(
    validator,
):

    features = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "tenure": 12,
        "MonthlyCharges": 70.5,
        "UnexpectedFeature": 123,
    }

    with pytest.raises(
        ValueError,
        match="Unexpected features",
    ):
        validator.validate(features)


def test_invalid_integer_type_raises_error(
    validator,
):

    features = {
        "gender": "Female",
        "SeniorCitizen": "yes",
        "tenure": 12,
        "MonthlyCharges": 70.5,
    }

    with pytest.raises(
        ValueError,
        match="Invalid type for feature 'SeniorCitizen'",
    ):
        validator.validate(features)


def test_invalid_float_type_raises_error(
    validator,
):

    features = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "tenure": 12,
        "MonthlyCharges": "expensive",
    }

    with pytest.raises(
        ValueError,
        match="Invalid type for feature 'MonthlyCharges'",
    ):
        validator.validate(features)


def test_invalid_string_type_raises_error(
    validator,
):

    features = {
        "gender": 123,
        "SeniorCitizen": 0,
        "tenure": 12,
        "MonthlyCharges": 70.5,
    }

    with pytest.raises(
        ValueError,
        match="Invalid type for feature 'gender'",
    ):
        validator.validate(features)


def test_missing_signature_raises_error():

    validator = SignatureValidator(None)

    with pytest.raises(
        RuntimeError,
        match="Model signature has not been loaded",
    ):
        validator.validate({})