import pandas as pd
import pytest

from ml.data.validator import validate_dataset


def test_valid_dataset():

    df = pd.DataFrame({
        "feature": [1, 2, 3],
        "target": [0, 1, 0],
    })

    validate_dataset(df, "target")


def test_empty_dataset():

    df = pd.DataFrame()

    with pytest.raises(ValueError, match="Dataset is empty"):
        validate_dataset(df, "target")


def test_missing_target():

    df = pd.DataFrame({
        "feature": [1, 2, 3],
    })

    with pytest.raises(
        ValueError,
        match="Target column 'target' not found",
    ):
        validate_dataset(df, "target")


def test_missing_target_values():

    df = pd.DataFrame({
        "feature": [1, 2, 3],
        "target": [0, None, 1],
    })

    with pytest.raises(
        ValueError,
        match="contains missing values",
    ):
        validate_dataset(df, "target")