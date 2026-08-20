import numpy as np
import pandas as pd

from ml.preprocessing.builder import build_preprocessor
from ml.schemas.dataset import DatasetSchema


def test_build_preprocessor():

    schema = DatasetSchema(
        target="target",
        numerical_features=["age", "income"],
        categorical_features=["employment"],
        dropped_features=["customer_id"],
    )

    preprocessor = build_preprocessor(schema)

    df = pd.DataFrame({
        "age": [25, 30, np.nan, 40],
        "income": [30000, 40000, 35000, np.nan],
        "employment": [
            "employed",
            "self-employed",
            "employed",
            None,
        ],
        "customer_id": ["A", "B", "C", "D"],
        "target": [0, 1, 0, 1],
    })

    X = df.drop(columns="target")

    transformed = preprocessor.fit_transform(X)

    assert transformed.shape[0] == 4

    assert transformed.shape[1] > 2

    assert not np.isnan(transformed).any()


def test_preprocessor_handles_unseen_categories():

    schema = DatasetSchema(
        target="target",
        numerical_features=["age"],
        categorical_features=["employment"],
        dropped_features=[],
    )

    preprocessor = build_preprocessor(schema)

    train = pd.DataFrame({
        "age": [20, 30, 40],
        "employment": [
            "employed",
            "unemployed",
            "employed",
        ],
    })

    test = pd.DataFrame({
        "age": [35],
        "employment": ["retired"],
    })

    preprocessor.fit(train)

    transformed = preprocessor.transform(test)

    assert transformed.shape[0] == 1
    assert not np.isnan(transformed).any()