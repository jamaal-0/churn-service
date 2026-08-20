import pandas as pd

from ml.schemas.dataset import DatasetSchema
from ml.training.trainer import train_model


def test_train_model():

    schema = DatasetSchema(
        target="target",
        numerical_features=["age", "income"],
        categorical_features=["employment"],
        dropped_features=[],
    )

    X_train = pd.DataFrame({
        "age": [20, 25, 30, 35, 40, 45],
        "income": [
            25000,
            30000,
            35000,
            40000,
            45000,
            50000,
        ],
        "employment": [
            "employed",
            "employed",
            "unemployed",
            "employed",
            "self-employed",
            "employed",
        ],
    })

    y_train = pd.Series([0, 0, 0, 1, 1, 1])

    model = train_model(
        X_train=X_train,
        y_train=y_train,
        schema=schema,
        model_type="logistic_regression",
        model_parameters={
            "max_iter": 1000,
        },
    )

    predictions = model.predict(X_train)

    assert len(predictions) == len(y_train)