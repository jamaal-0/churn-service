from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from ml.preprocessing.builder import build_preprocessor
from ml.schemas.dataset import DatasetSchema


def create_model(
    model_type: str,
    parameters: dict[str, Any],
):
    if model_type == "logistic_regression":
        return LogisticRegression(**parameters, random_state=42)

    raise ValueError(
        f"Unsupported model type: {model_type}"
    )


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    schema: DatasetSchema,
    model_type: str,
    model_parameters: dict[str, Any],
) -> Pipeline:

    preprocessor = build_preprocessor(schema)

    model = create_model(
        model_type=model_type,
        parameters=model_parameters,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    return pipeline