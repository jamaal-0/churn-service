import pandas as pd

from ml.schemas.dataset import DatasetSchema


def infer_schema(
    df: pd.DataFrame,
    target: str,
    drop_columns: list[str],
) -> DatasetSchema:

    feature_columns = [
        column
        for column in df.columns
        if column != target
        and column not in drop_columns
    ]

    numerical_features = [
        column
        for column in feature_columns
        if pd.api.types.is_numeric_dtype(df[column])
    ]

    categorical_features = [
        column
        for column in feature_columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]

    return DatasetSchema(
        target=target,
        numerical_features=numerical_features,
        categorical_features=categorical_features,
        dropped_features=drop_columns,
    )