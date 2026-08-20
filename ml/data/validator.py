import pandas as pd


def validate_dataset(
    df: pd.DataFrame,
    target: str,
) -> None:

    if df.empty:
        raise ValueError("Dataset is empty")

    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found in dataset"
        )

    if df.columns.duplicated().any():
        raise ValueError(
            "Dataset contains duplicate column names"
        )

    if df[target].isna().any():
        raise ValueError(
            f"Target column '{target}' contains missing values"
        )