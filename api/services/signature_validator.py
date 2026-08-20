from typing import Any

import pandas as pd
from mlflow.models import ModelSignature
from mlflow.types import DataType


class SignatureValidator:

    def __init__(
        self,
        signature: ModelSignature,
    ):
        self.signature = signature

    def validate(
        self,
        features: dict[str, Any],
    ) -> pd.DataFrame:

        if self.signature is None:
            raise RuntimeError(
                "Model signature has not been loaded."
            )

        expected_inputs = self.signature.inputs.inputs

        expected_columns = [
            field.name
            for field in expected_inputs
        ]

        actual_columns = list(features.keys())

        # Check for missing features
        missing_columns = (
            set(expected_columns)
            - set(actual_columns)
        )

        if missing_columns:
            raise ValueError(
                f"Missing required features: "
                f"{sorted(missing_columns)}"
            )

        # Check for unexpected features
        extra_columns = (
            set(actual_columns)
            - set(expected_columns)
        )

        if extra_columns:
            raise ValueError(
                f"Unexpected features: "
                f"{sorted(extra_columns)}"
            )

        # Validate Python values against MLflow types
        for field in expected_inputs:

            value = features[field.name]

            if not self._is_valid_type(
                value,
                field.type,
            ):
                raise ValueError(
                    f"Invalid type for feature "
                    f"'{field.name}'. Expected "
                    f"'{field.type}'."
                )

        return pd.DataFrame(
            [features],
            columns=expected_columns,
        )

    @staticmethod
    def _is_valid_type(
        value: Any,
        expected_type: DataType,
    ) -> bool:

        if expected_type in {
            DataType.integer,
            DataType.long,
        }:
            return (
                isinstance(value, int)
                and not isinstance(value, bool)
            )

        if expected_type in {
            DataType.float,
            DataType.double,
        }:
            return (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
            )

        if expected_type == DataType.boolean:
            return isinstance(value, bool)

        if expected_type == DataType.string:
            return isinstance(value, str)

        # For unsupported MLflow types, don't silently
        # claim that validation succeeded.
        raise ValueError(
            f"Unsupported MLflow input type: "
            f"{expected_type}"
        )