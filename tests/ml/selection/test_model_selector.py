from unittest.mock import patch

import pandas as pd
import pytest

from ml.selection.model_selector import select_best_run


def test_select_best_run():

    runs = pd.DataFrame(
        {
            "run_id": [
                "run_a",
                "run_b",
                "run_c",
            ],
            "metrics.roc_auc": [
                0.84,
                0.86,
                0.83,
            ],
            "metrics.recall": [
                0.60,
                0.58,
                0.70,
            ],
            "start_time": [
                1000,
                2000,
                3000,
            ],
        }
    )

    with patch(
        "ml.selection.model_selector.mlflow.search_runs",
        return_value=runs,
    ):

        result = select_best_run(
            experiment_name="churn-service",
            primary_metric="roc_auc",
            constraints={
                "recall": 0.55,
            },
        )

    assert result == "run_b"


def test_select_best_run_filters_ineligible_runs():

    runs = pd.DataFrame(
        {
            "run_id": [
                "run_a",
                "run_b",
            ],
            "metrics.roc_auc": [
                0.90,
                0.85,
            ],
            "metrics.recall": [
                0.40,
                0.60,
            ],
            "start_time": [
                1000,
                2000,
            ],
        }
    )

    with patch(
        "ml.selection.model_selector.mlflow.search_runs",
        return_value=runs,
    ):

        result = select_best_run(
            experiment_name="churn-service",
            primary_metric="roc_auc",
            constraints={
                "recall": 0.55,
            },
        )

    # run_a has higher ROC-AUC but fails the recall constraint
    assert result == "run_b"


def test_select_best_run_uses_recall_as_tiebreaker():

    runs = pd.DataFrame(
        {
            "run_id": [
                "run_a",
                "run_b",
            ],
            "metrics.roc_auc": [
                0.85,
                0.85,
            ],
            "metrics.recall": [
                0.56,
                0.65,
            ],
            "start_time": [
                1000,
                2000,
            ],
        }
    )

    with patch(
        "ml.selection.model_selector.mlflow.search_runs",
        return_value=runs,
    ):

        result = select_best_run(
            experiment_name="churn-service",
            primary_metric="roc_auc",
            constraints={
                "recall": 0.55,
            },
        )

    assert result == "run_b"


def test_select_best_run_uses_latest_run_as_final_tiebreaker():

    runs = pd.DataFrame(
        {
            "run_id": [
                "run_a",
                "run_b",
            ],
            "metrics.roc_auc": [
                0.85,
                0.85,
            ],
            "metrics.recall": [
                0.60,
                0.60,
            ],
            "start_time": [
                1000,
                2000,
            ],
        }
    )

    with patch(
        "ml.selection.model_selector.mlflow.search_runs",
        return_value=runs,
    ):

        result = select_best_run(
            experiment_name="churn-service",
            primary_metric="roc_auc",
            constraints={
                "recall": 0.55,
            },
        )

    assert result == "run_b"


def test_select_best_run_no_runs():

    runs = pd.DataFrame()

    with patch(
        "ml.selection.model_selector.mlflow.search_runs",
        return_value=runs,
    ):

        with pytest.raises(
            ValueError,
            match="No runs found",
        ):

            select_best_run(
                experiment_name="churn-service",
                primary_metric="roc_auc",
            )


def test_select_best_run_no_eligible_runs():

    runs = pd.DataFrame(
        {
            "run_id": ["run_a"],
            "metrics.roc_auc": [0.80],
            "metrics.recall": [0.40],
            "start_time": [1000],
        }
    )

    with patch(
        "ml.selection.model_selector.mlflow.search_runs",
        return_value=runs,
    ):

        with pytest.raises(
            ValueError,
            match="No runs satisfy",
        ):

            select_best_run(
                experiment_name="churn-service",
                primary_metric="roc_auc",
                constraints={
                    "recall": 0.55,
                },
            )