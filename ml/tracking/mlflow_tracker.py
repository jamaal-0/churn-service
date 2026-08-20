from typing import Any

import mlflow
import mlflow.sklearn


def start_run(
    tracking_uri: str,
    experiment_name: str,
    run_name: str | None = None,
):

    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)

    return mlflow.start_run(
        run_name=run_name
    )


def log_training_run(
    model: Any,
    parameters: dict,
    metrics: dict[str, float],
) -> None:

    mlflow.log_params(parameters)
    mlflow.log_metrics(metrics)

    mlflow.sklearn.log_model(
        model,
        name="model",
        skops_trusted_types=["numpy.dtype"],
    )