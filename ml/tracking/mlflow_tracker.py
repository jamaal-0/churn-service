from typing import Any

import mlflow
import mlflow.sklearn


def start_run(
    experiment_name: str,
    run_name: str | None = None,
) -> Any:

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
    )