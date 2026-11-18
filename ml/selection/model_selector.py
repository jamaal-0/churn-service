import mlflow
import pandas as pd


def select_best_run(
    experiment_name: str,
    primary_metric: str,
    constraints: dict[str, float] | None = None,
) -> str:

    constraints = constraints or {}

    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
    )

    if runs.empty:
        raise ValueError(
            f"No runs found in experiment '{experiment_name}'."
        )

    # Filter runs according to minimum metric constraints
    eligible_runs = runs.copy()

    for metric, minimum_value in constraints.items():

        column = f"metrics.{metric}"

        if column not in eligible_runs.columns:
            raise ValueError(
                f"Metric '{metric}' not found in experiment runs."
            )

        eligible_runs = eligible_runs[
            eligible_runs[column] >= minimum_value
        ]

    if eligible_runs.empty:
        raise ValueError(
            "No runs satisfy the selection criteria."
        )

    primary_column = f"metrics.{primary_metric}"

    if primary_column not in eligible_runs.columns:
        raise ValueError(
            f"Primary metric '{primary_metric}' not found in experiment runs."
        )

    # Deterministic ranking:
    # 1. Primary metric
    # 2. Other constraints, in the order supplied
    # 3. Most recent run as final tie-breaker
    sort_columns = [primary_column]
    ascending = [False]

    for metric in constraints:

        column = f"metrics.{metric}"

        if column != primary_column:
            sort_columns.append(column)
            ascending.append(False)

    sort_columns.append("start_time")
    ascending.append(False)

    eligible_runs = eligible_runs.sort_values(
        by=sort_columns,
        ascending=ascending,
    )

    best_run = eligible_runs.iloc[0]

    return best_run["run_id"]


# if __name__ == "__main__":
#     best_run_id = select_best_run(
#         experiment_name="churn-service",
#         primary_metric="roc_auc",
#         constraints={
#             "recall": 0.55,
#             },
#         )

#     print(best_run_id)
#     from ml.registry.model_registry import register_model

#     version = register_model(

#         run_id=best_run_id,

#         model_name="churn-model",

#     )

#     print(f"Registered model version: {version}")