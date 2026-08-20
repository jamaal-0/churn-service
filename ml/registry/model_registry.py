import mlflow


def register_model(
    run_id: str,
    model_name: str,
) -> str:

    model_uri = f"runs:/{run_id}/model"

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    return registered_model.version