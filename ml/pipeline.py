import pandas as pd
from sklearn.model_selection import train_test_split

from ml.config.loader import load_config
from ml.data.loader import load_data
from ml.data.validator import validate_dataset
from ml.data.schema_inference import infer_schema
from ml.training.trainer import train_model
from ml.evaluation.evaluator import evaluate_model
from ml.tracking.mlflow_tracker import (
    start_run,
    log_training_run,
)

class TrainingPipeline():
    def __init__(self):
        pass


    def run_pipeline(self, config_path: str) -> dict[str, float]:

        # 1. Load configuration
        config = load_config(config_path)

        # 2. Load dataset
        df = load_data(config.data.path)

        # 3. Validate dataset
        validate_dataset(
            df=df,
            target=config.data.target,
        )

        # 4. Infer dataset schema
        schema = infer_schema(
            df=df,
            target=config.data.target,
            drop_columns=config.data.drop_columns,
        )

        # 5. Separate features and target
        X = df.drop(columns=config.data.target)
        y = df[config.data.target]

        # 6. Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=config.split.test_size,
            random_state=config.split.random_state,
            stratify=y,
        )

        # 7. Start MLflow run
        with start_run(
            experiment_name=config.mlflow.experiment_name
        ):

            # 8. Train model
            model = train_model(
                X_train=X_train,
                y_train=y_train,
                schema=schema,
                model_type=config.model.type,
                model_parameters=config.model.parameters,
            )

            # 9. Evaluate model
            metrics = evaluate_model(
                model=model,
                X_test=X_test,
                y_test=y_test,
            )

            # 10. Log results to MLflow
            log_training_run(
                model=model,
                parameters={
                    "model_type": config.model.type,
                    **config.model.parameters,
                },
                metrics=metrics,
            )

        return metrics