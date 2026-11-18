from ml.pipeline import TrainingPipeline
from ml.registry.model_registry import register_model
from ml.selection.model_selector import select_best_run

def main():

    pipeline = TrainingPipeline()

    _ = pipeline.run_pipeline(
        "./ml/config/telco.yaml"
    )

    best_run_id = select_best_run(
        experiment_name="churn-service",
        primary_metric="roc_auc",
        constraints={
            "recall": 0.55,
        },
    )

    _ = register_model(
        run_id=best_run_id,
        model_name="churn-model_test",
    )

if __name__ == "__main__":
    main()