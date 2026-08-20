import pandas as pd

from sklearn.linear_model import LogisticRegression

from ml.evaluation.evaluator import evaluate_model


def test_evaluate_model():

    X_train = pd.DataFrame({
        "feature": [1, 2, 3, 4, 5, 6],
    })

    y_train = pd.Series([0, 0, 0, 1, 1, 1])

    X_test = pd.DataFrame({
        "feature": [1.5, 2.5, 4.5, 5.5],
    })

    y_test = pd.Series([0, 0, 1, 1])

    model = LogisticRegression()

    model.fit(X_train, y_train)

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    assert isinstance(metrics, dict)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics

    for value in metrics.values():
        assert 0 <= value <= 1