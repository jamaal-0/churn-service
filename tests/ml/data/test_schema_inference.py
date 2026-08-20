import pandas as pd

from ml.data.schema_inference import infer_schema


def test_infer_schema():

    df = pd.DataFrame({
        "age": [20, 30, 40],
        "income": [30000, 40000, 50000],
        "employment": ["employed", "self-employed", "employed"],
        "customer_id": ["A", "B", "C"],
        "target": [0, 1, 0],
    })

    schema = infer_schema(
        df=df,
        target="target",
        drop_columns=["customer_id"],
    )

    assert schema.target == "target"

    assert schema.numerical_features == [
        "age",
        "income",
    ]

    assert schema.categorical_features == [
        "employment",
    ]

    assert schema.dropped_features == [
        "customer_id",
    ]