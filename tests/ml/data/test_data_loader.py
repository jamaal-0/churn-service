import pandas as pd
import pytest

from ml.data.loader import load_data


def test_load_data(tmp_path):

    data_path = tmp_path / "test.csv"

    data_path.write_text(
        "feature,target\n"
        "10,0\n"
        "20,1\n"
        "30,0\n"
    )

    df = load_data(data_path)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 2)
    assert list(df.columns) == [
        "feature",
        "target",
    ]


def test_load_data_missing_file(tmp_path):

    data_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_data(data_path)


def test_load_data_path_is_directory(tmp_path):

    data_path = tmp_path / "dataset"

    data_path.mkdir()

    with pytest.raises(ValueError):
        load_data(data_path)