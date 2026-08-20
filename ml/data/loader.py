from pathlib import Path

import pandas as pd


def load_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    if not path.is_file():
        raise ValueError(
            f"Dataset path is not a file: {path}"
        )

    return pd.read_csv(path)