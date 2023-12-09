import pandas as pd
from sqlalchemy import ScalarResult


def create_dataframe(values: ScalarResult, selected_fields: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([{f: getattr(r, f) for f in selected_fields} for r in values])
    df.shape
    return df


def split_X_y(
    dataset_shape: tuple[int, int], training_set: pd.DataFrame, padding: int
) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    X: list[pd.DataFrame] = []
    y: list[pd.DataFrame] = []

    for i in range(padding, dataset_shape[0]):
        X.append(training_set.iloc[i - padding : i])
        y.append(training_set.iloc[[i]])

    return (X, pd.concat(y))
