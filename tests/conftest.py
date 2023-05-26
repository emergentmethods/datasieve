import pytest
import numpy as np
import pandas as pd
from typing import Tuple
import numpy.typing as npt


# make a dummy array
def dummy_array(rows: int, cols: int, withnans: bool = True) -> np.ndarray:

    arr = np.random.rand(rows, cols)
    # fake nans
    if withnans:
        arr = np.where(arr < 0.01, np.nan, arr)

    return arr


def dummy_pandas_df(rows: int, cols: int, withnans: bool = True) -> pd.DataFrame:

    df = pd.DataFrame(np.random.rand(rows, cols)) * 35
    # fake features
    df.columns = [f"%-{col}" for col in df.columns]
    # fake label
    df = df.set_axis([*df.columns[:-1], '&-a'], axis=1)
    # fake nans
    if withnans:
        df = df.mask(df < 0.01)

    return df


def extract_features_and_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame,
                                                           pd.DataFrame]:
    """
    Extract features and labels assuming feature columns
    are prepended with "%" and label columns are
    prepended with "&"
    """
    labels = df.filter(find_labels(df))
    features = df.filter(find_features(df))

    return features, labels


def find_labels(dataframe: pd.DataFrame) -> list:
    column_names = dataframe.columns
    labels = [c for c in column_names if c.startswith("&")]
    return labels


def find_features(dataframe: pd.DataFrame) -> list:
    column_names = dataframe.columns
    features = [c for c in column_names if c.startswith("%")]
    return features


def set_weights_higher_recent(num_weights: int, wfactor: float) -> npt.ArrayLike:
    """
    Set weights so that recent data is more heavily weighted during
    training than older data.
    """
    weights = np.exp(-np.arange(num_weights) / (wfactor * num_weights))[::-1]
    return weights


@pytest.fixture(scope="function")
def dummy_array_without_nans():
    """
    Fixture for a dummy array used to test the ML pipeline
    """
    np.random.seed(seed=20)
    return dummy_array(rows=100, cols=200, withnans=False)


@pytest.fixture(scope="function")
def dummy_array2_without_nans():
    """
    Fixture for a dummy array used to test the ML pipeline
    """
    np.random.seed(seed=19)
    return dummy_array(rows=100, cols=200, withnans=False)


@pytest.fixture(scope="function")
def dummy_df_with_nans():
    """
    Fixture for a dummy pandas df used to test the ML pipeline
    """
    np.random.seed(seed=20)
    return dummy_pandas_df(rows=100, cols=200, withnans=True)


@pytest.fixture(scope="function")
def dummy_df_without_nans():
    """
    Fixture for a dummy pandas df used to test the ML pipeline
    """
    np.random.seed(seed=20)
    return dummy_pandas_df(rows=100, cols=200, withnans=False)
