from joblib import parallel_backend
import pandas as pd
from sklearn.metrics import pairwise_distances
import numpy as np
import logging
import numpy.typing as npt
from typing import Union

logger = logging.getLogger('datasieve.utils')


def remove_outliers(X: npt.ArrayLike,
                    y: Union[npt.ArrayLike, None] = None,
                    sample_weight: Union[npt.ArrayLike, None] = None,
                    inliers: Union[npt.ArrayLike, None] = None):
    """
    Utility that takes 3 arrays and the outlier detection
    to remove data points equally across the 3 arrays
    and return them.

    :param X: Primary array where outlier points should be removed
    :param y: secondary array that should have the same points as the
    X array removed
    :param sample_weight: tertiary array that should have the same points
    as X array removed
    :param inliers: vector of inliers (inliers are 1, outliers are others)
    """
    X = X[(inliers == 1)]
    if y is not None:
        y = y[(inliers == 1)]
    if sample_weight is not None:
        sample_weight = sample_weight[(inliers == 1)]

    return X, y, sample_weight


def find_training_horizon(df: pd.DataFrame, target_horizon, test_pct=0.05,
                          threshold=0.25e-3, backend="loky", n_jobs=-1):
    """
    Given a set of raw data, determine the necessariy training horizon
    associated to the target horizon
    """
    step_size = 1
    change_window = 100
    std_ratio = np.array([])
    max_window = df.shape[0] - target_horizon
    if isinstance(test_pct, int):
        test_size = test_pct
    else:
        test_size = int(df.shape[0] * test_pct)
    horizon_features = df.iloc[-target_horizon:]

    for t in np.arange(0, max_window, step_size):
        current_window = df.iloc[-target_horizon-t:]
        with parallel_backend(backend, n_jobs=n_jobs):
            current_window_distances = pairwise_distances(
                current_window, metric="euclidean", n_jobs=8)
            # remove the diagonal distances which are itself distances ~0
            # np.fill_diagonal(current_window_distances, np.NaN)
            current_window_distances = current_window_distances.reshape(-1, 1)
            std_train_dist = current_window_distances[~np.isnan(current_window_distances)].std()
            distances_horizon_current_window = pairwise_distances(
                current_window, horizon_features, metric="euclidean", n_jobs=8)

        distances_horizon_current_window = distances_horizon_current_window.reshape(-1, 1)
        di_std = distances_horizon_current_window.std() / std_train_dist
        std_ratio = np.append(std_ratio, di_std)
        if t > change_window:
            change = np.mean(np.abs(np.diff(std_ratio[-change_window:])))
            if change < threshold:
                if t + test_size > df.shape[0]:
                    logger.warning(f"Training horizon {t} + test size {test_size} greater than "
                                   f"data size {df.shape[0]}")
                    return df.shape[0]
                else:
                    logger.info(f"Found training horizon of {t}.")
                return t + test_size

    logger.warning("Could not find training horizon. Using full data set.")
    return df.shape[0]
