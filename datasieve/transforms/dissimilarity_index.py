import logging
from sklearn.metrics.pairwise import pairwise_distances
from datasieve.transforms.base_transform import BaseTransform
import numpy as np
import numpy.typing as npt
from joblib import parallel_backend
from datasieve.utils import remove_outliers

logger = logging.getLogger('datasieve.pipeline')


class DissimilarityIndex(BaseTransform):
    """
    Object designed for computing the dissimilarity index for a set of training data and
    prediction points. fit() computes the avg_mean distance for the training data and
    stores the data for future "transforms" transform() uses the `di_threshold` to
    identify and remove outliers
    """

    def __init__(self, di_threshold: float = 1, n_jobs=-1, backend="loky", **kwargs):
        self.avg_mean_dist: float = 0
        self.trained_data: npt.ArrayLike = np.array([])
        self.di_threshold = di_threshold
        self.di_values: npt.ArrayLike = np.array([])
        self.n_jobs = n_jobs
        self.backend = backend

    def fit(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        """
        Compute the distances, save the average mean distance
        and save the trained_data array for future use
        """

        with parallel_backend(self.backend, n_jobs=self.n_jobs):
            pairwise = pairwise_distances(X)

        # remove the diagonal distances which are itself distances ~0
        np.fill_diagonal(pairwise, np.NaN)
        pairwise = pairwise.reshape(-1, 1)
        self.avg_mean_dist = pairwise[~np.isnan(pairwise)].mean()
        self.trained_data = X

        return X, y, sample_weight, feature_list

    def transform(self, X, y=None, sample_weight=None,
                  feature_list=None, outlier_check=False, **kwargs):
        """
        Compares the distance from each prediction point to each training data
        point. It uses this information to estimate a Dissimilarity Index (DI)
        and avoid making predictions on any points that are too far away
        from the training data set.
        """

        with parallel_backend(self.backend, n_jobs=self.n_jobs):
            distance = pairwise_distances(self.trained_data, X)

        self.di_values = distance.min(axis=0) / self.avg_mean_dist
        y_pred = np.where(self.di_values < self.di_threshold, 1, 0)

        if not outlier_check:
            X, y, sample_weight = remove_outliers(X, y, sample_weight, y_pred)
        else:
            y += y_pred
            y -= 1

        num_tossed = len(y_pred) - len(X)
        if num_tossed > 0:
            logger.info(
                f"DI tossed {num_tossed} predictions for "
                "being too far from training data."
            )

        return X, y, sample_weight, feature_list
