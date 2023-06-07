import logging
import numpy as np
import numpy.typing as npt
from joblib import parallel_backend
from sklearn import cluster
from sklearn.neighbors import NearestNeighbors
from datasieve.utils import remove_outliers
from datasieve.transforms.base_transform import BaseTransform

logger = logging.getLogger('datasieve.pipeline')


class DBSCAN(BaseTransform):
    """
    A custom DBSCAN transform with a fit, transform, fit_transform and
    inverse_transform all take the full set of params X, y, sample_weight (even if they
    are unused) to follow the Pipeline API.

    fit() automatically finds the optimal epsilon and min_samples for a set of train_features
    transform() appends datapoints to the train_features, using the same epsilon and
    min_samples as computed in fit, to then determing if any of the appended data points
    are outliers.
    """

    def __init__(self, backend="loky", n_jobs=-1, **kwargs) -> None:
        self._skl: cluster.DBSCAN = cluster.DBSCAN(**kwargs)
        self.train_features: npt.ArrayLike = np.array([])
        self.backend = backend
        self.n_jobs = n_jobs

    def fit_transform(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        self.fit(X, y, sample_weight)

        # in a fit_transform() situation, the user only wants
        # to get the outliers assocciated with the train_features.
        # In contrast, a user in the future wants to use the
        # "fit" dbscan to check for outliers on incoming points
        # which makes use of self.transform(). However, self.transform()
        # appends X to the self.train_features in order to determine
        # outliers, so we avoid that duplication by ensuring that
        # fit_transform simply uses the primary train_features only.
        inliers = np.where(self._skl.labels_ == -1, 0, 1)

        X, y, sample_weight = remove_outliers(X, y, sample_weight, inliers)

        logger.info(
            f"DBSCAN tossed {len(inliers) - X.shape[0]}"
            f" train points from {len(self._skl.labels_)} in fit_transform()"
        )

        return X, y, sample_weight, feature_list

    def fit(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        """
        Given a set of training features, find the best
        epsilond and min_samples
        """
        self._skl.eps, self._skl.min_samples = self.compute_epsilon_and_minpts(X)
        logger.info(f"Found eps {self._skl.eps} and min_samples {self._skl.eps} in fit")

        with parallel_backend(self.backend, n_jobs=self.n_jobs):
            self._skl.fit(X)

        self.train_features = X

        return X, y, sample_weight, feature_list

    def transform(self, X, y=None, sample_weight=None,
                  feature_list=None, outlier_check=False, **kwargs):
        """
        Given a data point (or data points), append them to the
        train_features and determine if they are inliers.
        """

        num_X = X.shape[0]
        fullX = np.concatenate([self.train_features, X], axis=0)

        with parallel_backend(self.backend, n_jobs=self.n_jobs):
            logger.info(f"Using eps {self._skl.eps} and min_samples"
                        f"{self._skl.min_samples} to transform")
            clustering = self._skl.fit(fullX)

        inliers = np.where(clustering.labels_[-num_X:] == -1, 0, 1)

        if not outlier_check:
            X, y, sample_weight = remove_outliers(X, y, sample_weight, inliers=inliers)
            logger.info(
                f"DBSCAN tossed {len(inliers) - X.shape[0]}"
                f" train points from {len(self._skl.labels_)} in transform()"
            )
        else:
            y += inliers
            y -= 1

        return X, y, sample_weight, feature_list

    def compute_epsilon_and_minpts(self, X):
        """
        Automatically compute the epsilon and min_samples for
        "fitting" the DBSCAN.
        """
        def normalise_distances(distances):
            normalised_distances = (distances - distances.min()) / \
                (distances.max() - distances.min())
            return normalised_distances

        def rotate_point(origin, point, angle):
            # rotate a point counterclockwise by a given angle (in radians)
            # around a given origin
            x = origin[0] + np.cos(angle) * (point[0] - origin[0]) - \
                np.sin(angle) * (point[1] - origin[1])
            y = origin[1] + np.sin(angle) * (point[0] - origin[0]) + \
                np.cos(angle) * (point[1] - origin[1])
            return (x, y)

        MinPts = int(X.shape[0] * 0.25)

        # measure pairwise distances to nearest neighbours
        with parallel_backend(self.backend, n_jobs=self.n_jobs):
            neighbors = NearestNeighbors(n_neighbors=MinPts)
            neighbors_fit = neighbors.fit(X)
            distances, _ = neighbors_fit.kneighbors(X)

        distances = np.sort(distances, axis=0).mean(axis=1)
        normalised_distances = normalise_distances(distances)
        x_range = np.linspace(0, 1, len(distances))
        line = np.linspace(normalised_distances[0],
                           normalised_distances[-1], len(normalised_distances))
        deflection = np.abs(normalised_distances - line)
        max_deflection_loc = np.where(deflection == deflection.max())[0][0]
        origin = x_range[max_deflection_loc], line[max_deflection_loc]
        point = x_range[max_deflection_loc], normalised_distances[max_deflection_loc]
        rot_angle = np.pi / 4
        elbow_loc = rotate_point(origin, point, rot_angle)

        epsilon = elbow_loc[1] * (distances[-1] - distances[0]) + distances[0]

        return epsilon, MinPts
