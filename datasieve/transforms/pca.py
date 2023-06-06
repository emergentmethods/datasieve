from sklearn.decomposition import PCA
from datasieve.transforms.base_transform import BaseTransform
import logging
import numpy as np

logger = logging.getLogger('datasieve.pipeline')


class DataSievePCA(BaseTransform):
    """
    A subclass of the SKLearn PCA that ensures fit, transform, fit_transform and
    inverse_transform all take the full set of params X, y, sample_weight (even if they
    are unused) to follow the FlowdaptPipeline API.
    """

    def __init__(self, n_components=0.9999, **kwargs):
        self._skl = PCA(n_components=n_components, **kwargs)

    def fit_transform(self, X, y=None, sample_weight=None, feature_list=None):
        X, y, sample_weight, feature_list = self.fit(X, y, sample_weight, feature_list)
        return self.transform(X, y, sample_weight, feature_list)

    def fit(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        n_components = X.shape[1]
        self._skl.fit(X)

        n_keep_components = self._skl.n_components_
        self.feature_list = [f"PC{i}" for i in range(0, n_keep_components)]
        logger.info(f"reduced feature dimension by {n_components - n_keep_components}")
        logger.info(f"explained variance {np.sum(self._skl.explained_variance_ratio_)}")
        return X, y, sample_weight, self.feature_list

    def transform(self, X, y=None, sample_weight=None,
                  outlier_check=False, feature_list=None, **kwargs):
        X = self._skl.transform(X)
        return X, y, sample_weight, self.feature_list

    def inverse_transform(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        return self._skl.inverse_transform(X), y, sample_weight, feature_list
