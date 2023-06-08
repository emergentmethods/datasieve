from sklearn import decomposition
from datasieve.transforms.base_transform import BaseTransform
import logging
import numpy as np

logger = logging.getLogger('datasieve.pipeline')


class PCA(BaseTransform):
    """
    A PCA that ensures the feature names are properly transformed and follow
    along with the X throughout the pipeline.
    """

    def __init__(self, **kwargs):
        self._skl: decomposition.PCA = decomposition.PCA(**kwargs)

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
