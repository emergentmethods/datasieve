import logging
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from datasieve.transforms.base_transform import BaseTransform

logger = logging.getLogger('datasieve.pipeline')


class DataSieveVarianceThreshold(BaseTransform):

    def __init__(self, **kwargs) -> None:
        self._skl: VarianceThreshold = VarianceThreshold(**kwargs)
        self.feature_list: list = []
        self.mask = None

    def fit_transform(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        self.fit(X, y, sample_weight, feature_list)
        return self.transform(X, y, sample_weight, feature_list)

    def fit(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        self._skl.fit(X)
        self.mask = self._skl.get_support()
        if feature_list is not None:
            self.feature_list = np.array(feature_list)[self.mask]
            logger.info("Variance will remove features "
                        f"{len(feature_list) - len(self.feature_list)} "
                        f"on transform. {np.array(feature_list)[~self.mask]}")
        else:
            self.feature_list = None

        return X, y, sample_weight, self.feature_list

    def transform(self, X, y=None, sample_weight=None, outlier_check=False, feature_list=None):

        # use mask to filter X array
        X = X[:, self.mask]

        return X, y, sample_weight, self.feature_list

    def inverse_transform(self, X, y=None, sample_weight=None, feature_list=None):
        return X, y, sample_weight, feature_list
