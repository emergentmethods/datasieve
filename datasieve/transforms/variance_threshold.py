import logging
import numpy as np
from sklearn.feature_selection import VarianceThreshold

logger = logging.getLogger('datasieve.pipeline')


class DataSieveVarianceThreshold(VarianceThreshold):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.feature_list: list = []
        self.mask = None

    def fit_transform(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        self.fit(X, y, sample_weight, feature_list)
        return self.transform(X, y, sample_weight, feature_list)

    def fit(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        super().fit(X)
        self.mask = self.get_support()
        self.feature_list = np.array(feature_list)[self.mask]
        logger.info(f"Variance will remove features {len(feature_list) - len(self.feature_list)} "
                    f"on transform. {np.array(feature_list)[~self.mask]}")

        return X, y, sample_weight, self.feature_list

    def transform(self, X, y=None, sample_weight=None, feature_list=None):

        # use mask to filter X array
        X = X[:, self.mask]

        return X, y, sample_weight, self.feature_list

    def inverse_transform(self, X, y=None, sample_weight=None, feature_list=None):
        return X, y, sample_weight, feature_list
