import logging
import numpy as np
from sklearn import feature_selection as fs
from datasieve.transforms.base_transform import BaseTransform

logger = logging.getLogger('datasieve.pipeline')


class VarianceThreshold(BaseTransform):
    """
    A VarianceThresholdd that removes any feature with a variance below a threshold.
    Features are removed and a feature list is held for use during subsequent transforms
    to keep feature columns consistent.
    """

    def __init__(self, **kwargs) -> None:
        self._skl: fs.VarianceThreshold = fs.VarianceThreshold(**kwargs)
        self.feature_list: list = []
        self.mask = None

    def fit(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        self._skl.fit(X)
        self.mask = self._skl.get_support()
        if feature_list is not None:
            self.feature_list = np.array(feature_list)[self.mask]
            if len(feature_list) - len(self.feature_list) > 0:
                logger.info("VarianceThreshold will remove "
                            f"{len(feature_list) - len(self.feature_list)} "
                            "features from the dataset."
                            f"on transform. {np.array(feature_list)[~self.mask]}")
        else:
            self.feature_list = None

        return X, y, sample_weight, self.feature_list

    def transform(self, X, y=None, sample_weight=None,
                  feature_list=None, outlier_check=False, **kwargs):

        # use mask to filter X array
        X = X[:, self.mask]

        return X, y, sample_weight, self.feature_list
