from sklearn.linear_model import SGDOneClassSVM
from datasieve.transforms.base_transform import BaseTransform
from datasieve.utils import remove_outliers
import logging
import numpy as np

logger = logging.getLogger('datasieve.pipeline')


class SVMOutlierExtractor(BaseTransform):
    """
    A transform that uses SGDOneClassSVM to detect and remove outlier datapoints
    from X, and follows through to remove the same points from y and sample_weights
    """

    def __init__(self, **kwargs):
        self._skl = SGDOneClassSVM(**kwargs)

    def fit_transform(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        self.fit(X, y, sample_weight=sample_weight)
        return self.transform(X, y, sample_weight, feature_list)

    def fit(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        self._skl.fit(X, y=y, sample_weight=sample_weight)
        return X, y, sample_weight, feature_list

    def transform(self, X, y=None, sample_weight=None,
                  feature_list=None, outlier_check=False, **kwargs):
        y_pred = self._skl.predict(X)
        y_pred = np.where(y_pred == -1, 0, y_pred)
        if not outlier_check:
            X, y, sample_weight = remove_outliers(X, y, sample_weight, y_pred)
            num_tossed = len(y_pred) - len(X)
            if num_tossed > 0:
                logger.info(
                    f"SVM detected {num_tossed} data points "
                    "as outliers."
                )
        else:
            y += y_pred
            y -= 1

        return X, y, sample_weight, feature_list
