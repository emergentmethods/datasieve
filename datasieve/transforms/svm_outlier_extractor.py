from sklearn.linear_model import SGDOneClassSVM
from datasieve.utils import remove_outliers
import logging

logger = logging.getLogger('datasieve.pipeline')


class SVMOutlierExtractor(SGDOneClassSVM):
    """
    A subclass of the SKLearn SGDOneClassSVM that adds a transform() method
    for removing detected outliers from X (as well as the associated y and
    sample_weight if they are also furnished.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit_transform(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        self.fit(X, y, sample_weight=sample_weight)
        return self.transform(X, y, sample_weight=sample_weight)

    def fit(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        super().fit(X, y=y, sample_weight=sample_weight)
        return X, y, sample_weight, feature_list

    def transform(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        y_pred = self.predict(X)

        X, y, sample_weight = remove_outliers(X, y, sample_weight, y_pred)

        num_tossed = len(y_pred) - len(X)
        if num_tossed > 0:
            logger.info(
                f"SVM detected {num_tossed} data points"
                "as outliers."
            )

        return X, y, sample_weight, feature_list

    def inverse_transform(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        """
        Unused
        """
        return X, y, sample_weight, feature_list
