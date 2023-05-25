from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger('datasieve.pipeline')


class DataSieveMinMaxScaler(MinMaxScaler):
    """
    A subclass of the SKLearn MinMaxScaler that ensures fit, transform, fit_transform and
    inverse_transform all take the full set of params X, y, sample_weight (even if they
    are unused) to follow the FlowdaptPipeline API.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit_transform(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        super().fit(X)
        X = super().transform(X)
        return X, y, sample_weight, feature_list

    def fit(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        super().fit(X)
        return X, y, sample_weight, feature_list

    def transform(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        X = super().transform(X)
        return X, y, sample_weight, feature_list

    def inverse_transform(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        return super().inverse_transform(X), y, sample_weight, feature_list
