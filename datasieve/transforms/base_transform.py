from abc import ABC


class BaseTransform(ABC):
    """
    Base class for all transforms.
    """

    def __init__(self, name: str):
        self.name = name

    def fit(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        return X, y, sample_weight, feature_list

    def transform(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        return X, y, sample_weight, feature_list

    def fit_transform(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        return X, y, sample_weight, feature_list

    def inverse_transform(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        return X, y, sample_weight, feature_list
