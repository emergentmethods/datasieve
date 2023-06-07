from datasieve.transforms.base_transform import BaseTransform
import numpy as np


class Noise(BaseTransform):
    """
    Add noise to the train features only. Anything that passes through `transform` remains
    untouched. This makes this step unique in the sense that `fit_transform()` is the
    only way to apply noise to the train features.
    """
    def __init__(self, mu=0, sigma=0.01, **kwargs):
        self.mu = mu
        self.sigma = sigma
        self.noise = np.array([])

    def fit(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        """
        During the fit we do not expect any transformation to occur
        """
        self.noise = np.random.normal(self.mu, self.sigma, [X.shape[0], X.shape[1]])
        return X, y, sample_weight, feature_list

    def fit_transform(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        """
        We only want the train features to have noise added to them, so during a fit_transform
        we add noise.
        """
        self.fit(X)
        X += self.noise
        return X, y, sample_weight, feature_list

    def transform(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        """
        We do not add any transformation when this is applied to test features
        """
        return X, y, sample_weight, feature_list
