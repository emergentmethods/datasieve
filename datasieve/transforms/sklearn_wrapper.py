from datasieve.transforms.base_transform import BaseTransform
from sklearn.base import BaseEstimator
from joblib import parallel_backend

class SKLearnWrapper(BaseTransform):
    """
    Wrapper that takes *most* SKLearn transforms and allows them to
    work wiith the datasieve pipeline
    """
    def __init__(self, sklearninstance: BaseEstimator, n_jobs=-1, backend="loky", **kwargs):
        self.backend = backend
        self.n_jobs = n_jobs
        self._skl = sklearninstance

    def fit(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        with parallel_backend(self.backend, n_jobs=self.n_jobs):
            self._skl = self._skl.fit(X, y=y)
        return X, y, sample_weight, feature_list

    def transform(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        with parallel_backend(self.backend, n_jobs=self.n_jobs):
            X = self._skl.transform(X)
        return X, y, sample_weight, feature_list

    def fit_transform(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        self.fit(X, y=y, sample_weight=sample_weight, feature_list=feature_list)
        X, y, sample_weight, feature_list = self.transform(X, y=y, sample_weight=sample_weight,
                                                           feature_list=feature_list)
        return X, y, sample_weight, feature_list

    def inverse_transform(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        X = self._skl.inverse_transform(X)
        return X, y, sample_weight, feature_list
