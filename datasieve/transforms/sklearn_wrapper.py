from datasieve.transforms.base_transform import BaseTransform
from sklearn.base import BaseEstimator
from joblib import parallel_backend


class SKLearnWrapper(BaseTransform):
    """
    Wrapper that takes *most* SKLearn transforms and allows them to
    work with the datasieve pipeline
    :param sklearninstance: Any instantiated SKLearn transform, e.g. MinMaxScaler()
    :param n_jobs: if transform is parallelizable, will use this as the number of threads
    :param backed: if the transform is occurring in a special environment, it may benefit
        from backends such as "dask"
    """
    def __init__(self, sklearninstance: BaseEstimator, n_jobs=-1, backend="loky", **kwargs):
        self.backend = backend
        self.n_jobs = n_jobs
        self._skl = sklearninstance

    def fit(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        with parallel_backend(self.backend, n_jobs=self.n_jobs):
            self._skl = self._skl.fit(X, y=y)
        return X, y, sample_weight, feature_list

    def transform(self, X, y=None, sample_weight=None,
                  feature_list=None, outlier_check=False, **kwargs):
        with parallel_backend(self.backend, n_jobs=self.n_jobs):
            X = self._skl.transform(X)
        return X, y, sample_weight, feature_list

    def inverse_transform(self, X, y=None, sample_weight=None, feature_list=None, **kwargs):
        X = self._skl.inverse_transform(X)
        return X, y, sample_weight, feature_list
