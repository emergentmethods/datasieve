import logging
from typing import List, Tuple
import numpy.typing as npt
import pandas as pd
import numpy as np
import copy

logger = logging.getLogger('datasieve.pipeline')


class Pipeline:

    def __init__(self, steps: List[Tuple] = [], fitparams: dict[str, dict] = {}):
        """
        Pipeline object which holds a list of fit/transform objects.
        :param steps: list of tuples (str, transform())
        :param fitparams: dictionary of dictionaries, where the string key
        matches the str used to name the step in the steps list.
        """
        self.steps: List[Tuple] = steps
        self.fitparams: dict[str, dict] = self._validate_fitparams(fitparams, steps)
        self.pandas_types: bool = False
        self.feature_list: list = []
        self.label_list: list = []

    def _validate_fitparams(self, fitparams: dict[str, dict], steps: List[Tuple]):
        for _, (name, _) in enumerate(steps):
            if name not in fitparams.keys():
                fitparams[name] = {}  # add an empty dict

        return fitparams

    def fit_transform(self, X, y=None, sample_weight=None) -> Tuple[npt.ArrayLike,
                                                                    npt.ArrayLike,
                                                                    npt.ArrayLike]:
        """
        Iterate through the pipeline calling fit_transform() on each of the
        transformations
        """
        X, y, sample_weight = self._validate_arguments(X, y, sample_weight, fit=True)
        feature_list = copy.deepcopy(self.feature_list)

        for _, (name, trans) in enumerate(self.steps):
            logger.debug(f"Fit transforming {name}")
            X, y, sample_weight, feature_list = trans.fit_transform(
                X,
                y,
                sample_weight=sample_weight,
                feature_list=feature_list,
                **self.fitparams[name]
            )

        X, y, sample_weight = self._convert_back_to_df(X, y, sample_weight, feature_list)

        return X, y, sample_weight

    def transform(self, X, y=None, sample_weight=None) -> Tuple[npt.ArrayLike,
                                                                npt.ArrayLike,
                                                                npt.ArrayLike]:
        X, y, sample_weight = self._validate_arguments(X, y, sample_weight)
        feature_list = copy.deepcopy(self.feature_list)

        for _, (name, trans) in enumerate(self.steps):
            logger.debug(f"Transforming {name}")
            X, y, sample_weight, feature_list = trans.transform(
                X,
                y,
                sample_weight=sample_weight,
                feature_list=feature_list,
                **self.fitparams[name]
            )

        X, y, sample_weight = self._convert_back_to_df(X, y, sample_weight, feature_list)

        return X, y, sample_weight

    def fit(self, X, y=None, sample_weight=None):
        X, y, sample_weight = self._validate_arguments(X, y, sample_weight, fit=True)
        feature_list = copy.deepcopy(self.feature_list)

        for _, (name, trans) in enumerate(self.steps):
            logger.debug(f"Fitting {name}")
            X, y, sample_weight, feature_list = trans.fit(
                X,
                y,
                sample_weight=sample_weight,
                feature_list=feature_list,
                **self.fitparams[name]
            )

        return self

    def inverse_transform(self, X, y=None, sample_weight=None) -> Tuple[npt.ArrayLike,
                                                                        npt.ArrayLike,
                                                                        npt.ArrayLike]:
        X, y, sample_weight = self._validate_arguments(X, y, sample_weight)
        feature_list = copy.deepcopy(self.feature_list)

        for _, (name, trans) in reversed(list(enumerate(self.steps))):
            logger.debug(f"Inverse Transforming {name}")
            X, y, sample_weight, feature_list = trans.inverse_transform(
                X,
                y=y,
                sample_weight=sample_weight,
                feature_list=feature_list,
                **self.fitparams[name]
            )

        X, y, sample_weight = self._convert_back_to_df(X, y, sample_weight, feature_list)

        return X, y, sample_weight

    def _validate_arguments(self, X, y, sample_weight, fit=False):
        if isinstance(X, pd.DataFrame) and fit:
            self.pandas_types = True
            self.feature_list = X.columns
            if y is not None:
                self.label_list = y.columns
        elif fit:
            self.pandas_types = False
            self.feature_list = list(np.arange(0, X.shape[1]))
            if y is not None:
                if len(y.shape) > 1:
                    self.label_list = list(np.arange(0, y.shape[1]))
                else:
                    self.label_list = [0]
        elif isinstance(X, pd.DataFrame) and not fit:
            if list(X.columns) != list(self.feature_list):
                raise Exception(f"Pipeline expected {self.feature_list} but got {X.columns}.")
        elif not isinstance(X, pd.DataFrame) and not fit and self.pandas_types:
            X = pd.DataFrame(X, columns=self.feature_list)

        if self.pandas_types:
            try:
                X = X.to_numpy()
                if y is not None:
                    y = y.to_numpy()
            except AttributeError:
                raise Exception("Pipeline was fit with a dataframe, but it received "
                                f" {type(X)} instead.")

        if isinstance(sample_weight, pd.DataFrame) and sample_weight is not None:
            sample_weight = sample_weight.to_numpy()

        return X, y, sample_weight

    def _convert_back_to_df(self, X, y, sample_weight, feature_list):
        if not self.pandas_types:
            return X, y, sample_weight

        assert X.shape[1] == len(feature_list)

        X = pd.DataFrame(X, columns=feature_list)

        if y is not None:
            y = pd.DataFrame(y, columns=self.label_list)

        return X, y, sample_weight
