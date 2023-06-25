import logging
from typing import List, Tuple, Dict
import numpy.typing as npt
import pandas as pd
import numpy as np
import copy
from datasieve.transforms.base_transform import BaseTransform

logger = logging.getLogger('datasieve.pipeline')


class Pipeline:

    def __init__(self, steps: List[Tuple] = [],
                 fitparams: Dict[str, dict] = {}):
        """
        Pipeline object which holds a list of fit/transform objects.
        :param steps: list of tuples (str, transform())
        :param fitparams: dictionary of dictionaries, where the string key
        matches the str used to name the step in the steps list.
        """
        self.steps: List[Tuple] = steps
        self.fitparams: Dict[str, dict] = self._validate_fitparams(fitparams, steps)
        self.pandas_types: bool = False
        self.feature_list: list = []
        self.label_list: list = []
        self.step_strings: list = []

    def _validate_fitparams(self, fitparams: Dict[str, dict], steps: List[Tuple]):
        for _, (name, _) in enumerate(steps):
            if name not in fitparams.keys():
                fitparams[name] = {}  # add an empty dict

        return fitparams

    def __getitem__(self, name: str):
        for _, (step_name, step) in enumerate(self.steps):
            if step_name == name:
                return step

        logger.warning(f"Could not find step {name} in pipeline, returning None")
        return None

    def __contains__(self, name: str):
        for _, (step_name, _) in enumerate(self.steps):
            if step_name == name:
                return True

        return False

    def append(self, step: Tuple[str, object], fitparams: dict = {}):
        """
        Append a step to the pipeline
        :param step: tuple of (str, transform())
        :param fitparams: dictionary of parameters to pass to fit
        """
        if step[0] in self.step_strings:
            raise ValueError(f"Step name {step[0]} already exists in pipeline."
                             "Ensure each step has a unique name.")
        self.step_strings.append(step[0])
        self.steps += [step]
        self.fitparams[step[0]] = fitparams

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
            self._validate_step(trans)
            X, y, sample_weight, feature_list = trans.fit_transform(
                X,
                y,
                sample_weight=sample_weight,
                feature_list=feature_list,
                **self.fitparams[name]
            )

        X, y, sample_weight = self._convert_back_to_df(X, y, sample_weight, feature_list)

        return X, y, sample_weight

    def transform(self, X, y=None, sample_weight=None, outlier_check=False) -> Tuple[npt.ArrayLike,
                                                                                     npt.ArrayLike,
                                                                                     npt.ArrayLike]:
        X, y, sample_weight = self._validate_arguments(
            X, y, sample_weight, outlier_check=outlier_check)
        feature_list = copy.deepcopy(self.feature_list)

        for _, (name, trans) in enumerate(self.steps):
            logger.debug(f"Transforming {name}")
            self._validate_step(trans)
            X, y, sample_weight, feature_list = trans.transform(
                X,
                y,
                sample_weight=sample_weight,
                feature_list=feature_list,
                outlier_check=outlier_check,
                **self.fitparams[name]
            )

        X, y, sample_weight = self._convert_back_to_df(X, y, sample_weight, feature_list, outlier_check)

        return X, y, sample_weight

    def fit(self, X, y=None, sample_weight=None):
        X, y, sample_weight = self._validate_arguments(X, y, sample_weight, fit=True)
        feature_list = copy.deepcopy(self.feature_list)

        for _, (name, trans) in enumerate(self.steps):
            logger.debug(f"Fitting {name}")
            self._validate_step(trans)
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
            self._validate_step(trans)
            X, y, sample_weight, feature_list = trans.inverse_transform(
                X,
                y=y,
                sample_weight=sample_weight,
                feature_list=feature_list,
                **self.fitparams[name]
            )

        X, y, sample_weight = self._convert_back_to_df(X, y, sample_weight, feature_list)

        return X, y, sample_weight

    # flake8: noqa: C901
    def _validate_arguments(self, X, y, sample_weight, fit=False, outlier_check=False):
        if isinstance(X, pd.DataFrame) and fit:
            self.pandas_types = True
            self.feature_list = X.columns
            self.features_in = X.columns
            if y is not None:
                self.label_list = y.columns
        elif fit:
            self.pandas_types = False
            self.features_in = list(np.arange(0, X.shape[1]))
            if y is not None:
                if len(y.shape) > 1:
                    self.label_list = list(np.arange(0, y.shape[1]))
                else:
                    self.label_list = [0]
        elif isinstance(X, pd.DataFrame) and not fit:
            if list(X.columns) != list(self.features_in):
                raise Exception(f"Pipeline expected {self.features_in} but got {X.columns}.")
        elif not isinstance(X, pd.DataFrame) and not fit and self.pandas_types:
            X = pd.DataFrame(X, columns=self.features_in)

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

        if not fit and outlier_check:
            if y is not None:
                raise Exception("Asking for outlier_check vector, but passed in y."
                                "outlier_check functionality only works when passing X"
                                "for pipeline.transform(X)")
            else:
                y = np.ones(X.shape[0])

        if fit and outlier_check:
            raise Exception("Asking for outlier_check with fit() is not possible."
                            "outlier_check functionality only works with transform.")

        return X, y, sample_weight

    def _convert_back_to_df(self, X, y, sample_weight, feature_list, outlier_check=False):
        if not self.pandas_types:
            return X, y, sample_weight

        assert X.shape[1] == len(feature_list)

        X = pd.DataFrame(X, columns=feature_list)

        if y is not None and not outlier_check:
            y = pd.DataFrame(y, columns=self.label_list)

        return X, y, sample_weight

    def _validate_step(cls, trans: BaseTransform):
        """
        Raise exception if `trans` is not a BaseTransform
        class
        """
        if not isinstance(trans, BaseTransform):
            raise Exception(
                f"{trans} is not a BaseTransform class. If you are using"
                "an SKLearn class, please wrap it inside the SKLearnWrapper()."
            )

        return
