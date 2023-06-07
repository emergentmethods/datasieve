from abc import ABC, abstractmethod
from typing import Union
import numpy.typing as npt

ArrayOrNone = Union[npt.ArrayLike, None]
ListOrNone = Union[list, None]


class BaseTransform(ABC):
    """
    Base class for all transforms.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def fit(self, X: npt.ArrayLike,
            y: ArrayOrNone = None,
            sample_weight: ArrayOrNone = None,
            feature_list: ListOrNone = None,
            **kwargs):
        """
        All fit logic contained here.
        :param X: array to be used for fit
        :param y: array which may assist in fit or not
        :param sample_weight: array which may assist in fit or not
        :param feature_list: list of features which may assist in fit or not
        """
        return X, y, sample_weight, feature_list

    @abstractmethod
    def transform(self, X: npt.ArrayLike,
                  y: ArrayOrNone = None,
                  sample_weight: ArrayOrNone = None,
                  feature_list: ListOrNone = None,
                  outlier_check: bool = False,
                  **kwargs):
        """
        All transform logic contained here.
        :param X: array to be transformed
        :param y: array which may assist/be affected by the transform
        :param sample_weight: array which may assist/be affected by the transform
        :param feature_list: list of features which may be assist/be affected by the transform
        :param outlier_check: boolean flag to indicate if outlier check should be performed
        """
        return X, y, sample_weight, feature_list

    def fit_transform(self, X: npt.ArrayLike,
                      y: ArrayOrNone = None,
                      sample_weight: ArrayOrNone = None,
                      feature_list: ListOrNone = None,
                      **kwargs):
        """
        Can be left undefined if the fit_transform follows simple self.fit - self.transform
        pattern
        :param X: array to be fit_transformed
        :param y: array which may assist/be affected by the transform
        :param sample_weight: array which may assist/be affected by the transform
        :param feature_list: list of features which may be assist/be affected by the transform
        """
        self.fit(X, y, sample_weight, feature_list)
        return self.transform(X, y, sample_weight, feature_list)

    def inverse_transform(self, X: npt.ArrayLike,
                          y: ArrayOrNone = None,
                          sample_weight: ArrayOrNone = None,
                          feature_list: ListOrNone = None,
                          **kwargs):
        """
        Can be left undefined if the transform is not reversible, in that case it simply
        passes the X, y, sample_weight, feature_list through to the subsequent transform.
        :param X: array to be inverse_transformed
        :param y: array which may assist/be affected by the transform
        :param sample_weight: array which may assist/be affected by the transform
        :param feature_list: list of features which may be assist/be affected by the transform
        """
        return X, y, sample_weight, feature_list
