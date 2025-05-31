from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


class BaseMultiStateEstimator(BaseEstimator, ABC):
    """Base class for multi-state models following sklearn conventions."""

    @abstractmethod
    def fit(self, X, y=None, **fit_params):
        """Fit the model."""
        pass

    @abstractmethod
    def predict(self, X):
        """Make predictions."""
        pass

    def _check_is_fitted(self):
        """Check if the model has been fitted."""
        check_is_fitted(self, attributes=["is_fitted_"])
