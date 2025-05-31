"""
Base estimator that serves multiple model types.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sksurv.util import Surv


class BaseCompetingRisksModel(ABC, BaseEstimator):
    """
    Foundation for competing risks analysis.

    This abstract base class provides a common interface for competing risks models,
    where multiple event types can occur and the occurrence of one event prevents
    the occurrence of others.

    Parameters
    ----------
    base_estimator : object, optional
        The underlying estimator to use for modeling. Should implement fit/predict
        methods compatible with survival analysis.

    Attributes
    ----------
    base_estimator_ : object
        The fitted base estimator.
    event_types_ : array-like
        Unique event types found in the training data.
    duration_col_ : str
        Name of the duration column used during fitting.
    event_col_ : str
        Name of the event column used during fitting.
    """

    @abstractmethod
    def __init__(self, base_estimator: BaseEstimator | None = None) -> None:
        """
        Initialize the competing risks model.

        Parameters
        ----------
        base_estimator : BaseEstimator, optional
            The underlying estimator to use for modeling. If None, a default
            estimator should be used by the concrete implementation.
        """
        self.base_estimator = base_estimator

    @abstractmethod
    def fit(self, df: pd.DataFrame, duration_col: str, event_col: str) -> "BaseCompetingRisksModel":
        """
        Fit the competing risks model to training data.

        Parameters
        ----------
        df : pandas.DataFrame
            Training data containing duration and event information.
        duration_col : str
            Name of the column containing time-to-event or censoring times.
        event_col : str
            Name of the column containing event indicators. Should contain:
            - 0 for censored observations
            - Positive integers for different event types (1, 2, 3, etc.)

        Returns
        -------
        self : BaseCompetingRisksModel
            Fitted estimator.

        Notes
        -----
        The event column should encode competing events as distinct positive integers,
        with 0 indicating censoring (no event observed).
        """
        pass

    @abstractmethod
    def predict_cumulative_incidence(
        self, X: pd.DataFrame | np.ndarray, times: list[float] | np.ndarray, event_type: int
    ) -> np.ndarray:
        """
        Predict cumulative incidence function for a specific event type.

        The cumulative incidence function (CIF) represents the probability that
        an event of the specified type occurs by time t, accounting for the
        presence of competing risks.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray of shape (n_samples, n_features)
            Feature matrix for which to make predictions.
        times : array-like of shape (n_times,)
            Time points at which to evaluate the cumulative incidence function.
        event_type : int
            The event type for which to calculate cumulative incidence.
            Must be one of the event types seen during fitting.

        Returns
        -------
        cumulative_incidence : numpy.ndarray of shape (n_samples, n_times)
            Cumulative incidence probabilities for each sample at each time point.
            Values range from 0 to 1, representing the probability that the
            specified event occurs by each time point.

        Raises
        ------
        ValueError
            If event_type was not seen during fitting or if the model hasn't
            been fitted yet.
        """
        pass

    def predict_survival_function(self, X: pd.DataFrame | np.ndarray, times: list[float] | np.ndarray) -> np.ndarray:
        """
        Predict overall survival function (probability of being event-free).

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray of shape (n_samples, n_features)
            Feature matrix for which to make predictions.
        times : array-like of shape (n_times,)
            Time points at which to evaluate the survival function.

        Returns
        -------
        survival_probs : numpy.ndarray of shape (n_samples, n_times)
            Survival probabilities for each sample at each time point.
            Represents P(T > t) where T is time to any event.
        """
        # Default implementation - can be overridden by subclasses
        # S(t) = 1 - sum of all cumulative incidences
        if not hasattr(self, "event_types_"):
            raise ValueError("Model must be fitted before making predictions")

        total_cif = np.zeros((X.shape[0], len(times)))
        for event_type in self.event_types_:
            if event_type != 0:  # Skip censoring indicator
                total_cif += self.predict_cumulative_incidence(X, times, event_type)

        return 1 - total_cif

    def get_event_types(self) -> np.ndarray:
        """
        Get the event types identified during fitting.

        Returns
        -------
        event_types : numpy.ndarray
            Array of unique event type identifiers, excluding censoring (0).

        Raises
        ------
        ValueError
            If the model hasn't been fitted yet.
        """
        if not hasattr(self, "event_types_"):
            raise ValueError("Model must be fitted before accessing event types")
        return self.event_types_[self.event_types_ != 0]


class CompetingRisksModel(BaseEstimator):
    """Competing risks model that works directly with scikit-survival estimators."""

    def __init__(self, base_estimator=None, tie_breaking="epsilon", epsilon_range=(0.0, 0.0001)):
        # Default to Cox if not specified
        if base_estimator is None:
            from sksurv.linear_model import CoxPHSurvivalAnalysis

            base_estimator = CoxPHSurvivalAnalysis()

        self.base_estimator = base_estimator
        self.tie_breaking = tie_breaking
        self.epsilon_range = epsilon_range

    def fit(self, df, duration_col="T", event_col="E", **fit_params):
        """Fit competing risks model."""
        # ... validation code ...

        self.event_types_ = df[event_col].unique()
        self.event_types_ = self.event_types_[self.event_types_ > 0]

        self.event_models_ = {}
        self.event_times_ = {}

        for event_type in self.event_types_:
            # Create a copy of the base estimator
            from sklearn.base import clone

            model = clone(self.base_estimator)

            # Prepare data for this specific event
            event_data = self._prepare_event_data(df, event_type, event_col)

            # Fit the model directly
            X = event_data.drop([duration_col, event_col], axis=1)
            y = self._make_survival_y(event_data, duration_col, event_col)

            model.fit(X, y, **fit_params)
            self.event_models_[event_type] = model

            # Store event times for this event type
            self.event_times_[event_type] = np.unique(y[y["event"]]["time"])

        return self

    def _prepare_event_data(self, df: pd.DataFrame, event_type: int, event_col: str) -> pd.DataFrame:
        df = df.copy()
        df[event_col] = (df[event_col] == event_type).astype(int)
        return df

    def _make_survival_y(self, event_data: pd.DataFrame, duration_col: str, event_col: str):
        """Return structured array compatible with scikit-survival"""
        return Surv.from_arrays(event=event_data[event_col].astype(bool).values, time=event_data[duration_col].values)

    def _get_hazard_at_times(self, model, X, times):
        """Adapter method to get hazard from any sksurv model.

        This handles the interface difference between what competing risks
        needs and what sksurv provides.
        """
        # Get cumulative hazard function
        cum_hazard_funcs = model.predict_cumulative_hazard_function(X)

        # Evaluate at specified times
        n_samples = X.shape[0]
        cum_hazard = np.zeros((n_samples, len(times)))

        for i, func in enumerate(cum_hazard_funcs):
            cum_hazard[i, :] = func(times)

        # Convert to hazard (instantaneous risk)
        hazard = np.zeros_like(cum_hazard)
        if len(times) > 0:
            hazard[:, 0] = cum_hazard[:, 0]
            if len(times) > 1:
                hazard[:, 1:] = np.diff(cum_hazard, axis=1)

        return hazard

    def predict_cif(self, X, times, event_type):
        """Predict cumulative incidence function."""
        # Use the adapted interface
        model = self.event_models_[event_type]
        event_times = self.event_times_[event_type]

        # Get hazard at event times for this specific risk
        hazard = self._get_hazard_at_times(model, X, event_times)

        # Get overall survival (from all risks)
        survival = self._predict_overall_survival(X, event_times)

        # Compute CIF
        cif_values = np.cumsum(hazard * survival, axis=1)

        # Interpolate to requested times
        from scipy.interpolate import interp1d

        interpolated_cif = np.zeros((X.shape[0], len(times)))

        for i in range(X.shape[0]):
            f = interp1d(
                event_times, cif_values[i, :], kind="previous", bounds_error=False, fill_value=(0, cif_values[i, -1])
            )
            interpolated_cif[i, :] = f(times)

        return interpolated_cif

    def _predict_overall_survival(self, X, times):
        """Compute overall survival by combining all event-specific hazards."""
        n_samples = X.shape[0]
        n_times = len(times)
        total_cum_hazard = np.zeros((n_samples, n_times))

        for _, model in self.event_models_.items():
            # Get cumulative hazard for this event
            cum_hazard_funcs = model.predict_cumulative_hazard_function(X)
            cum_hazard = np.zeros((n_samples, n_times))

            for i, func in enumerate(cum_hazard_funcs):
                cum_hazard[i, :] = func(times)

            total_cum_hazard += cum_hazard

        # Overall survival: S(t) = exp(-sum_k H_k(t))
        overall_survival = np.exp(-total_cum_hazard)
        return overall_survival
