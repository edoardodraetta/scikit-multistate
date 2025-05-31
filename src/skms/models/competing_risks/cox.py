class CoxPHEstimator(EventSpecificEstimator):
    """Cox Proportional Hazards estimator using scikit-survival.

    This is a wrapper around sksurv.linear_model.CoxPHSurvivalAnalysis
    that implements the EventSpecificEstimator interface.

    Parameters
    ----------
    alpha : float, default=0.0
        Regularization parameter for ridge regression penalty
    ties : {'breslow', 'efron'}, default='breslow'
        Method for handling tied event times
    n_iter : int, default=100
        Maximum number of iterations
    tol : float, default=1e-4
        Convergence tolerance
    verbose : bool, default=False
        Whether to print convergence messages
    """

    def __init__(self, alpha=0.0, ties="breslow", n_iter=100, tol=1e-4, verbose=False):
        self.alpha = alpha
        self.ties = ties
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None, **fit_params):
        """Fit the Cox proportional hazards model.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data
        y : structured array, shape = (n_samples,)
            A structured array containing the binary event indicator
            as first field, and time of event or time of censoring as
            second field.
        sample_weight : array-like, shape = (n_samples,), optional
            Sample weights (not supported by scikit-survival's Cox model)
        **fit_params : dict
            Additional parameters (unused)

        Returns
        -------
        self : object
        """
        from sksurv.linear_model import CoxPHSurvivalAnalysis

        # Validate input
        X = check_array(X, ensure_min_samples=2, estimator=self)

        # Handle sample weights
        if sample_weight is not None:
            warnings.warn(
                "sample_weight is not supported by CoxPHSurvivalAnalysis "
                "and will be ignored. Consider using CoxnetSurvivalAnalysis "
                "with sample weights transformed into the target."
            )

        # Store feature information
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)
        self.n_features_in_ = X.shape[1]

        # Initialize and fit the underlying model
        self.estimator_ = CoxPHSurvivalAnalysis(alpha=self.alpha, ties=self.ties, n_iter=self.n_iter, tol=self.tol, verbose=self.verbose)

        self.estimator_.fit(X, y)

        # Store unique event times
        self.unique_event_times_ = np.unique(y[y["event"]]["time"])

        # Store training data info for baseline hazard
        self.event_times_ = y["time"]
        self.event_indicators_ = y["event"]

        return self

    def predict_cumulative_hazard_function(self, X, times=None):
        """Predict cumulative hazard function."""
        check_is_fitted(self)
        X = check_array(X, ensure_min_samples=1, estimator=self)

        if times is None:
            times = self.unique_event_times_

        # Get the cumulative hazard functions
        cumhaz_funcs = self.estimator_.predict_cumulative_hazard_function(X)

        # Evaluate at specified times
        n_samples = X.shape[0]
        n_times = len(times)
        cumhaz = np.zeros((n_samples, n_times))

        for i, func in enumerate(cumhaz_funcs):
            cumhaz[i, :] = func(times)

        return cumhaz

    def predict_survival_function(self, X, times=None):
        """Predict survival function."""
        check_is_fitted(self)
        X = check_array(X, ensure_min_samples=1, estimator=self)

        if times is None:
            times = self.unique_event_times_

        # Get the survival functions
        surv_funcs = self.estimator_.predict_survival_function(X)

        # Evaluate at specified times
        n_samples = X.shape[0]
        n_times = len(times)
        survival = np.zeros((n_samples, n_times))

        for i, func in enumerate(surv_funcs):
            survival[i, :] = func(times)

        return survival

    def get_unique_event_times(self):
        """Get unique event times from the training data."""
        check_is_fitted(self)
        return self.unique_event_times_

    def predict_linear_hazard(self, X):
        """Predict linear hazard (log hazard ratio).

        This is specific to Cox models and represents the linear combination
        of covariates and coefficients.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Data to predict on

        Returns
        -------
        linear_hazard : ndarray, shape = (n_samples,)
            Linear hazard values
        """
        check_is_fitted(self)
        return self.estimator_.predict(X)

    def get_baseline_hazard(self, times=None):
        """Get baseline hazard function.

        Parameters
        ----------
        times : array-like, optional
            Times at which to evaluate the baseline hazard

        Returns
        -------
        baseline_hazard : ndarray
            Baseline hazard at specified times
        """
        check_is_fitted(self)

        if times is None:
            times = self.unique_event_times_

        # Get baseline cumulative hazard
        baseline_cumhaz = self.estimator_.cum_baseline_hazard_
        baseline_cumhaz_func = baseline_cumhaz.iloc[:, 0]

        # Interpolate to requested times
        from scipy.interpolate import interp1d

        f = interp1d(
            baseline_cumhaz.index,
            baseline_cumhaz_func.values,
            kind="previous",
            bounds_error=False,
            fill_value=(0, baseline_cumhaz_func.iloc[-1]),
        )

        cumhaz_at_times = f(times)

        # Convert to hazard
        hazard = np.zeros_like(cumhaz_at_times)
        if len(times) > 0:
            hazard[0] = cumhaz_at_times[0]
            if len(times) > 1:
                hazard[1:] = np.diff(cumhaz_at_times)

        return hazard

    @property
    def coef_(self):
        """Coefficients of the features."""
        check_is_fitted(self)
        return self.estimator_.coef_

    def get_feature_names_out(self, input_features=None):
        """Get output feature names."""
        check_is_fitted(self)
        if input_features is not None:
            return np.asarray(input_features)
        elif hasattr(self, "feature_names_in_"):
            return self.feature_names_in_
        else:
            return np.array([f"x{i}" for i in range(self.n_features_in_)])
