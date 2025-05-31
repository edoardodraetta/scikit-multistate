class AFTEstimator(EventSpecificEstimator):
    """Accelerated Failure Time estimator using scikit-survival.

    This is a wrapper around sksurv.linear_model.IPCRidge
    that implements the EventSpecificEstimator interface.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization parameter
    fit_intercept : bool, default=True
        Whether to fit an intercept
    normalize : bool, default=False
        Whether to normalize features
    copy_X : bool, default=True
        Whether to copy X
    max_iter : int, default=None
        Maximum number of iterations
    tol : float, default=1e-3
        Convergence tolerance
    solver : {'auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'}
        Solver to use
    """

    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=1e-3, solver="auto"):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver

    def fit(self, X, y, sample_weight=None, **fit_params):
        """Fit the AFT model."""
        from sksurv.linear_model import IPCRidge

        # Validate input
        X = check_array(X, ensure_min_samples=2, estimator=self)

        # Store feature information
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)
        self.n_features_in_ = X.shape[1]

        # Initialize and fit the underlying model
        self.estimator_ = IPCRidge(
            alpha=self.alpha,
            fit_intercept=self.fit_intercept,
            normalize=self.normalize,
            copy_X=self.copy_X,
            max_iter=self.max_iter,
            tol=self.tol,
            solver=self.solver,
        )

        self.estimator_.fit(X, y, sample_weight=sample_weight)

        # Store unique event times
        self.unique_event_times_ = np.unique(y[y["event"]]["time"])

        return self

    # Implement other required methods...
    # (Similar pattern to CoxPHEstimator but adapted for AFT model)
