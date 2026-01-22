from numbers import Real
from sklearn.linear_model._glm.glm import _GeneralizedLinearRegressor
from sklearn.utils._param_validation import Interval
from .loss import HalfNegativeBinomialLoss
from sklearn._loss.loss import HalfBinomialLoss


class NegativeBinomialRegressor(_GeneralizedLinearRegressor):
    """
    Generalized Linear Model with a Negative Binomial distribution.
    Requires the compiled '_loss' Cython module.
    """
    # Constrain k to be non-negative
    _parameter_constraints: dict = {
        **_GeneralizedLinearRegressor._parameter_constraints,
        "k": [Interval(Real, 0, None, closed="left")],
    }

    def __init__(
        self,
        *,
        alpha=1.0,
        k=1.0,
        fit_intercept=True,
        solver="lbfgs",
        max_iter=100,
        tol=1e-4,
        warm_start=False,
        verbose=0,
    ):
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            solver=solver,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            verbose=verbose,
        )
        self.k = k

    def _get_loss(self):
        # Use the Cython class
        return HalfNegativeBinomialLoss(k=self.k)


class BinomialRegressor(_GeneralizedLinearRegressor):
    """
    Generalized Linear Model with a Binomial distribution.
    """
    _parameter_constraints: dict = {
        **_GeneralizedLinearRegressor._parameter_constraints
    }
    def __init__(
        self,
        *,
        alpha=1.0,
        fit_intercept=True,
        solver="lbfgs",
        max_iter=100,
        tol=1e-4,
        warm_start=False,
        verbose=0,
    ):
        super().__init__(
            alpha=alpha,
            fit_intercept=fit_intercept,
            solver=solver,
            max_iter=max_iter,
            tol=tol,
            warm_start=warm_start,
            verbose=verbose,
        )

    def _get_loss(self):
        # Use the Cython class
        return HalfBinomialLoss()
    
    def predict_proba(self, X):
        return self.predict(X=X)
