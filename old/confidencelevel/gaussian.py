import numpy as np
from scipy import stats
from scipy.optimize import bisect

class GaussianUnconstrainedInterval(object):
    """Calculate central confidence intervals for an unconstraine Gaussian mean.

    Parameters
    ----------
    measured : float
        The measured value.
    sigma : float
        The *true* std. deviation of the measured value.
    cl : float
        The confidence level.
    """
    def __init__(self, **kwargs):
        self._measured = None
        self.best_fit = None
        self.sigma = None
        self.cl = None

        self._x = None

        self.update_parameters(**kwargs)

    @property
    def measured(self):
        return self._measured

    @measured.setter
    def measured(self, val):
        self._measured = val
        self.best_fit = self.calc_best_fit(self._measured)

    def update_parameters(self, measured=None, sigma=None, cl=None):
        if measured is not None:
            self.measured = measured
        if sigma is not None:
            self.sigma = sigma
        if cl is not None:
            if not 0 < cl < 1:
                raise ValueError('confidence level cl must be 0 < cl < 1')
            self.cl = cl

    def likelihood_ratio(self, mu_test):
        """Calculate the likelihood ratio.

        .. math::

            \\lambda(\\mu_{test}) = \\frac{L(\\mu_{test}; x)}{L(\\mu_{best}(x); x)}
                                  = e^{-\\frac{(x - \\mu_{test})^2}{2 \\sigma^2}}

        Parameters
        ----------
        mu_test : float
            The mean-parameter value to test.
        """
        return np.exp(-0.5 / self.sigma**2 * (self.measured-mu_test)**2)

    def diff(self, mu_test):
        return self.likelihood_ratio(mu_test) - self.critical_value(mu_test)

    def calc_best_fit(self, measured):
        """Best fit for an unconstrained Gaussian.

        It is just the `measured` value.
        """
        return measured

    def critical_value(self, __mu_test_unused):
        alpha = 1.0 - self.cl
        xi = stats.norm.isf(0.5 * alpha)
        return np.exp(-0.5 * xi**2)

    def upper_limit(self, **kwargs):
        """Calculate the upper limit

        The parameters can be preset with the constructor or
        :meth:`update_parameters`.

        Parameters
        ----------
        x : float, optional
            The measured value.
        sigma : float, optional
            The std.deviation of the distribution.
        cl : float, optional
            The confidence level.

        Returns
        -------
        sol : float
            The upper limit of the confidence interval.
        """
        self.update_parameters(**kwargs)

        assert self.diff(self.best_fit) > 0

        n = 1
        while self.diff(self.best_fit + n * self.sigma) > 0:
            n *= 2

        sol = bisect(self.diff, self.best_fit, self.best_fit + n * self.sigma)
        return sol

    def lower_limit(self, **kwargs):
        """Calculate the upper limit

        Parameters
        ----------
        x : float
            The measured value.
        igma : float
            The std.deviation of the distribution.
        cl : float
            The confidence level.

        Returns
        -------
        sol : float
            The lower limit of the confidence interval.
        """
        self.update_parameters(**kwargs)

        assert self.diff(self.best_fit) > 0

        n = 1
        while self.diff(self.best_fit - n * self.sigma) > 0:
            n *= 2

        sol = bisect(self.diff, self.best_fit - n * self.sigma, self.best_fit)
        return sol

    def confidence_interval(self, **kwargs):
        self.update_parameters(**kwargs)
        ll = self.lower_limit()
        ul = self.upper_limit()
        return (ll, ul)

def unconstrained_ci(measured, sigma, cl):
    ci = GaussianUnconstrainedInterval(measured, sigma, cl)
    return ci.confidence_interval()

class GaussianUnifiedInterval(GaussianUnconstrainedInterval):
    """
    Estimate confidence intervals for mean of a Normal distributed observable
    with known variance and constraint (mu >= 0) using critical values for the
    likelihood ratio.

    This is meant for comparison with Feldman+Cousins paper.
    """
    def calc_best_fit(self, measured):
        return max(0.0, measured)

    def likelihood_ratio(self, mu_test):
        a = -0.5 * (self.measured - mu_test)**2 / self.sigma**2
        if self.measured < 0:
            a += 0.5 * self.measured**2 / self.sigma**2
        return np.exp(a)

def unified_ci(measured, sigma, cl):
    ci = GaussianUnifiedInterval(measured=measured, sigma=sigma, cl=cl)
    return ci.confidence_interval()


