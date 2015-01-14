"""\

Gaussian Unified Confidence Intervals with *known* variance
-----------------------------------------------------------

Estimate confidence intervals for the expectation value of a Gaussian
distributed observable with known variance and constraint (:math:`\mu \geq 0`) using
critical values for the likelihood ratio :math:`\Lambda` (more precisely :math:`-2\log \Lambda`)

Functions of Interest
---------------------

* :func:`confidence_interval`
* :func:`lower_limit`
* :func:`upper_limit`
"""

from scipy import stats
from scipy.optimize import bisect


def neg_2_log_likelihood_ratio_CDF(l, mu, sigma):
    """Calculate the CDF of -2log(likelihood ratio) CDF.

    Parameters
    ----------
    l : float
        The :math:`l = \lambda = -2ln(\Lambda)` value to calculate the CDF at.
    mu : float
        The expectation value to test.
    sigma : float
        The standard deviation of the distribution.

    Returns
    -------
    prob : float (between 0 and 1)
        The probability :math:`CDF(l) = Prob(\lambda < l)`.
    """
    p_lz = stats.norm.cdf(0, loc=mu, scale=sigma)
    p_gz = 1 - p_lz

    l_crit = mu**2 / sigma**2
    chi2 = stats.chi2(1)
    nu = chi2.cdf(l_crit)

    if l < l_crit:
        p_plus = p_gz * 2 * chi2.cdf(l) / (1.0 + nu)
    else:
        p_plus = p_gz * (chi2.cdf(l) + nu) / (1.0 + nu)

    tau = 0.5 * sigma**2 / mu * (mu**2 / sigma**2 - l)
    if tau < 0:
        p_minus = p_lz - stats.norm.cdf(tau, loc=mu, scale=sigma)
    else:
        p_minus = 0
    return p_plus + p_minus

def critical_value(mu_test, sigma, alpha):
    """Estimate the critical value for -2log(likelihood ratio) from bisecting CDF.

    Parameters
    ----------
    mu_test : float
        The assumed expectation value of the distribution to test.
    sigma : float
        The standaard deviation of the distribution.
    alpha : float (between 0 and 1)
        The significance level (i.e. 1-CL).

    Returns
    -------
    sol : float
        The critical value for :math:`\lambda=-2\log(\Lambda)`.

    See also
    --------
    Rotes Buch V, p.57--59
    """
    if mu_test > 0.0:
        def target(l):
            return 1 - alpha - neg_2_log_likelihood_ratio_CDF(l, mu_test, sigma)
        a = 0.0
        fa = target(a)
        if fa < 0:
            raise RuntimeError("1 - alpha - CDF_{-2ln(R)}(0) is negative!")
        b = 10.0
        while target(b) > 0:
            b = 10*b
        sol = bisect(target, 0.0, b)
    else:
        p_gz = 1 - stats.norm.cdf(0, loc=mu_test, scale=sigma)
        t = 1.0 - alpha / p_gz
        if t > 0:
            sol = stats.chi2.ppf(1.0 - alpha / p_gz, 1)
        else:
            print 'encountered questionable value < 0.0'
            sol = 0.0

    return sol

def neg_2_log_likelihood_ratio(mu_test, x):
    """Calculate -2log(likelihood ratio).

    Parameters
    ----------
    mu_test : float
        The assumed expectation value of the distribution.
    x : float
        The measured value.

    Returns
    -------
    lambda : float
        :math:`\lambda = -2\log(\Lamba)`.
    """
    if x > 0:
        return (x - mu_test)**2
    else:
        return mu_test * (mu_test - 2*x)

def fit_mu(x):
    """The positive-confined best-fit for the expectation value.

    Parameters
    ----------
    x : float
        The measured value.
    """
    return max(x, 0.0)

def upper_limit(x, sigma, cl):
    """Calculate the upper limit

    Parameters
    ----------
    x : float
        The measured value.
    sigma : float
        The std.deviation of the distribution.
    cl : float
        The confidence level.

    Returns
    -------
    sol : float
        The upper limit of the confidence interval.
    """
    alpha = 1.0 - cl
    mu_hat = fit_mu(x)

    def diff(mu):
        return critical_value(mu, sigma, alpha) - neg_2_log_likelihood_ratio(mu, x)
    assert diff(mu_hat) > 0

    n = 1
    while diff(mu_hat + n * sigma) > 0:
        n *= 2

    sol = bisect(diff, mu_hat, mu_hat + n * sigma)
    return sol

def lower_limit(x, sigma, cl):
    """Calculate the lower limit of the confidence interval for the expecation value.

    Parameters
    ----------
    x : float
        The measured value.
    sigma : float
        The std.deviation of the distribution.
    cl : float
        The confidence level.

    Returns
    -------
    sol : float
        The lower limit of the confidence interval.
    """
    alpha = 1.0 - cl
    mu_hat = fit_mu(x)

    if mu_hat == 0.0:
        return 0.0

    def diff(mu):
        return critical_value(mu, sigma, alpha) - neg_2_log_likelihood_ratio(mu, x)
    assert diff(mu_hat) > 0

    if diff(0.0) >= 0.0:
        return 0.0

    sol = bisect(diff, 0.0, mu_hat)
    return sol

def confidence_interval(x, sigma, cl):
    """Calculate the confidence interval for the positive-constrained expectation value.

    Parameters
    ----------
    x : float
        The measured value.
    sigma : float
        The std.deviation of the distribution.
    cl : float
        The confidence level.

    Returns
    -------
    ll, ul : float
        The lower and upper limits of the confidence interval.
    """
    ll = lower_limit(x, sigma, cl)
    ul = upper_limit(x, sigma, cl)
    return ll, ul
