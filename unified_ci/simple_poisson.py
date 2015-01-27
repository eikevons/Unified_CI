"""\
Poissonian Unified Confidence Intervals with *known* background
---------------------------------------------------------------

Estimate confidence intervals for the expectation value of a Poissonian
distributed observable with known Poissonian-distributed background
:math:`b` using critical values for the likelihood ratio.

Functions of Interest
---------------------

* :func:`confidence_interval`
* :func:`lower_limit`
* :func:`upper_limit`

Todo
----
Treat `b == 0`.
"""
from __future__ import print_function, division, absolute_import
import numpy as np
import logging
# from scipy import stats
from scipy import special

from .tools import bisect

def poisson_pmf(k, mu):
    """Calculate the Poissonian PMF.

    Copied from `scipy.stats.poisson._pmf` and `scipy.stats.poisson._logpmf` 
    """
    Pk = k * np.log(mu) - special.gammaln(k + 1) - mu
    return np.exp(Pk)

def poisson_minor_isf(q_upper, mu):
    """Calculate the Poissonian "minor" inverse survival function.

    Here, "minor" means that the returned quantile `n` is chosen to be the
    smallest value where the corresponding upper-tail probability is less
    than or equal to `q_upper`:

        .. math::

            n = \min {{ k : P(N >= k) <= q_{{upper}} }}

    Parameters
    ----------
    q_upper : float
        The upper tail probability.
    mu : float
        The expectation value of the Poisson distribution.

    Returns
    -------
    n : int
        The "minor" quantile.
    """
    return np.ceil(special.pdtrik(1.0 - q_upper, mu)).astype(np.int)


def poisson_sf(n, mu):
    """Calculate the upper-tail probability for a Poisson distribution.

    The calculated probability is:

    .. math::

        SF(n) = P(N > n) = 1 - CDF(n)

    Parameters
    ----------
    n : int
        Quantile.
    mu : 
        The expectation value of the Poisson distribution.

    Returns
    -------

    """
    return special.pdtrc(n, mu)



def fit_theta(n, b):
    """The positive-confined best-fit for Poissonian signal parameter theta.

    Parameters
    ----------
    n : int
        The number of observed events.
    b : float
        The background rate.

    Returns
    -------
    theta_fit : float
        The best-fit theta value.
    """
    return np.fmax(0.0, n-b)


def likelihood_ratio(n, b, t):
    """The likelihood ratio for a theta-value of `t` for background `b` and measurent `n`.

    Parameters
    ----------
    n : int
        The number of observed events.
    b : float
        The background rate.
    t : float
        The assumed theta value.

    Returns
    -------
    lr : float
        The likelihood ratio.
    """
    t_fit = fit_theta(n, b)
    return ((t + b) / (t_fit + b))**n * np.exp(t_fit - t)


def critical_value(b, t, alpha):
    """Calculate the critical likelihood ratio value.

    The critical theta value is defined by:

    .. math::
        Prob( L < L_{{crit}}; t) <= \alpha = 1 - clvl

    Parameters
    ----------
    b : float
        Background rate.
    t : float
        Signal rate.
    alpha : float
        Lower-tail probability (:math:`\alpha = 1 - clvl`).

    Returns
    -------
    lr_crit : float
        The critical likelihood ratio value.
    """
    # poi = stats.poisson(b+t)
    mu = b + t

    # Find n_max
    # criteria are
    #   P[n > n_max] < p_thresh
    #   P[n = n_max] < p_thresh
    #   P[n >= n_max] < p_thresh 
    #  ==> SF(n_max-1) < p_thresh
    #  ==> n_max = 1 + ISF(p_thresh)
    # NOTE: SF(M) = P[n > M] = P[n >= M+1]
    # p_thresh = min(alpha, poi.pmf(0))
    p_thresh = min(alpha, poisson_pmf(0, mu))
    # n_max = int(poi.isf(p_thresh)) + 1
    n_max = poisson_minor_isf(p_thresh, mu)

    # n_p_lr = [(n_max, poi.sf(n_max), likelihood_ratio(n_max+1, b, t))]
    n_p_lr = [(n_max, poisson_sf(n_max, mu), likelihood_ratio(n_max+1, b, t))]
    # n_p_lr.extend((n, poi.pmf(n), likelihood_ratio(n, b, t)) for n in xrange(n_max+1))
    n_p_lr.extend((n, poisson_pmf(n, mu), likelihood_ratio(n, b, t)) for n in xrange(n_max+1))
    n_p_lr = sorted(n_p_lr, key=operator.itemgetter(2))

    p_cum = 0.0
    for i, (n, p, lr) in enumerate(n_p_lr):
        p_cum += p
        if p_cum >= alpha:
            if i == 0:
                # TODO: is this the proper way to deal with this?
                logging.warn('i==0: algorithm failed for b={} t={} alpha={}: accepting overcoverage!'.format(b, t, alpha))
                return n_p_lr[i][2]
            return n_p_lr[i-1][2]
    raise RuntimeError('this should never raise!')


def mk_delta_func(n, b, clvl):
    """Prepare 'likelihood ratio minus critical value' function."""
    alpha = 1.0 - clvl
    cache = {}
    def delta(t):
        if t not in cache:
            r = likelihood_ratio(n, b, t) - critical_value(b, t, alpha)
            cache[t] = r
        else:
            r = cache[t]
        return r
    return delta


def lower_limit(n, b, clvl, delta=None):
    """Calculate the lower limit of the confidence interval.

    Parameters
    ----------
    n : int
    b : float
    clvl : float
    delta : callable, optional
    """
    t_best = fit_theta(n, b)

    if delta is None:
        delta = mk_delta_func(n, b, clvl)

    if t_best == 0.0 or delta(0.0) >= 0.0:
        return 0.0
    else:
        # NOTE: The standard functions do not work here, because there are
        # whole intervals where `delta(t) == 0` and we need the *inner* bounds of
        # these intervals.
        # t0 = optimize.brentq(f, 0, t_best)
        # t0 = optimize.bisect(f, 0, t_best, xtol=1e-4)
        # So we have to use a hand-crafted root-finding.
        return bisect(delta, t_best, 0)


def upper_limit(n, b, clvl, delta=None):
    """Calculate the upper limit of the confidence interval.

    Parameters
    ----------
    n : int
        The measured value.
    b : float
        The background rate.
    clvl : float
        The confidence level.

    Returns
    -------
    ul : float
        The upper limits of the confidence interval.
    """
    t_best = fit_theta(n, b)

    if delta is None:
        delta = mk_delta_func(n, b, clvl)

    u = t_best
    v = max(1.0, 2*u)
    while delta(v) >= 0:
        u = v
        v = 2*u
    t1 = bisect(delta, u, v)
    return t1


def confidence_interval(n, b, clvl):
    """Calculate the confidence interval for the expectation value.

    Parameters
    ----------
    n : int
        The measured value.
    b : float
        The background rate.
    clvl : float
        The confidence level.

    Returns
    -------
    ll, ul : float
        The lower and upper limits of the confidence interval.
    """
    delta = mk_delta_func(n, b, clvl)
    t0 = lower_limit(n, b, clvl, delta)
    t1 = upper_limit(n, b, clvl, delta)
    return t0, t1


##############################
## TESTS
from .tools import conservative_quantile
import itertools
import operator




def critical_theta_mc(b, t, alpha, N_mc):
    """Estimate the critical c_theta value from MC sampling.

    Parameters
    ----------
    b : float
        The background rate.
    t : float
        The theta value to test.
    alpha : float
        1-clvl
    N_mc : int
        The Monte Carlo sample size.
    """
    n_sample = np.random.poisson(t+b, size=N_mc)
    lr = likelihood_ratio(n_sample, b, t)

    return conservative_quantile(lr, -alpha)[0]

# critical_value = lambda b, t, alpha: critical_theta_mc(b, t, alpha, 10000)


def compare_crit_thetas(bs, ts):
    """Compare Monte-Carlo and analytic results for theta_crit."""
    for b, t in itertools.product(bs, ts):
        if b == t == 0.0:
            continue
        print('b={}  t={}'.format(b, t))
        for clvl in (0.5, 0.6832, 0.9, 0.95, 0.99):
            alpha = 1.0 - clvl
            an = critical_value_an(b, t, alpha)
            mc = critical_theta_mc(b, t, alpha, 10000)
            print('analytic: {}  mc: {}'.format(an, mc))


def visual_test_coverage(b, t, clvl, N_tests):
    sample = np.random.poisson(t+b, size=N_tests)
    n_cov = 0
    for n in sample:
        t0, t1 = confidence_interval(n, b, clvl)
        if t0 <= t <= t1:
            n_cov += 1
    print("Target clvl: {0:.4%} => {1}/{2} = {3:.4%}".format(clvl, n_cov, N_tests, 1.0*n_cov/N_tests))
    return n_cov
