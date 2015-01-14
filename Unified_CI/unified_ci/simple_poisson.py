"""\
Poissonian Unified Confidence Intervals with *known* background
---------------------------------------------------------------

Estimate confidence intervals for the expectation value of a Poissonian
distributed observable with known Poissonian-distributed background
:math:`b` using critical values for the likelihood ratio.

Functions of Interest
---------------------

* :func:`confidence_interval`
"""
from __future__ import print_function, division, absolute_import
import numpy as np
from scipy import stats

from .tools import bisect

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

    ..math::
        Prob{ L < L_{{crit}}; t} <= 1 - clvl

    Parameters
    ----------
    b : float
        Background rate.
    t : float
        Signal rate.
    clvl : float
        Confidence level.

    Returns
    -------
    lr_crit : float
        The critical likelihood ratio value.
    """
    poi = stats.poisson(b+t)

    # Find n_max
    # criteria are
    #   P[n > n_max] < p_thresh
    #   P[n = n_max] < p_thresh
    #   P[n >= n_max] < p_thresh 
    #  ==> SF(n_max-1) < p_thresh
    #  ==> n_max = 1 + ISF(p_thresh)
    # NOTE: SF(M) = P[n > M] = P[n >= M+1]
    p_thresh = min(alpha, poi.pmf(0))
    n_max = int(poi.isf(p_thresh)) + 1

    n_p_lr = [(n_max, poi.sf(n_max), likelihood_ratio(n_max+1, b, t))]
    n_p_lr.extend((n, poi.pmf(n), likelihood_ratio(n, b, t)) for n in xrange(n_max+1))
    n_p_lr = sorted(n_p_lr, key=operator.itemgetter(2))

    p_cum = 0.0
    for i, (n, p, lr) in enumerate(n_p_lr):
        p_cum += p
        if p_cum >= alpha:
            if i == 0:
                raise RuntimeError('i==0: algorithm failed!')
            return n_p_lr[i-1][2]
    raise RuntimeError('this should never raise!')


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
    t_best = fit_theta(n, b)
    alpha = 1.0 - clvl

    cache = {}
    # lr - crit_theta
    def delta(t):
        if t not in cache:
            cache[t] = likelihood_ratio(n, b, t) - critical_value(b, t, alpha)
        return cache[t]

    if t_best == 0.0 or delta(0.0) >= 0:
        t0 = 0.0
    else:
        # NOTE: The standard functions do not work here, because there are
        # whole intervals where `f(t) == 0` and we need the *inner* bounds of
        # these intervals.
        # t0 = optimize.brentq(f, 0, t_best)
        # t0 = optimize.bisect(f, 0, t_best, xtol=1e-4)
        # So we have to use a hand-crafted root-finding.
        t0 = bisect(delta, t_best, 0)

    u = t_best
    v = max(1.0, 2*u)
    while delta(v) >= 0:
        u = v
        v = 2*u
    t1 = bisect(delta, u, v)

    return t0, t1


##############################
## TESTS
import itertools
import operator


def conservative_quantile(x, p):
    """Calculate conservative quantiles.

    Conservative quantile means that the lower tail frequency is
    ensured (:math:`Pr{x_i <= q} >= p`).

    Conservative upper quantile means that the upper tail frequency is
    ensured (:math:`Pr{x_i >= q} >= 1-p`).

    Parameters
    ----------
    x : ndarray-like
        The data sample.
    p : float or ndarray of floats
        The target lower-tail frequency `0 <= p <= 1`.

    Returns
    -------
    q : float of ndarray of floats
        The quantile values.
    n : int
        The number of unique values in `x`.
    """
    if np.any(np.logical_or(p < 0, p > 1)):
        raise ValueError("Probability must be between 0 and 1")

    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("Data sample must be 1-dim")

    items, freqs = itemfreq(x).T

    cumfreq = np.cumsum(freqs / x.size)

    if np.isscalar(p):
        i = np.where(cumfreq >= p)[0][0]
        return items[i], len(items)

    else:
        r = np.empty(len(p), dtype=x.dtype)
        for k, pk in enumerate(p):
            i = np.where(cumfreq >= p)[0][0]
            r[k] = items[i]
        return r, len(items)


def critical_theta_mc(b, t, clvl, N_mc):
    """Estimate the critical c_theta value from MC sampling.

    Parameters
    ----------
    b : float
        The background rate.
    t : float
        The theta value to test.
    clvl : float
        The confidence level.
    N_mc : int
        The Monte Carlo sample size.
    """
    n_sample = np.random.poisson(t+b, size=N_mc)
    lr = likelihood_ratio(n_sample, b, t)

    return conservative_quantile(lr, -(1.0 - clvl))[0]


def compare_crit_thetas(bs, ts):
    """Compare Monte-Carlo and analytic results for theta_crit."""
    for b, t in itertools.product(bs, ts):
        if b == t == 0.0:
            continue
        print('b={}  t={}'.format(b, t))
        for clvl in (0.5, 0.6832, 0.9, 0.95, 0.99):
            an = critical_value(b, t, clvl)
            mc = critical_theta_mc(b, t, clvl, 10000)
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
