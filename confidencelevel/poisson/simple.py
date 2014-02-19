"""
Confidence limits for a Poissonian signal + *known* background

This gives the same results as the unified algorithm as tabulated in
Feldman+Cousins(1998).
"""
from __future__ import print_function, division
import itertools
import operator
import numpy as np
from scipy import stats

from tools import conservative_quantile, bisect


def fit_theta(n, b):
    """The positive-confined best-fit for poissonian signal parameter theta."""
    return np.fmax(0.0, n-b)


def likelihood_ratio(n, b, t):
    """The likelihood ratio for a theta-value of `t` with background `b` and measured value `n`."""
    t_fit = fit_theta(n, b)
    return ((t+b)/(t_fit+b))**n * np.exp(t_fit - t)


def critical_theta(b, t, clvl):
    poi = stats.poisson(b+t)
    alpha = 1.0 - clvl

    # Find n_max
    # criteria are
    #   P[n > n_max] < p_thresh
    #   P[n = n_max] < p_thresh
    #   P[n >= n_max] < p_thresh 
    #  ==> SF(n_max-1) < p_thresh
    #  ==> n_max = 1 + ISF(p_thresh)
    # NOTE: sf(M) = P[n > M] = P[n >= M+1]
    p_thresh = min(alpha, poi.pmf(0))
    # try:
    n_max = int(poi.isf(p_thresh)) + 1
    # except Exception, e:
        # print(e)
        # print(b, t, clvl)
        # print(p_thresh)
        # return -1
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
    t_best = fit_theta(n, b)

    cache = {}
    def f(t):
        if t not in cache:
            cache[t] = likelihood_ratio(n, b, t) - critical_theta(b, t, clvl)
        return cache[t]

    if t_best == 0.0:
        t0 = 0.0
    else:
        # Use full MC precision for searching the actual limit.
        # NOTE: The standard functions do not work here, because there are
        # whole intervals where `f(t) == 0` and we need the *inner* bounds of
        # these intervals.
        # t0 = optimize.brentq(f, 0, t_best)
        # t0 = optimize.bisect(f, 0, t_best, xtol=1e-4)
        # So we have to use a hand-crafted root-finding.
        t0 = bisect(f, t_best, 0)

    u = t_best
    v = max(1.0, 2*u)
    while f(v) >= 0:
        u = v
        v = 2*u
    t1 = bisect(f, u, v)

    return t0, t1


##############################
## TESTS
def critical_theta_mc(b, t, clvl, N_mc):
    """Estimate the critical c_theta value."""
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
            an = critical_theta(b, t, clvl)
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
