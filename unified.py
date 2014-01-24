"""
Unified confidence limits for a Poissonian signal + *known* background
"""
from __future__ import print_function
import numpy as np

from tools import conservative_quantile, bisect


def fit_theta(n, b):
    """The positive-confined best-fit for poissonian signal parameter theta."""
    return np.fmax(0.0, n-b)


def likelihood_ratio(n, b, t):
    """The likelihood ratio for a theta-value of `t` with background `b` and measured value `n`."""
    t_fit = fit_theta(n, b)
    return ((t+b)/(t_fit+b))**n * np.exp(t_fit - t)


def critical_theta(b, t, clvl, N_mc):
    """Estimate the critical c_theta value."""
    n_sample = np.random.poisson(t+b, size=N_mc)
    lr = likelihood_ratio(n_sample, b, t)

    return conservative_quantile(lr, -(1.0 - clvl))[0]


def confidence_interval(n, b, clvl, N_mc):
    t_best = fit_theta(n, b)

    cache = {}
    def f(t, N):
        k = (t, N)
        if k not in cache:
            cache[k] = poisson_lr(n, b, t) - poisson_c_t(b, t, clvl, N)
        return cache[k]

    # Use full MC precision for searching the lower limit.
    if t_best == 0.0 or f(0, N_mc) >= 0:
        t0 = 0
    else:
        # Use full MC precision for searching the actual limit.
        # NOTE: The standard functions do not work here, because there are
        # whole intervals where `f(t) == 0` and we need the outer bounds of
        # these intervals.
        # t0 = optimize.brentq(f, 0, t_best)
        # t0 = optimize.bisect(f, 0, t_best, xtol=1e-4)
        # So we have to use a hand-crafted root-finding.
        t0 = bisect(f, 0, t_best, args=(N_mc,))

    u = t_best
    v = max(1.0, 2*u)
    # Don't waste MC time for finding the bounds of the search intervals.
    N_coarse = 1000
    while f(v, N_coarse) >= 0:
        u = v
        v = 2*u

    # Use full MC precision for searching the actual limit.
    try:
        t1 = bisect(f, v, u, args=(N_mc,))
    except ValueError:
        # Error is raised if at high-precision `sign(f(u)) == sign(f(v))`.
        # Restart searching bounds with high MC precision if 
        u = t_best
        while f(v, N_mc) >= 0:
            v = 2*v
        t1 = bisect(f, v, u, args=(N_mc,))

    return t0, t1


def test_coverage(b, t, clvl, N_tests, N_mc=10000):
    sample = np.random.poisson(t+b, size=N_tests)
    n_cov = 0
    for n in sample:
        t0, t1 = poisson_cl(n, b, clvl, N_mc)
        if t0 <= t <= t1:
            n_cov += 1
    print("Target clvl: {0:.4%} => {1}/{2} = {3:.4%}".format(clvl, n_cov, N_tests, 1.0*n_cov/N_tests))
    return n_cov
