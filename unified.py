"""
Unified confidence limits for a Poissonian signal + known background
"""
from __future__ import print_function
import numpy as np

from tools import conservative_quantile, bisect


def poisson_fit(n, b):
    """The positive-confined best-fit for poissonian signal parameter theta."""
    return np.fmax(0.0, n-b)


def poisson_lr(n, b, t):
    """The likelihood ratio for a theta-value of `t` with background `b` and measured value `n`."""
    t_fit = poisson_fit(n, b)
    return ((t+b)/(t_fit+b))**n * np.exp(t_fit - t)


def poisson_c_t(b, t, clvl, N_mc):
    """Estimate the critical c_theta value."""
    n_sample = np.random.poisson(t+b, size=N_mc)
    lr = poisson_lr(n_sample, b, t)

    # hand-crafted quantile
    # v, f = itemfreq(lr).T
    # cumf = np.cumsum(f/N_mc)
    # i = np.where(cumf <= clvl)[0][-1] + 1
    # # print(">>", t, i)
    # # print("  ", v[[i-1,i,i+1]])
    # # print("  ", cumf[[i-1, i, i+1]])
    # return v[i]

    # return np.percentile(lr, 100.0 * (1.0 - clvl))
    # return mquantiles(lr, clvl)[0]
    return conservative_quantile(lr, -(1.0 - clvl))[0]


def poisson_cl(n, b, clvl, N_mc):
    t_best = poisson_fit(n, b)

    N = N_mc
    cache = {}
    def f(t):
        if t not in cache:
            cache[t] = poisson_lr(n, b, t) - poisson_c_t(b, t, clvl, N)
        return cache[t]

    # print("f(t_best)", f(t_best))

    # Don't waste MC time for finding the bounds of the search intervals.
    N = N_mc
    if t_best == 0.0 or f(0) >= 0:
        t0 = 0
    else:
        # Use full MC precision for searching the actual limit.
        N = N_mc
        # NOTE: The standard functions do not work here, because there are
        # whole intervals where `f(t) == 0` and we need the outer bounds of
        # these intervals.
        # t0 = optimize.brentq(f, 0, t_best)
        # t0 = optimize.bisect(f, 0, t_best, xtol=1e-4)
        # So we have to use a hand-crafted root-finding.
        t0 = bisect(f, 0, t_best)

    u = t_best
    v = max(1.0, 2*u)
    # print("++", u, v)
    # Don't waste MC time for finding the bounds of the search intervals.
    # N = 1000
    while f(v) >= 0:
        u = v
        v = 2*u
        # print("++", u, v)
    # Use full MC precision for searching the actual limit.
    N = N_mc
    t1 = bisect(f, v, u)
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
