"""\
Poissonian Unified Confidence Intervals with *unknown* background
-----------------------------------------------------------------

Unified confidence limits for a Poissonian signal with *unknown* background
estimated by hybrid resampling.

This implementation follows [Sen+Walker+Woodroofe. (2009) Stat.Sin.(19) 301--314]

Functions of Interest
---------------------

* :func:`confidence_interval`
"""
# Use float division
from __future__ import print_function, division, absolute_import
import numpy as np

from .tools import conservative_quantile, bisect


def global_fit_b(n, m, gamma):
    """Global fit of the background rate parameter.

    Parameters
    ----------
    n : int
        The number of counts in the signal region.
    m : int
        Number of counts in the background region.
    gamma : float
        The ratio of background to signal region.

    Returns
    -------
    b_fit : float
        The fitted background-rate value.
    """
    return (m + n - global_fit_theta(n, m, gamma)) / (1.0 + gamma)


def global_fit_theta(n, m, gamma):
    """Global fit of the signal parameter.

    The global fit value is given by :math:`max(n - m / \gamma, 0)`.

    Parameters
    ----------
    n : int
        The number of counts in the signal region.
    m : int
        Number of counts in the background region.
    gamma : float
        The ratio of background to signal region.

    Returns
    -------
    theta_fit : float
        The fitted theta value.
    """
    return np.fmax(0.0, n - m/gamma)


def local_fit_b(n, m, theta, gamma):
    """Local fit of the background rate parameter for given signal parameter.

    Parameters
    ----------
    n : int
        The number of counts in the signal region.
    m : int
        Number of counts in the background region.
    theta : float
        The assumed signal rate.
    gamma : float
        The ratio of background to signal region.

    Returns
    -------
    b_fit : float
        The fitted background-rate value assuming signal rate `theta`.
    """
    return (m + n - (1+gamma)*theta + np.sqrt( ((1+gamma)*theta - m - n)**2 + 4*(1+gamma)*m*theta )) / (2 * (1+gamma))


def likelihood_ratio(n, m, theta, gamma):
    """Calculate the likelihood ratio.

    Parameters
    ----------
    n : int
        The number of counts in the signal region.
    m : int
        Number of counts in the background region.
    theta : float
        The assumed signal rate.
    gamma : float
        The ratio of background to signal region.

    Returns
    -------
    lr : float
        The likelihood ratio.
    """
    bh = global_fit_b(n, m, gamma)
    th = global_fit_theta(n, m, gamma)
    bhh = local_fit_b(n, m, theta, gamma)

    return (bhh/bh)**m * ((theta+bhh)/(th+bh))**n * np.exp(n + m - (1+gamma)*bhh - theta)


def critical_value(n, m, theta, gamma, clvl, N_mc):
    """Calculate the critical likelihood ratio value using hybrid resampling.

    Parameters
    ----------
    n : int
        Number of counts in the signal region.
    m : int
        Number of counts in the background region.
    theta : float
        The expectation value of the signal rate.
    gamma : float
        The ratio of background to signal region.
    clvl : float
        The target confidence level.
    N_mc : int
        The number of MC toy experiments used to estimate the critical
        likelihood ratio.

    Returns
    -------
    cv : floate
        The critical likelihood ratio.
    """
    bhh = local_fit_b(n, m, theta, gamma)

    ns = np.random.poisson(theta + bhh, size=N_mc)
    ms = np.random.poisson(gamma*bhh, size=N_mc)
    l = likelihood_ratio(ns, ms, theta, gamma)
    return conservative_quantile(l, -(1.0 - clvl))[0]


def confidence_interval(n, m, gamma, clvl, N_mc):
    """Calculate unified confidence interval for Poissonian signal with unknown background.

    Parameters
    ----------
    n : int
        Number of counts in the signal region.
    m : int
        Number of counts in the background region.
    gamma : float
        The ratio of background to signal region.
    clvl : float
        The target confidence level.
    N_mc : int
        The number of MC toy experiments used to estimate the critical
        likelihood ratio.

    Returns
    -------
    ll, ul : float
        Lower and upper limits of the confidence interval.
    """
    theta_best = global_fit_theta(n, m, gamma)

    cache = {}
    def f(theta, n_mc):
        k = (theta, n_mc)
        if k not in cache:
            ret =  likelihood_ratio(n, m, theta, gamma) - critical_value(n, m, theta, gamma, clvl, n_mc)
            cache[k] = ret
        else:
            ret = cache[k]
        return ret

    t0 = 0.0
    if theta_best > 0 and f(0, N_mc) < 0:
        t0 = bisect(f, theta_best, 0, args=(N_mc,))

    u = theta_best
    v = max(1, 2*u)
    while f(v, N_mc) > 0:
        u = v
        v = 2*u

    t1 = bisect(f, u, v, args=(N_mc,))
    return t0, t1


def mk_plot(ts, N_mc):
    """
    Reproduce Fig.4.2 from [Sen+Walker+Woodroofe. (2009) Stat.Sin.(19) 301--314]
    """
    import pylab as P
    c_ts = np.empty_like(ts)
    ls = np.empty_like(ts)

    n = 0
    m = 23
    g = 4
    a = 0.10

    for i, t in enumerate(ts):
        c_ts[i] = critical_value(n, m, t, g, a, N_mc)
        ls[i] = likelihood_ratio(n, m, t, g)
    P.plot(ts, c_ts, "r-")
    P.plot(ts, ls, "b-")
    P.xlabel("$\\theta$")
    P.ylabel("$\\Lambda_\\theta$ and $c_\\theta$")
    P.show()



# if __name__ == "__main__":
    # # print("Comparison with numerical values in [http://www.stat.columbia.edu/~bodhi/research/UnifiedCI.pdf ; Fig.4.2]")
    # # print(c_t(0, 23, 0.82, 6.0, 0.1))
    # # print(likelihood_ratio(0, 23, 0.82, 6.0))
    # mk_plot(np.linspace(0.0, 2.5, 2501), 10000)

