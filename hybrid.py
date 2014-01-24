import numpy as np

from tools import conservative_quantile


def b_hat(n, m, g):
    return (m + n - t_hat(n, m, g)) / (1.0 + g)

def t_hat(n, m, g):
    return np.fmax(0.0, n - m/g)

def b_hathat(n, m, t, g):
    return (m + n - (1+g)*t + np.sqrt( ((1+g)*t - m - n)**2 + 4*(1+g)*m*t )) / (2 * (1+g))


def likelihood_ratio(n, m, t, g):
    bh = b_hat(n, m, g)
    th = t_hat(n, m, g)
    bhh = b_hathat(n, m, t, g)

    return (bhh/bh)**m * ((t+bhh)/(th+bh))**n * np.exp(n + m - (1+g)*bhh - t)


def c_t(n, m, t, g, clvl, N_mc=10000):
    bhh = b_hathat(n, m, t, g)

    ns = np.random.poisson(t + bhh, size=N_mc)
    ms = np.random.poisson(g*bhh, size=N_mc)
    l = likelihood_ratio(ns, ms, t, g)
    # return np.percentile(l, 100*a)
    return conservative_quantile(l, -(1.0 - clvl)

def mk_plot(ts, N_mc):
    import pylab as P
    c_ts = np.empty_like(ts)
    ls = np.empty_like(ts)

    n = 0
    m = 23
    g = 4
    a = 0.10

    for i, t in enumerate(ts):
        c_ts[i] = c_t(n, m, t, g, a, N_mc)
        ls[i] = likelihood_ratio(n, m, t, g)
    P.plot(ts, c_ts, "r-")
    P.plot(ts, ls, "b-")
    P.xlabel("$\\theta$")
    P.ylabel("$\\Lambda_\\theta$ and $c_\\theta$")
    P.show()


if __name__ == "__main__":
    # print("Comparison with numerical values in [http://www.stat.columbia.edu/~bodhi/research/UnifiedCI.pdf ; Fig.4.2]")
    # print(c_t(0, 23, 0.82, 6.0, 0.1))
    # print(likelihood_ratio(0, 23, 0.82, 6.0))
    mk_plot(np.linspace(0.0, 2.5, 2501), 10000)

