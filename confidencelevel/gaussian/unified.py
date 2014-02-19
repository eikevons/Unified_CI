"""
Estimate confidence intervals for mean of a Normal distributed observable
with known variance and constraint (mu >= 0) using critical values for the
likelihood ratio.

This is meant for comparison with Feldman+Cousins paper.
"""
import numpy as np
from scipy import stats
from scipy.optimize import bisect

import pylab as P

from tools import conservative_quantile

def critical_value_cdf_analytic(l, mu, sigma):
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

def compare_cdf(mu, sigma, lmax=20, N_mc=10000):
    import statsmodels.api as sm
    l = np.linspace(0, lmax, 100)
    acdf = [critical_value_cdf_analytic(li, mu, sigma) for li in l]
    lams_mc = -2 * np.log([likelihood_ratio(mu, xi, sigma) for xi in stats.norm.rvs(loc=mu, scale=sigma, size=N_mc)])
    ecdf = sm.distributions.ECDF(lams_mc)
    P.plot(l, acdf, 'g-')
    P.plot(l, ecdf(l), 'b--')

def critical_value_analytic(mu_test, sigma, alpha):
    """Estimate the critical LR value from bisecting CDF.

    See
    ---
    Rotes Buch V, p.57--59
    """
    if mu_test > 0.0:
        def target(cv):
            if cv == 0:
                return -1
            l = -2*np.log(cv)
            return 1 - alpha - critical_value_cdf_analytic(l, mu_test, sigma)
        sol = bisect(target, 0.0, 1.0)
    else:
        p_gz = 1 - stats.norm.cdf(0, loc=mu_test, scale=sigma)
        t = 1.0 - alpha / p_gz
        if t > 0:
            u = stats.chi2.ppf(1.0 - alpha / p_gz, 1)
            sol = np.exp(-0.5 * u)
        else:
            print 'encountered questionable value < 0.0'
            sol = 1.0

    return sol

def critical_value_MC(mu_test, sigma, alpha, N_mc=10000):
    """Estimate the critical LR value from MC sampling."""
    lrs = np.asarray([likelihood_ratio(mu_test, xi, sigma) for xi in stats.norm.rvs(loc=mu_test, scale=sigma, size=N_mc)])

    return conservative_quantile(lrs, -alpha)

critical_value = critical_value_analytic

def likelihood_ratio(mu_test, x, sigma):
    """Calculate the likelihood ratio.

    Parameters
    ----------
    mu_test : float
        The mean-parameter value.
    x : float
        The measured value.
    sigma : float
        The std.deviation of the distribution.
    """
    a = -0.5 * (x - mu_test)**2 / sigma**2
    if x < 0:
        a += 0.5 * x**2 / sigma**2
    return np.exp(a)

def best_fit(x):
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
    mu_hat = best_fit(x)

    def diff(mu):
        return likelihood_ratio(mu, x, sigma) - critical_value(mu, sigma, alpha)
    assert diff(mu_hat) > 0

    n = 1
    while diff(mu_hat + n * sigma) > 0:
        n *= 2

    sol = bisect(diff, mu_hat, mu_hat + n * sigma)
    return sol

def lower_limit(x, sigma, cl):
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
    alpha = 1.0 - cl
    mu_hat = best_fit(x)

    if mu_hat == 0.0:
        return 0.0

    def diff(mu):
        return likelihood_ratio(mu, x, sigma) - critical_value(mu, sigma, alpha)
    assert diff(mu_hat) > 0

    if diff(0.0) >= 0.0:
        return 0.0

    sol = bisect(diff, 0.0, mu_hat)
    return sol

def test_coverage(mu_true, sigma, cl, n_test=1000):
    def test():
        x = np.random.randn() * sigma + mu_true
        ll = lower_limit(x, sigma, cl)
        ul = upper_limit(x, sigma, cl)
        return ll <= mu_true <= ul

    succ = sum(test() for i in xrange(n_test))
    return 1.0 * succ / n_test, np.sqrt(cl * (1.0 - cl) / n_test)




from itertools import product
import multiprocessing

def _mp_target(args):
    mu, sigma, cl, n_test = args
    cl_obs, cl_err = test_coverage(mu, sigma, cl, n_test)
    return mu, sigma, cl_obs, cl_err

def mk_coverage_grid(mus, sigmas, cl, n_test):
    print '# target CL=%.5e' % cl
    print '# N_test=%d' % n_test

    arggen = ((mu, sigma, cl, n_test) for (mu, sigma) in product(mus, sigmas))
    p = multiprocessing.Pool()
    res = p.imap(_mp_target, arggen, 1)

    for mu, sigma, cl_obs, cl_err in res:
        print '%.5e  %.5e  %.5e  %.5e' % (mu, sigma, cl_obs, cl_err)

def load_cov_test(path):
    with open(path) as f:
        l = f.readline()
        target_cl = float(l.strip().split('=')[1])
        l = f.readline()
        n_test = int(l.strip().split('=')[1])
        mu, sigma, cl_obs, cl_err = np.loadtxt(f).T
    return target_cl, n_test, mu, sigma, cl_obs, cl_err

def plot_cov_grid(path):
    target_cl, n_test, mu, sigma, cl_obs, cl_err = load_cov_test(path)
    idx_good_exp = (np.abs(cl_obs - target_cl) < cl_err)
    idx_good_obs = (np.abs(cl_obs - target_cl) < cl_obs.std(ddof=1))

    P.figure()
    P.title(path)
    P.scatter(mu[idx_good_exp], sigma[idx_good_exp], marker='o', color='g')
    P.scatter(mu[-idx_good_exp], sigma[-idx_good_exp], marker='o', color='r')
    P.xlabel('mu_true')
    P.ylabel('sigma')
    P.text(0.99, 0.99,
            'average coverage: %.4f'
            '\nobs. (exp.) std. dev: %.4f (%.4f)'
            '\nmin, max: %.2f, %.2f'
            '\ncoverage inside expected 1-sigma: %.4f'
            '\ncoverage inside observed 1-sigma: %.4f' % (
                cl_obs.mean(),
                cl_obs.std(ddof=1), cl_err[0],
                cl_obs.min(), cl_obs.max(),
                1.0*idx_good_exp.sum()/idx_good_exp.size,
                1.0*idx_good_obs.sum()/idx_good_obs.size),
           ha='right', va='top',
           transform=P.gca().transAxes)

def plot_cov_distribution(path):
    target_cl, n_test, mu, sigma, cl_obs, cl_err = load_cov_test(path)
    P.figure()
    P.title(path)
    n = np.arange(0, n_test + 1)
    pmf = stats.binom.pmf(n, n_test, target_cl)
    P.plot(n, pmf, 'go', label='exp. binom pmf')
    entries, bins, patches = P.hist(cl_obs * n_test, int(0.1*cl_obs.size), histtype='stepfilled', normed=1, label='obs. coverage')
    P.xlabel('#covered')
    P.ylabel('pmf')
    P.legend(loc='best')
    P.xlim(bins[[0,-1]])
    P.ylim((0, 1.2 * max(pmf.max(), entries.max())))

# if __name__ == "__main__":
    # mk_coverage_grid(np.linspace(0, 5, 20), [0.1, 0.5, 0.8, 1.0, 2.0, 10.0], 0.9, 1000)
