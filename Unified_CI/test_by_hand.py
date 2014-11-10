import numpy as np
import pylab as P
import statsmodels.api as sm
from itertools import product, izip, chain
import multiprocessing

from unified_ci.gaussian import *
from unified_ci import gaussian
from unified_ci.tools import conservative_quantile

def critical_value_MC(mu_test, sigma, alpha, N_mc=10000):
    """Estimate the critical -2log(likelihood ratio) value from MC sampling."""
    lrs = np.asarray([neg_log_likelihood_ratio(mu_test, xi, sigma) for xi in stats.norm.rvs(loc=mu_test, scale=sigma, size=N_mc)])

    return conservative_quantile(lrs, 1-alpha)

def compare_cdf(mu, sigma, lmax=20, N_mc=10000):
    """Compare visually the CDF results from pseudo-analytic calculations and from MC sampling."""
    l = np.linspace(0, lmax, 100)
    acdf = [neg_log_likelihood_ratio_CDF(li, mu, sigma) for li in l]
    lams_mc = [neg_log_likelihood_ratio(mu, xi) for xi in stats.norm.rvs(loc=mu, scale=sigma, size=N_mc)]
    ecdf = sm.distributions.ECDF(lams_mc)
    P.plot(l, acdf, 'g-', label='analytic')
    P.plot(l, ecdf(l), 'b--', label='MC')
    P.legend(loc="best")

def iflatten(*its):
    """Flatten 

    Takes an arbitrary number of iterators and yields each item

    """
    for items in izip(a, b):
        for i in items:
            yield i

def generate_FC_tables():
    clvls = (0.6827, 0.90, 0.95, 0.99)
    tmpl = "{: .1f}  " + "  ".join(["{:.3f},{:.3f}"] * len(clvls))
    for xi in np.arange(-3.0, 3.15, 0.1):
        ll = [gaussian.lower_limit(xi, 1.0, cl) for cl in clvls]
        ul = [gaussian.upper_limit(xi, 1.0, cl) for cl in clvls]
        print tmpl.format(xi, *list(chain.from_iterable(izip(ll, ul))))


def test_coverage(mu_true, sigma, cl, n_test=1000):
    def test():
        x = np.random.randn() * sigma + mu_true
        ll = lower_limit(x, sigma, cl)
        ul = upper_limit(x, sigma, cl)
        return ll <= mu_true <= ul

    succ = sum(test() for i in xrange(n_test))
    return 1.0 * succ / n_test, np.sqrt(cl * (1.0 - cl) / n_test)

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
