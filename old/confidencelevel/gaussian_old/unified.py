"""
Estimate confidence intervals for mean of a Normal distributed observable
with known variance and constraint (mu >= 0) using critical values for the
likelihood ratio.

This is meant for comparison with Feldman+Cousins paper.
"""
import numpy as np
from scipy import stats

import pylab as P

from unconstrained import GaussianUnconstrainedInterval

class GaussianUnifiedInterval(GaussianUnconstrainedInterval):
    def calc_best_fit(self, measured):
        return max(0.0, measured)

    def likelihood_ratio(self, mu_test):
        a = -0.5 * (self.measured - mu_test)**2 / self.sigma**2
        if self.measured < 0:
            a += 0.5 * self.measured**2 / self.sigma**2
        return np.exp(a)

def unified_ci(measured, sigma, cl):
    ci = GaussianUnifiedInterval(measured=measured, sigma=sigma, cl=cl)
    return ci.confidence_interval()



def test_coverage(mu_true, sigma, cl, n_test=1000):
    def test():
        x = np.random.randn() * sigma + mu_true
        ll, ul = unified_ci(x, sigma, cl)
        # ll = lower_limit(x, sigma, cl)
        # ul = upper_limit(x, sigma, cl)
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
