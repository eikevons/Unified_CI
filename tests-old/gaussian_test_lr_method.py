"""
Estimate a confidence interval for mean of a Normal distributed observable
with known variance *without* constraints using critical values for the
likelihood ratio.

This is meant for testing/comparison with expected interval::

    => gives the expected interval (compare the crossing of the LR_crit
       (blue dotted) with likelihood ratio (magenta dashed) and the expected
       CI boundaries (black vertical lines).
"""
import sys
import numpy as np
from scipy import stats

import pylab as P

try:
    from confidencelevel.gaussian import GaussianUnconstrainedInterval
    print('Using installed confidencelevel package')
except ImportError:
    sys.path.append('..')
    from confidencelevel.gaussian import GaussianUnconstrainedInterval
    print('Using source confidencelevel package')


measured = 10.0
sigma = 9.0

cl = 0.99
alpha = 1.0 - cl

# simple case with no constraints
ll = stats.norm.ppf(0.5*alpha, loc=measured, scale=sigma)
ul = stats.norm.isf(0.5*alpha, loc=measured, scale=sigma)

x = np.linspace(measured - 2 * (measured - ll), measured + 2 * (ul - measured))
P.plot(x, stats.norm.pdf(x, loc=measured, scale=sigma), label='normal pdf')
P.axvline(ll, ls='-', c='k', label='expected CI boundaries')
P.axvline(ul, ls='-', c='k')

ci = GaussianUnconstrainedInterval(measured=measured, sigma=sigma, cl=cl)
llci, ulci = ci.confidence_interval()

P.axvline(llci, ls='--', c='r', label='calculated CI boundaries')
P.axvline(ulci, ls='--', c='r')

P.plot(x, ci.likelihood_ratio(x), '--m', label='likelihood ratio')

# xi = stats.norm.isf(0.5*alpha)
# crit_al = np.exp(-0.5 * xi**2)
crit_al = ci.critical_value('unused')
P.axhline(crit_al, ls=':', c='b', label='crit. LR')

P.legend(loc='best')
P.xlabel('observable or mean')
P.ylabel('pdf or likelihood ratio')

P.show()
