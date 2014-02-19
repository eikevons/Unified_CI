"""
Estimate a confidence interval for mean of a Normal distributed observable
with known variance *without* constraints using critical values for the
likelihood ratio.

This is meant for testing/comparison with expected interval::

    => gives the expected interval (compare the crossing of the LR_crit
       (blue dotted) with likelihood ratio (magenta dashed) and the expected
       CI boundaries (black vertical lines).
"""
import numpy as np
from scipy import stats

import pylab as P

t = 10.0
sigma = 9.0

alpha = 0.01

# simple case with no constraints
ll = stats.norm.ppf(0.5*alpha, loc=t, scale=sigma)
ul = stats.norm.isf(0.5*alpha, loc=t, scale=sigma)

x = np.linspace(t - 2 * (t - ll), t + 2 * (ul - t))
P.plot(x, stats.norm.pdf(x, loc=t, scale=sigma), label='normal pdf')
P.axvline(ll, ls='-', c='k', label='expected CI boundaries')
P.axvline(ul, ls='-', c='k')

def lr_simple(x):
    return np.exp(-0.5 / sigma**2 * (t-x)**2)

P.plot(x, lr_simple(x), '--m', label='likelihood ratio')

xi = stats.norm.isf(0.5*alpha)
crit_al = np.exp(-0.5 * xi**2)
P.axhline(crit_al, ls=':', c='b', label='crit. LR')

P.legend(loc='best')
P.xlabel('observable or mean')
P.ylabel('pdf or likelihood ratio')

P.show()
