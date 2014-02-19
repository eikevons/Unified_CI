"""
Estimate the coverage of hybrid confidence intervals.

Usage: hybrid_coverage.py theta b gamma cl ntest
"""
from __future__ import division
import sys
from numpy.random import poisson
from hybrid import confidence_interval

if len(sys.argv) != 6:
    sys.exit(__doc__)

theta = float(sys.argv[1])
b = float(sys.argv[2])
gamma = float(sys.argv[3])
cl = float(sys.argv[4])
ntest = int(sys.argv[5])

N_MC = 10000

ncov = 0
for i in xrange(ntest):
    # signal region
    n = poisson(theta + b)
    # backgrund region
    m = poisson(gamma * b)
    t0, t1 = confidence_interval(n, m, gamma, cl, N_MC)
    if t0 <= theta <= t1:
        ncov += 1

print 'target confidence level: %.6e' % cl
print 'observed coverage: %.12e = %d / %d' % (ncov/ntest, ncov, ntest)
