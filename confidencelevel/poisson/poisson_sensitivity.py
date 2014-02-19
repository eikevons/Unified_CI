from os.path import exists
import pylab as P
import numpy as np
from poisson_lh_conflimits import poisson_cl


n_mc = 1000
b_max = 50
cachefile = __file__ + '.cache'

def poisson_ul(b):
    ll, ul = zip(*[poisson_cl(n, b, 0.95, 1000) for n in np.random.poisson(lam=b, size=n_mc)])
    return np.mean(ul)


def mk_cache():
    print 'Filling cache'
    b = np.linspace(0, b_max)[1:]
    t = np.empty((b.size, 2))
    for i, b in enumerate(b):
        ul = poisson_ul(b)
        t[i] = (b, ul)
    np.savetxt(cachefile, t)

if __name__ == "__main__":
    if not exists(cachefile):
        mk_cache()
    b, ul = np.loadtxt(cachefile).T
    P.loglog(b, ul - ul[0])
    P.loglog(b, P.sqrt(b))
    P.xlabel("background b")
    P.ylabel("delta sensitivity")
    P.show()
