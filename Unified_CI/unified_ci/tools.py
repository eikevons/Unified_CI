"""\
Common tools for interval estimation
------------------------------------

.. autofunction:: bisect

.. autofunction:: conservative_quantile
"""
import numpy as np
from scipy.stats import itemfreq

def bisect(f, a, b, xtol=1e-2, ftol=1e-6, args=None):
    """Bisection search for root of `f` in interval `[a, b]`.

    If there's a sub-intervall where `f(x) == 0` the edge nearest to `a` is
    returned.

    Parameters
    ----------
    f : callable
        Scalar function callable as `f(x, *args)`.
    a, b : float
        The initial interval for root search. `f(a)` and `f(b)` must have
        opposite signs.
    xtol, ftol : float, optimal
        Absolute convergence criteria. Convergence is assumed if::
            abs(a-x) <= xtol
        or::
            abs(f(a) - f(b)) <= ftol
    args : tuple, optional
        Additional arguments for `f`.

    Returns
    -------
    r : float
        The root nearest to `a`.
    """
    if args is None:
        args = ()

    fa = f(a, *args)
    fb = f(b, *args)
    if np.sign(fa) == np.sign(fb):
        raise ValueError("f(a) and f(b) must have opposite sign: f(%r)=%r  f(%r)=%r" % (a, fa, b, fb))

    while abs(a-b) > xtol and abs(fa - fb) > ftol:
        t = 0.5 * (a+b)
        ft = f(t, *args)
        if np.sign(ft) == np.sign(fa):
            a = t
            fa = ft
        else:
            b = t
            fb = ft

    return a

def conservative_quantile(x, p):
    """Calculate upper/lower tail conservative quantiles.

    Conservative lower quantile means that the lower tail probability is
    ensured (:math:`Pr{x_i <= q} >= p`).

    Conservative upper quantile means that the upper tail probability is
    ensured (:math:`Pr{x_i <= q} <= p <=> Pr{x_i >= q} >= (1-p)`).

    Parameters
    ----------
    x : ndarray-like
        The data sample.
    p : float, ndarray
        The target probabilities `0 <= p <= 1`.
        If `p < 0` conservative upper quantiles are returned.

    Returns
    -------
    q : float or ndarray of floats
        The quantile values.
    n : int
        The number of unique values in `x`.
    """
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError("Data sample must be 1-dim")

    items, freqs = itemfreq(x).T
    cumfreq = np.cumsum(freqs / x.size)
    if np.isscalar(p):
        i = np.where(cumfreq >= abs(p))[0][0]
        if p < 0 and cumfreq[i] != abs(p):
            i -= 1
        return items[i], len(items)
    else:
        r = np.empty(len(p), dtype=x.dtype)
        for k, pk in enumerate(p):
            i = np.where(cumfreq >= abs(pk))[0][0]
            if pk < 0 and cumfreq[i] != abs(pk):
                i -= 1
            r[k] = items[i]
        return r, len(items)
