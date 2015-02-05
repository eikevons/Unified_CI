"""\
Estimate the coverage of unified confidence intervals.

Usage: coverages_simple.py COMMAND ARGS

Possible COMMANDs:
    simple_poisson  to calculate a coverage grid for simple_poisson
    simple_gauss    to calculate a coverage grid for simple_gaussian
    hybrid_poisson  to calculate a coverage grid for hybrid_poisson
    plot            to plot a coverage grid file
    hist            to plot a histogram of a coverage grid file
    cdf             to plot a cdf histogram of a coverage grid file

Expected ARGS:
    - simple_poisson: THETAs Bs CLs NTEST
    - simple_gauss: MUs SIGMAs CLs NTEST
    - hybrid_poisson: THETAs Bs GAMMAs CLs NTEST
    - plot, hist, cdf: FILEPATH

The PARAMs are comma separated lists of the respective parameter values or
colon separated triples of `np.linspace` arguments:
    1.0,1.1,1.3,1.4 to test the specified values
or
    1.0:5.0:10 to test a grid of 10 points between 1.0 and 5.0

Output Data
-----------
The header lines are prefixed with '#'. Comments can be added or lines can
be commented by prefixing with '!'. Preceding whitespaces are *not*
stripped!

"""
from __future__ import division, print_function
import sys
from itertools import product, cycle
import multiprocessing
import numpy as np

import context
from unified_ci.simple_poisson import confidence_interval as simpoi_ci
from unified_ci.simple_gaussian import confidence_interval as simgau_ci
from unified_ci.hybrid_poisson import confidence_interval as hybpoi_ci


# Argument handling
def parse_arg(param, parname='unknown'):
    """Parse PARAM argument."""
    try:
        if ":" in param:
            a, b, c = param.split(":")
            a = float(a)
            b = float(b)
            c = int(c)
            # Convert to plain list to allow eval'ing
            return list(np.linspace(a, b, c))
        else:
            return [float(t) for t in param.split(",")]
    except Exception as e:
        sys.exit("Failed to parse argument for parameter {}: '{}'\n{}".format(parname, param, e))

# Confidence grid computation
def _mp_target_simple_poisson(args):
    theta, b, cl, n_test = args
    n_succ = 0
    for i in range(n_test):
        n = np.random.poisson(theta + b)
        ll, ul = simpoi_ci(n, b, cl)
        if ll <= theta <= ul:
            n_succ += 1
    return theta, b, cl, n_succ

def simple_poisson(thetas, bs, cls, ntest):
    theta = parse_arg(thetas, "THETAs")
    b = parse_arg(bs, "Bs")
    cl = parse_arg(cls, "CLs")
    try:
        N_mc = int(ntest)
    except ValueError:
        sys.exit("Failed to parse argument for parameter NTEST: '{}'".format(ntest))

    print("# theta: {0!r}".format(theta))
    print("# b:     {0!r}".format(b))
    print("# cl:    {0!r}".format(cl))
    print("# ntest: {0!r}".format(N_mc))
    print("# theta  b  cl  N_success")

    arggen = ((x, y, z, N_mc) for (x, y, z) in product(theta, b, cl))
    p = multiprocessing.Pool()
    results = p.imap(_mp_target_simple_poisson, arggen)

    r0 = results.next()
    template = "  ".join(((len(r0) - 1) * ("{:.5e}",)) + ("{:d}",))
    print(template.format(*r0))
    for r in results:
        print(template.format(*r))

def _mp_target_simple_gauss(args):
    mu, sigma, cl, n_test = args
    n_succ = 0
    for i in range(n_test):
        x = np.random.randn() * sigma + mu
        ll, ul = simgau_ci(x, sigma, cl)
        if ll <= mu <= ul:
            n_succ += 1
    return mu, sigma, cl, n_succ

def simple_gauss(mus, sigmas, cls, ntest):
    mu = parse_arg(mus, "MUs")
    sigma = parse_arg(sigmas, "SIGMAs")
    cl = parse_arg(cls, "CLs")
    try:
        N_mc = int(ntest)
    except ValueError:
        sys.exit("Failed to parse argument for parameter NTEST: '{}'".format(ntest))

    print("# mu:    {0!r}".format(mu))
    print("# sigma: {0!r}".format(sigma))
    print("# cl:    {0!r}".format(cl))
    print("# ntest: {0!r}".format(N_mc))
    print("# mu  sigma  cl  N_success")

    arggen = ((x, y, z, N_mc) for (x, y, z) in product(mu, sigma, cl))
    p = multiprocessing.Pool()
    results = p.imap(_mp_target_simple_gauss, arggen)

    r0 = results.next()
    template = "  ".join(((len(r0) - 1) * ("{:.5e}",)) + ("{:d}",))
    print(template.format(*r0))
    for r in results:
        print(template.format(*r))

def _mp_target_hybrid_poisson(args):
    theta, b, gamma, cl, n_test = args
    n_succ = 0
    for i in range(n_test):
        n = np.random.poisson(theta + b)
        m = np.random.poisson(gamma * b)
        ll, ul = hybpoi_ci(n, m, gamma, cl, 10000)
        if ll <= theta <= ul:
            n_succ += 1
    return theta, b, gamma, cl, n_succ

def hybrid_poisson(thetas, bs, gammas, cls, ntest):
    theta = parse_arg(thetas, "THETAs")
    b = parse_arg(bs, "Bs")
    gamma = parse_arg(gammas, "GAMMAs")
    cl = parse_arg(cls, "CLs")
    try:
        N_mc = int(ntest)
    except ValueError:
        sys.exit("Failed to parse argument for parameter NTEST: '{}'".format(ntest))

    print("# theta: {0!r}".format(theta))
    print("# b:     {0!r}".format(b))
    print("# gamma: {0!r}".format(gamma))
    print("# cl:    {0!r}".format(cl))
    print("# ntest: {0!r}".format(N_mc))
    print("# theta  b  gamma  cl  N_success")

    arggen = (i + (N_mc,) for i in product(theta, b, gamma, cl))
    p = multiprocessing.Pool()
    results = p.imap(_mp_target_hybrid_poisson, arggen)

    r0 = results.next()
    template = "  ".join(((len(r0) - 1) * ("{:.5e}",)) + ("{:d}",))
    print(template.format(*r0))
    for r in results:
        print(template.format(*r))


# Visualization
def load_coverage_file(path):
    params = {}
    with open(path) as fd:
        # parse header
        for line in fd:
            if line.startswith("!"):
                continue
            if line.startswith("#"):
                if ":" in line:
                    toks = line.split()
                    name = toks[1][:-1]
                    # XXX: This is not really safe, but easy.
                    vals = eval("".join(toks[2:]))
                    params[name] = vals
                else:
                    # Last line of header
                    colnames = line.split()[1:]
                    break

        vals = np.loadtxt(fd, comments="!")
    return params, colnames, vals

def calc_cl_uncertainty(taplrget_cl, N_mc):
    """Calculate uncertainty of CL estimated from binomial distribution."""
    print(target_cl, N_mc)
    return np.sqrt(target_cl * (1 - target_cl) / N_mc)

def plot(path):
    params, colnames, vals = load_coverage_file(path)
    from matplotlib import pyplot as plt

    n_MC = params["ntest"]
    all_target_cls = set()

    plt.figure()
    plt.title("coverage grid {}".format(path))
    # all unique parameter value combinations (the last 2 columns are CL and N_success)
    upars = list(product(*[np.unique(vals[:,i]) for i in range(vals.shape[1]-2)]))
    for i, up in enumerate(upars):
        # build index array for this unique parameter combination
        idx = np.ones_like(vals[:,0], dtype=np.bool)
        for j, p in enumerate(up):
            idx *= (vals[:,j] == p)

        n_points = idx.sum()
        if not n_points:
            print("No data for {0!r}".format(up))
            continue
        n_succ = vals[idx,-1]
        target_cls = vals[idx,-2]
        all_target_cls.update(target_cls)

        dx = min(0.6 / n_points, 0.10)
        x = i + dx * np.arange(n_points) - 0.5 * dx * (n_points - 1)

        plt.vlines(x,
                target_cls * n_MC, n_succ)

        plt.plot(x, n_succ, "xk")


    for target_cl in all_target_cls:
        delta_cl = calc_cl_uncertainty(target_cl, n_MC)
        plt.axhline(target_cl * n_MC, color="g")
        plt.axhspan(n_MC * (target_cl - delta_cl), n_MC * (target_cl + delta_cl), alpha=0.5, color="g")


    plt.xticks(range(len(upars)), [repr(u) for u in upars], rotation=90)
    plt.ylabel("# of properly covering intervals")
    plt.xlabel("-".join(colnames[:-2]) + " combinations")
    plt.xlim((-0.5, len(upars)-0.5))
    plt.yscale("log")
    plt.ylim((0.9 * n_MC * min(all_target_cls), 1.1 * n_MC))
    plt.show()

def hist(path):
    params, colnames, vals = load_coverage_file(path)
    from matplotlib import pyplot as plt
    from scipy.stats import binom

    # Prepare binning
    n_MC = params["ntest"]
    n_min = min(n_MC * min(params["cl"]), vals[:,-1].min())
    bin_centers = np.arange(n_min, n_MC + 1)
    bins = np.empty((bin_centers.size + 1,))
    bins[:bin_centers.size] = bin_centers - 0.5
    bins[-1] = bin_centers[-1] + 0.5

    plt.figure()
    plt.title("coverage histogram {}".format(path))
    ls_col = cycle(product(("solid", "dashed", "dashdot", "dotted"), "rgbmk"))
    for  targ_cl, (ls, col) in zip(params["cl"], ls_col):
        idx = vals[:,-2] == targ_cl
        n_cov = vals[idx,-1]

        plt.hist(n_cov, bins, normed=True, histtype="step", color=col, linestyle=ls, linewidth=3, label="{0:.5f}".format(targ_cl))
        plt.plot(bin_centers, binom.pmf(bin_centers, n_MC, targ_cl), color=col, linestyle=ls, marker="o", mew=0)

    plt.legend(loc="best", title="target CL")
    plt.xlabel("#(covered)")
    plt.ylabel("frequency")
    plt.show()

def cdf(path):
    params, colnames, vals = load_coverage_file(path)
    from matplotlib import pyplot as plt
    from scipy.stats import binom
    from statsmodels.distributions import ECDF

    # Prepare binning
    n_MC = params["ntest"]
    n_min = min(n_MC * min(params["cl"]), vals[:,-1].min())
    bin_centers = np.arange(n_min, n_MC + 1)

    plt.figure()
    plt.title("coverage histogram {}".format(path))
    ls_col = cycle(product(("solid", "dashed", "dashdot", "dotted"), "rgbmk"))
    for  targ_cl, (ls, col) in zip(params["cl"], ls_col):
        idx = vals[:,-2] == targ_cl
        n_cov = vals[idx,-1]
        ecdf = ECDF(n_cov)
        plt.plot(bin_centers, ecdf(bin_centers),
                 color=col, linestyle=ls, linewidth=3,
                 label="{0:.5f}".format(targ_cl))
        plt.plot(bin_centers, binom.cdf(bin_centers, n_MC, targ_cl), color=col, linestyle=ls, marker="o", mew=0)

    plt.legend(loc="best", title="target CL")
    plt.xlabel("#(covered)")
    plt.ylabel("cumulative frequency (CDF)")
    plt.show()

commands = {"simple_poisson": simple_poisson,
            "simple_gauss": simple_gauss,
            "hybrid_poisson": hybrid_poisson,
            "plot": plot,
            "hist": hist,
            "cdf": cdf
           }

def main():
    try:
        cmd = sys.argv[1]
        args = sys.argv[2:]
    except IndexError:
        sys.exit("\n\n".join((__doc__, "Bad arguments: '{}'".format(" ".join(sys.argv[1:])))))

    if cmd not in commands:
        sys.exit("\n\n".join((__doc__, "Unknown command: '{}'".format(cmd))))

    try:
        commands[cmd](*args)
    except TypeError as e:
        print(e)
        sys.exit("\n\n".join((__doc__, "Bad arguments for command '{}': '{}'".format(cmd, ' '.join(args)))))


if __name__ == "__main__":
    main()
