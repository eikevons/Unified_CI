"""\
Estimate the coverage of unified confidence intervals.

Usage: coverages_simple.py COMMAND ARGS

Possible COMMANDs:
    poisson     to calculate a coverage grid for simple_poisson
    gauss       to calculate a coverage grid for simple_gaussian
    plot        to plot a coverage grid file

Expected ARGS:
    - poisson: THETAs Bs CLs NTEST
    - gauss: MUs SIGMAs CLs NTEST
    - plot: FILEPATH

The PARAMs are comma separated lists of the respective parameter values or
colon separated triples of `np.linspace` arguments:
    1.0,1.1,1.3,1.4 to test the specified values
or
    1.0:5.0:10 to test a grid of 10 points between 1.0 and 5.0
"""
from __future__ import division, print_function
import sys
from itertools import product
import multiprocessing
import numpy as np

import context
from unified_ci.simple_poisson import confidence_interval as simpoi_ci
from unified_ci.simple_gaussian import confidence_interval as simgau_ci


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


def _mp_target_simple_gauss(args):
    mu, sigma, cl, n_test = args
    n_succ = 0
    for i in range(n_test):
        x = np.random.randn() * sigma + mu
        ll, ul = simgau_ci(x, sigma, cl)
        if ll <= mu <= ul:
            n_succ += 1
    return mu, sigma, cl, n_succ


def gauss(mus, sigmas, cls, ntest):
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


def _mp_target_simple_poisson(args):
    theta, b, cl, n_test = args
    n_succ = 0
    for i in range(n_test):
        n = np.random.poisson(theta + b)
        ll, ul = simpoi_ci(n, b, cl)
        if ll <= theta <= ul:
            n_succ += 1
    return theta, b, cl, n_succ


def poisson(thetas, bs, cls, ntest):
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

def load_coverage_file(path):
    params = {}
    with open(path) as fd:
        # parse header
        for line in fd:
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

        vals = np.loadtxt(fd)
    return params, colnames, vals

def calc_cl_uncertainty(target_cl, N_mc):
    """Calculate uncertainty of CL estimated from binomial distribution."""
    print(target_cl, N_mc)
    return np.sqrt(target_cl * (1 - target_cl) / N_mc)


def plot(file):
    params, colnames, vals = load_coverage_file(file)
    from matplotlib import pyplot as plt

    target_cls = set()

    plt.figure()
    plt.title("coverage grid {}".format(file))
    # all unique parameter value combinations (the last 2 columns are CL N_success)
    upars = list(product(*[np.unique(vals[:,i]) for i in range(vals.shape[1]-2)]))
    for i, up in enumerate(upars):
        # build index array for this unique parameter combination
        idx = np.ones_like(vals[:,0], dtype=np.bool)
        for j, p in enumerate(up):
            idx *= (vals[:,j] == p)

        target_cls.update(vals[idx,-2])
        cl = vals[idx,-1].astype(np.float) / params["ntest"]
        plt.plot([i] * len(cl), cl, "xk")


    for target_cl in target_cls:
        delta_cl = calc_cl_uncertainty(target_cl, params["ntest"])
        plt.axhline(target_cl, color="g")
        plt.axhspan(target_cl - delta_cl, target_cl + delta_cl, alpha=0.5, color="g")


    plt.xticks(range(len(upars)), [repr(u) for u in upars], rotation=90)
    plt.ylabel("confidence level")
    plt.xlabel("-".join(colnames[:-2]) + " combinations")
    plt.xlim((-0.5, len(upars)-0.5))
    plt.show()








commands = {"poisson" : poisson,
            "gauss" : gauss,
            "plot": plot
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
    except TypeError:
        sys.exit("\n\n".join((__doc__, "Bad arguments for command '{}': '{}'".format(cmd, ' '.join(args)))))


if __name__ == "__main__":
    main()
