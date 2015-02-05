"""\
Unified CI: Tools to calculate unified confidence intervals
===========================================================

The modules in this package contain tools to calculate unified
confidence intervals in different situations:
For each  a single module exists.

:mod:`simple_gaussian`
  Expectation value of a Gaussian distributed variable with *known*
  variance constrained to the positive domain. 
:mod:`simple_poisson`
  Expectation value of Poissonian with *known* background.
:mod:`hybrid_poisson`
  Expectation value of Poissonian with *unknown* background.

The modules offer a more-or-less uniform interface to calculate the lower
and upper bounds of uniform confidence intervals as well as to calculate the
necessary intermediate results (e.g. likelihood ratios, critical values):

* :func:`lower_limit(...)`
* :func:`upper_limit(...)`
* :func:`confidence_interval(...)`
* :func:`critical_value(...)`
* :func:`likelihood_ratio(...)`

The `simple_*` cases above are the ones discussed in the original paper
"Unified approach to the classical statistical analysis of small signals" by
G.Feldman and R.Cousins `arXiv:physics/9611021
<http://arxiv.org/abs/physics/9711021>`_. For a introduction to confidence
intervals see e.g. `Wikipedia
<https://en.wikipedia.org/wiki/Confidence_interval>`_.

"""
