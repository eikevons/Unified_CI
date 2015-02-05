Unified CI: Tools to calculate unified confidence intervals
===========================================================

Introduction
------------

The modules in this package contain tools to calculate unified
confidence intervals in different situations: For each situation a
single module exists.

`simple_gaussian`
    Expectation value of a Gaussian distributed variable with *known*
    variance constrained to the positive domain. 

`simple_poisson`
    Expectation value of Poissonian with *known* background.

`hybrid_poisson`
    Expectation value of Poissonian with *unknown* background.

The modules offer a more-or-less uniform interface to calculate the lower
and upper bounds of uniform confidence intervals as well as to calculate the
necessary intermediate results (e.g. likelihood ratios, critical values):

* `lower_limit(...)`
* `upper_limit(...)`
* `confidence_interval(...)`
* `critical_value(...)`
* `likelihood_ratio(...)`

The `simple_*` cases above are the ones discussed in the original paper
"Unified approach to the classical statistical analysis of small
signals" by G.Feldman and R.Cousins `arXiv:physics/9611021
<http://arxiv.org/abs/physics/9711021>`_. For a introduction to
confidence intervals see e.g. `Wikipedia
<https://en.wikipedia.org/wiki/Confidence_interval>`_.

Installation
------------

The code is running with Python 2 and 3. Except for the functions
defined in `<unified_ci/tools.py>`_ and general dependances on `NumPy`_ and
`SciPy`_, each module is self sufficient and can be used by copying it to
your respective source tree.

If you want to use the complete package it can be installed by:

1. Get the source code by cloning the repositiory::

       git clone https://github.com/eikevons/Unified_CI.git

2. Install by calling `setup.py` inside `Unified_CI`::
       
       python setup.py install

3. Test the installation::

       python -c "import unified_ci; print(unified_ci)"

.. _NumPy: http://numpy.scipy.org
.. _SciPy: http://scipy.org/scipylib/index.html
