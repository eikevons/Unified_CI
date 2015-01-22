"""\
Manually add directory containing unified_ci to sys.path if necessary.

It is expected, that the test scripts are called from within the `tests/`
directory or the root directory of the repository (namely `tests/..`).

If this is not working, an `ImportError` is raised.
"""
try:
    import unified_ci
except ImportError as e:
    import os
    import sys

    if os.path.isdir("unified_ci"):
        newpath = os.getcwd()
    elif os.path.isdir("../unified_ci"):
        newpath = os.path.abspath("..")
    else:
        raise e
    sys.path.insert(0, newpath)
    import unified_ci
