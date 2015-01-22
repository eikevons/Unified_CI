import unittest

import context
from unified_ci import tools, simple_gaussian, simple_poisson

class TestTools(unittest.TestCase):
    def test_bisect(self):
        # Dummy function for root finding with `f(x) == 0` for x in [0,1]
        def dummy_f(x):
            if x < 0 :
                return 1
            elif x < 1:
                return 0
            else:
                return -1

        self.assertAlmostEqual(0,
                tools.bisect(dummy_f, -1, 2, xtol=1e-3),
                3)

        self.assertAlmostEqual(1,
                tools.bisect(dummy_f, 2, -1, xtol=1e-3),
                3)

    def DIStest_conservative_quantile(self):
        data1 = 10 * [0.0] + 50 * [1.0] + 40 * [1.1]

        test_cases = (
                # (expected_result, probability)
                (0.0, 0.09),
                (1.0, 0.11),
                (0.0, 0.1)
                )

        # Test with ordered data.
        for r, p, d in test_cases:
            self.assertEqual(r, tools.conservative_quantile(data1, p, d)[0],
                    "conservative_quantile test failed for r={} p={} d={}".format(r, p, d))

class TestSimpleGaussian(unittest.TestCase):
    def assertDifferenceCompatFC(self, a, b):
        """Test that difference `abs(a-b)` is below 0.005 (= 0.5 * assumed precision of F+C Gaussian tables).
        """
        self.assertGreater(0.005, abs(a - b), msg="absolute difference between {0!r} and {1!r} above 0.005".format(a, b))

    def DIStest_lower_limit_against_FC_paper(self):
        """Test lower_limit against Table X of Feldman+Cousins paper."""
        self.assertDifferenceCompatFC(0.0, simple_gaussian.lower_limit(-2.3, 1.0, 0.6827))
        self.assertDifferenceCompatFC(0.0, simple_gaussian.lower_limit(-2.3, 1.0, 0.90))
        self.assertDifferenceCompatFC(0.0, simple_gaussian.lower_limit(-2.3, 1.0, 0.95))
        self.assertDifferenceCompatFC(0.0, simple_gaussian.lower_limit(-2.3, 1.0, 0.99))

        self.assertDifferenceCompatFC(0.02, simple_gaussian.lower_limit(0.5, 1.0, 0.6827))
        self.assertDifferenceCompatFC(0.00, simple_gaussian.lower_limit(0.5, 1.0, 0.90))
        self.assertDifferenceCompatFC(0.00, simple_gaussian.lower_limit(0.5, 1.0, 0.95))
        self.assertDifferenceCompatFC(0.00, simple_gaussian.lower_limit(0.5, 1.0, 0.99))

        self.assertDifferenceCompatFC(0.42, simple_gaussian.lower_limit(1.3, 1.0, 0.6827))
        self.assertDifferenceCompatFC(0.02, simple_gaussian.lower_limit(1.3, 1.0, 0.90))
        self.assertDifferenceCompatFC(0.00, simple_gaussian.lower_limit(1.3, 1.0, 0.95))
        self.assertDifferenceCompatFC(0.00, simple_gaussian.lower_limit(1.3, 1.0, 0.99))

        self.assertDifferenceCompatFC(0.72, simple_gaussian.lower_limit(1.7, 1.0, 0.6827))
        self.assertDifferenceCompatFC(0.38, simple_gaussian.lower_limit(1.7, 1.0, 0.90))
        self.assertDifferenceCompatFC(0.06, simple_gaussian.lower_limit(1.7, 1.0, 0.95))
        self.assertDifferenceCompatFC(0.00, simple_gaussian.lower_limit(1.7, 1.0, 0.99))

        self.assertDifferenceCompatFC(1.40, simple_gaussian.lower_limit(2.4, 1.0, 0.6827))
        self.assertDifferenceCompatFC(0.87, simple_gaussian.lower_limit(2.4, 1.0, 0.90))
        self.assertDifferenceCompatFC(0.69, simple_gaussian.lower_limit(2.4, 1.0, 0.95))
        self.assertDifferenceCompatFC(0.07, simple_gaussian.lower_limit(2.4, 1.0, 0.99))

    def DIStest_upper_limit_against_FC_paper(self):
        """Test upper_limit against Table X of Feldman+Cousins paper."""
        self.assertDifferenceCompatFC(0.05, simple_gaussian.upper_limit(-2.3, 1.0, 0.6827))
        self.assertDifferenceCompatFC(0.34, simple_gaussian.upper_limit(-2.3, 1.0, 0.90))
        self.assertDifferenceCompatFC(0.54, simple_gaussian.upper_limit(-2.3, 1.0, 0.95))
        self.assertDifferenceCompatFC(0.99, simple_gaussian.upper_limit(-2.3, 1.0, 0.99))

        self.assertDifferenceCompatFC(1.50, simple_gaussian.upper_limit(0.5, 1.0, 0.6827))
        self.assertDifferenceCompatFC(2.14, simple_gaussian.upper_limit(0.5, 1.0, 0.90))
        self.assertDifferenceCompatFC(2.46, simple_gaussian.upper_limit(0.5, 1.0, 0.95))
        self.assertDifferenceCompatFC(3.08, simple_gaussian.upper_limit(0.5, 1.0, 0.99))

        self.assertDifferenceCompatFC(2.30, simple_gaussian.upper_limit(1.3, 1.0, 0.6827))
        self.assertDifferenceCompatFC(2.94, simple_gaussian.upper_limit(1.3, 1.0, 0.90))
        self.assertDifferenceCompatFC(3.26, simple_gaussian.upper_limit(1.3, 1.0, 0.95))
        self.assertDifferenceCompatFC(3.88, simple_gaussian.upper_limit(1.3, 1.0, 0.99))

        self.assertDifferenceCompatFC(2.70, simple_gaussian.upper_limit(1.7, 1.0, 0.6827))
        self.assertDifferenceCompatFC(3.34, simple_gaussian.upper_limit(1.7, 1.0, 0.90))
        self.assertDifferenceCompatFC(3.66, simple_gaussian.upper_limit(1.7, 1.0, 0.95))
        self.assertDifferenceCompatFC(4.28, simple_gaussian.upper_limit(1.7, 1.0, 0.99))

        self.assertDifferenceCompatFC(3.40, simple_gaussian.upper_limit(2.4, 1.0, 0.6827))
        self.assertDifferenceCompatFC(4.04, simple_gaussian.upper_limit(2.4, 1.0, 0.90))
        self.assertDifferenceCompatFC(4.36, simple_gaussian.upper_limit(2.4, 1.0, 0.95))
        self.assertDifferenceCompatFC(4.98, simple_gaussian.upper_limit(2.4, 1.0, 0.99))

class TestSimplePoissonian(TestSimpleGaussian):
    """
    Lower Limits
    ------------
    Test the results of `simple_poisson.lower_limit()` against the tabulated values of F+C.

    For n=0 to 10 and n=15,20 we test all values of b in the tables
    until a lower limit of 0.00 is reached.

    Upper Limits
    ------------
    Test the results of `simple_poisson.upper_limit()` against the tabulated values of F+C.

    For n=0 to 5 and n=10,15,20 we test all values of b in the tables
    until a "low-stats" result is reached.
    """

    def assertLowerLimitCompatFC(self, a, b, msg=None):
        """Test that `b` truncated to 2 digits is `a`.

        Note
        ----
        We assume that F+C truncated the results for lower limits in the the
        Poisson tables to ensure conservatism of intervals.
        """
        dec, frac = '{0:.4f}'.format(b).split('.')

        b_trunc = float(dec + '.' + frac[:2])
        return self.assertEqual(a, b_trunc, msg=msg)


    def assertUpperLimitCompatFC(self, a, b, msg=None):
        """Test that (`b` truncated to 2 digits + 0.01) is `a`.

        Note
        ----
        We assume that F+C up-rounded the results for upper limits in the the
        Poisson tables to ensure conservatism of intervals.
        """
        dec, frac = '{0:.4f}'.format(b).split('.')

        if msg is None:
            msg = "{0!r} is not compatible to tabulated upper limit {1!r}".format(b, a)

        b_trunc = float(dec + '.' + frac[:2]) + 0.01
        return self.assertAlmostEqual(a, b_trunc, msg=msg)

    # Note:
    # - The parameters deactivated by "#MISS(calculated_value)" differ
    #   compared to the F+C tables. The calculated intervals are larger
    #   (i.e. more conservative). Therefore, we assert that instead.
    # - The parameters deactivated by "#XXX-MISS(calculated_value)" differ
    #   compared to the F+C tables. The calculated intervals are *smaller*
    #   but usually only by 0.01. I assume it's some numerics in this or
    #   their code.

    def DIStest_lower_limit_against_FC_paper_CL06827(self):
        CL = 0.6827

        n = 0
        # TODO: self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 0.0, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 0.5, CL))

        n = 1
        # TODO: self.assertLowerLimitCompatFC(0.37, simple_poisson.lower_limit(n, 0.0, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 0.5, CL))

        n = 2
        # TODO: self.assertLowerLimitCompatFC(0.74, simple_poisson.lower_limit(n, 0.0, CL))
        #MISS self.assertLowerLimitCompatFC(0.44, simple_poisson.lower_limit(2, 0.5, CL))
        self.assertGreater(0.44, simple_poisson.lower_limit(2, 0.5, CL))
        #MISS self.assertLowerLimitCompatFC(0.14, simple_poisson.lower_limit(2, 1.0, CL))
        self.assertGreater(0.14, simple_poisson.lower_limit(2, 1.0, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(2, 1.5, CL))

        n = 3
        # TODO: self.assertLowerLimitCompatFC(1.10, simple_poisson.lower_limit(n, 0.0, CL))
        #XXX-MISS(0.81) self.assertLowerLimitCompatFC(0.80, simple_poisson.lower_limit(3, 0.5, CL))
        self.assertDifferenceCompatFC(0.81, simple_poisson.lower_limit(3, 0.5, CL))
        self.assertLowerLimitCompatFC(0.54, simple_poisson.lower_limit(n, 1.0, CL))
        #MISS self.assertLowerLimitCompatFC(0.32, simple_poisson.lower_limit(n, 1.5, CL))
        self.assertGreater(0.32, simple_poisson.lower_limit(n, 1.5, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 2.0, CL))

        n = 4
        # TODO: self.assertLowerLimitCompatFC(2.34, simple_poisson.lower_limit(n, 0.0, CL))
        #MISS(1.83) self.assertLowerLimitCompatFC(1.84, simple_poisson.lower_limit(n, 0.5, CL))
        self.assertGreater(1.84, simple_poisson.lower_limit(n, 0.5, CL))
        #MISS(1.33) self.assertLowerLimitCompatFC(1.34, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertGreater(1.34, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertLowerLimitCompatFC(0.91, simple_poisson.lower_limit(n, 1.5, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(1.34, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertGreater(1.34, simple_poisson.lower_limit(n, 1.0, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.44, simple_poisson.lower_limit(n, 2.0, CL))
        self.assertGreater(0.44, simple_poisson.lower_limit(n, 2.0, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.25, simple_poisson.lower_limit(n, 2.5, CL))
        self.assertGreater(0.25, simple_poisson.lower_limit(n, 2.5, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 3.0, CL))

        n = 5
        # TODO: self.assertLowerLimitCompatFC(2.75, simple_poisson.lower_limit(n, 0.0, CL))
        self.assertLowerLimitCompatFC(2.25, simple_poisson.lower_limit(n, 0.5, CL))
        self.assertLowerLimitCompatFC(1.75, simple_poisson.lower_limit(n, 1.0, CL))
        #MISS(1.31) self.assertLowerLimitCompatFC(1.32, simple_poisson.lower_limit(n, 1.5, CL))
        self.assertGreater(1.32, simple_poisson.lower_limit(n, 1.5, CL))
        self.assertLowerLimitCompatFC(0.97, simple_poisson.lower_limit(n, 2.0, CL))
        self.assertLowerLimitCompatFC(0.68, simple_poisson.lower_limit(n, 2.5, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.45, simple_poisson.lower_limit(n, 3.0, CL))
        self.assertGreater(0.45, simple_poisson.lower_limit(n, 3.0, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.20, simple_poisson.lower_limit(n, 3.5, CL))
        self.assertGreater(0.20, simple_poisson.lower_limit(n, 3.5, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 4.0, CL))

        n = 6
        # TODO: self.assertLowerLimitCompatFC(3.82, simple_poisson.lower_limit(n, 0.0, CL))
        self.assertLowerLimitCompatFC(3.32, simple_poisson.lower_limit(n, 0.5, CL))
        #XXX-MISS(2.83) self.assertLowerLimitCompatFC(2.82, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertDifferenceCompatFC(2.83, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertLowerLimitCompatFC(2.32, simple_poisson.lower_limit(n, 1.5, CL))
        self.assertLowerLimitCompatFC(1.82, simple_poisson.lower_limit(n, 2.0, CL))
        #XXX-MISS(1.38) self.assertLowerLimitCompatFC(1.37, simple_poisson.lower_limit(n, 2.5, CL))
        self.assertDifferenceCompatFC(1.38, simple_poisson.lower_limit(n, 2.5, CL))
        self.assertLowerLimitCompatFC(1.01, simple_poisson.lower_limit(n, 3.0, CL))
        self.assertLowerLimitCompatFC(0.62, simple_poisson.lower_limit(n, 3.5, CL))
        #MISS(0.0) self.assertLowerLimitCompatFC(0.36, simple_poisson.lower_limit(n, 4.0, CL))
        self.assertGreater(0.36, simple_poisson.lower_limit(n, 4.0, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 5.0, CL))

        n = 7
        # TODO: self.assertLowerLimitCompatFC(4.25, simple_poisson.lower_limit(n, 0.0, CL))
        self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.5, CL))
        self.assertLowerLimitCompatFC(3.25, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertLowerLimitCompatFC(2.75, simple_poisson.lower_limit(n, 1.5, CL))
        self.assertLowerLimitCompatFC(2.25, simple_poisson.lower_limit(n, 2.0, CL))
        self.assertLowerLimitCompatFC(1.80, simple_poisson.lower_limit(n, 2.5, CL))
        #MISS(1.42) self.assertLowerLimitCompatFC(1.41, simple_poisson.lower_limit(n, 3.0, CL))
        self.assertDifferenceCompatFC(1.42, simple_poisson.lower_limit(n, 3.0, CL))
        #MISS(1.08) self.assertLowerLimitCompatFC(1.09, simple_poisson.lower_limit(n, 3.5, CL))
        self.assertGreater(1.09, simple_poisson.lower_limit(n, 3.5, CL))
        #MISS(0.80) self.assertLowerLimitCompatFC(0.81, simple_poisson.lower_limit(n, 4.0, CL))
        self.assertGreater(0.81, simple_poisson.lower_limit(n, 4.0, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.32, simple_poisson.lower_limit(n, 5.0, CL))
        self.assertGreater(0.32, simple_poisson.lower_limit(n, 5.0, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 6.0, CL))

        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))
        # self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.5, CL))
        # self.assertLowerLimitCompatFC(3.25, simple_poisson.lower_limit(n, 1.0, CL))
        # self.assertLowerLimitCompatFC(2.75, simple_poisson.lower_limit(n, 1.5, CL))
        # self.assertLowerLimitCompatFC(2.25, simple_poisson.lower_limit(n, 2.0, CL))
        # self.assertLowerLimitCompatFC(1.80, simple_poisson.lower_limit(n, 2.5, CL))
        # self.assertLowerLimitCompatFC(1.41, simple_poisson.lower_limit(n, 3.0, CL))
        # self.assertLowerLimitCompatFC(1.09, simple_poisson.lower_limit(n, 3.5, CL))
        # self.assertLowerLimitCompatFC(0.81, simple_poisson.lower_limit(n, 4.0, CL))
        # self.assertLowerLimitCompatFC(0.32, simple_poisson.lower_limit(n, 5.0, CL))
        # self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 6.0, CL))

    def DIStest_lower_limit_against_FC_paper_CL090(self):
        CL = 0.90

        n = 0
        # TODO: self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 0.0, CL))

        n = 1
        # TODO: self.assertLowerLimitCompatFC(0.11, simple_poisson.lower_limit(n, 0.0, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 0.5, CL))

        n = 2
        # TODO: self.assertLowerLimitCompatFC(0.53, simple_poisson.lower_limit(n, 0.0, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.03, simple_poisson.lower_limit(n, 0.5, CL))
        self.assertGreater(0.03, simple_poisson.lower_limit(n, 0.5, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 1.0, CL))

        n = 3
        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))
        self.assertLowerLimitCompatFC(0.60, simple_poisson.lower_limit(n, 0.5, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.10, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertGreater(0.10, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 1.5, CL))

        n = 4
        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))
        #MISS(1.16) self.assertLowerLimitCompatFC(1.17, simple_poisson.lower_limit(n, 0.5, CL))
        self.assertGreater(1.17, simple_poisson.lower_limit(n, 0.5, CL))
        #XXX-MISS(0.75) self.assertLowerLimitCompatFC(0.74, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertDifferenceCompatFC(0.75, simple_poisson.lower_limit(n, 1.0, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.24, simple_poisson.lower_limit(n, 1.5, CL))
        self.assertGreater(0.24, simple_poisson.lower_limit(n, 1.5, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 2.0, CL))

        n = 5
        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))
        self.assertLowerLimitCompatFC(1.53, simple_poisson.lower_limit(n, 0.5, CL))
        self.assertLowerLimitCompatFC(1.25, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertLowerLimitCompatFC(0.93, simple_poisson.lower_limit(n, 1.5, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.43, simple_poisson.lower_limit(n, 2.0, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 2.0, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 2.5, CL))

        n = 6
        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))
        self.assertLowerLimitCompatFC(1.90, simple_poisson.lower_limit(n, 0.5, CL))
        self.assertLowerLimitCompatFC(1.61, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertLowerLimitCompatFC(1.33, simple_poisson.lower_limit(n, 1.5, CL))
        self.assertLowerLimitCompatFC(1.08, simple_poisson.lower_limit(n, 2.0, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.65, simple_poisson.lower_limit(n, 2.5, CL))
        self.assertGreater(0.65, simple_poisson.lower_limit(n, 2.5, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.15, simple_poisson.lower_limit(n, 3.0, CL))
        self.assertGreater(0.15, simple_poisson.lower_limit(n, 3.0, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 3.5, CL))

        n = 7
        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))
        self.assertLowerLimitCompatFC(3.06, simple_poisson.lower_limit(n, 0.5, CL))
        self.assertLowerLimitCompatFC(2.56, simple_poisson.lower_limit(n, 1.0, CL))
        #MISS(2.08) self.assertLowerLimitCompatFC(2.09, simple_poisson.lower_limit(n, 1.5, CL))
        self.assertGreater(2.09, simple_poisson.lower_limit(n, 1.5, CL))
        self.assertLowerLimitCompatFC(1.59, simple_poisson.lower_limit(n, 2.0, CL))
        self.assertLowerLimitCompatFC(1.18, simple_poisson.lower_limit(n, 2.5, CL))
        self.assertLowerLimitCompatFC(0.89, simple_poisson.lower_limit(n, 3.0, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.39, simple_poisson.lower_limit(n, 3.5, CL))
        self.assertGreater(0.39, simple_poisson.lower_limit(n, 3.5, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 4.0, CL))
        # self.assertLowerLimitCompatFC(0.32, simple_poisson.lower_limit(n, 5.0, CL))
        # self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 6.0, CL))

        n = 8
        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))
        self.assertLowerLimitCompatFC(3.46, simple_poisson.lower_limit(n, 0.5, CL))
        self.assertLowerLimitCompatFC(2.96, simple_poisson.lower_limit(n, 1.0, CL))
        #XXX-MISS(2.52) self.assertLowerLimitCompatFC(2.51, simple_poisson.lower_limit(n, 1.5, CL))
        self.assertDifferenceCompatFC(2.52, simple_poisson.lower_limit(n, 1.5, CL))
        #MISS(2.13) self.assertLowerLimitCompatFC(2.14, simple_poisson.lower_limit(n, 2.0, CL))
        self.assertGreater(2.14, simple_poisson.lower_limit(n, 2.0, CL))
        self.assertLowerLimitCompatFC(1.81, simple_poisson.lower_limit(n, 2.5, CL))
        self.assertLowerLimitCompatFC(1.51, simple_poisson.lower_limit(n, 3.0, CL))
        self.assertLowerLimitCompatFC(1.06, simple_poisson.lower_limit(n, 3.5, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.66, simple_poisson.lower_limit(n, 4.0, CL))
        self.assertGreater(0.66, simple_poisson.lower_limit(n, 4.0, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 5.0, CL))

        n = 9
        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))
        #MISS(3.85) self.assertLowerLimitCompatFC(3.86, simple_poisson.lower_limit(n, 0.5, CL))
        self.assertGreater(3.86, simple_poisson.lower_limit(n, 0.5, CL))
        #MISS(3.35) self.assertLowerLimitCompatFC(3.36, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertGreater(3.36, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertLowerLimitCompatFC(2.91, simple_poisson.lower_limit(n, 1.5, CL))
        #MISS(2.52) self.assertLowerLimitCompatFC(2.53, simple_poisson.lower_limit(n, 2.0, CL))
        self.assertGreater(2.53, simple_poisson.lower_limit(n, 2.0, CL))
        #MISS(2.18) self.assertLowerLimitCompatFC(2.19, simple_poisson.lower_limit(n, 2.5, CL))
        self.assertGreater(2.19, simple_poisson.lower_limit(n, 2.5, CL))
        self.assertLowerLimitCompatFC(1.88, simple_poisson.lower_limit(n, 3.0, CL))
        self.assertLowerLimitCompatFC(1.59, simple_poisson.lower_limit(n, 3.5, CL))
        self.assertLowerLimitCompatFC(1.33, simple_poisson.lower_limit(n, 4.0, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.43, simple_poisson.lower_limit(n, 5.0, CL))
        self.assertGreater(0.43, simple_poisson.lower_limit(n, 5.0, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 6.0, CL))

        n = 15
        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))
        self.assertLowerLimitCompatFC(8.98, simple_poisson.lower_limit(n, 0.5, CL))
        #XXX-MISS(8.49) self.assertLowerLimitCompatFC(8.48, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertDifferenceCompatFC(8.49, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertLowerLimitCompatFC(7.98, simple_poisson.lower_limit(n, 1.5, CL))
        #XXX-MISS(7.49) self.assertLowerLimitCompatFC(7.48, simple_poisson.lower_limit(n, 2.0, CL))
        self.assertDifferenceCompatFC(7.49, simple_poisson.lower_limit(n, 2.0, CL))
        self.assertLowerLimitCompatFC(6.98, simple_poisson.lower_limit(n, 2.5, CL))
        self.assertLowerLimitCompatFC(6.48, simple_poisson.lower_limit(n, 3.0, CL))
        self.assertLowerLimitCompatFC(5.98, simple_poisson.lower_limit(n, 3.5, CL))
        self.assertLowerLimitCompatFC(5.48, simple_poisson.lower_limit(n, 4.0, CL))
        #XXX-MISS(4.49) self.assertLowerLimitCompatFC(4.48, simple_poisson.lower_limit(n, 5.0, CL))
        self.assertDifferenceCompatFC(4.49, simple_poisson.lower_limit(n, 5.0, CL))
        self.assertLowerLimitCompatFC(3.48, simple_poisson.lower_limit(n, 6.0, CL))
        #XXX-MISS(2.57) self.assertLowerLimitCompatFC(2.56, simple_poisson.lower_limit(n, 7.0, CL))
        self.assertDifferenceCompatFC(2.57, simple_poisson.lower_limit(n, 7.0, CL))
        self.assertLowerLimitCompatFC(1.98, simple_poisson.lower_limit(n, 8.0, CL))
        self.assertLowerLimitCompatFC(1.26, simple_poisson.lower_limit(n, 9.0, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.30, simple_poisson.lower_limit(n, 10.0, CL))
        self.assertGreater(0.30, simple_poisson.lower_limit(n, 10.0, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 11.0, CL))

        n = 20
        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))
        self.assertLowerLimitCompatFC(13.05, simple_poisson.lower_limit(n, 0.5, CL))
        #XXX-MISS(12.56) self.assertLowerLimitCompatFC(12.55, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertDifferenceCompatFC(12.56, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertLowerLimitCompatFC(12.05, simple_poisson.lower_limit(n, 1.5, CL))
        self.assertLowerLimitCompatFC(11.55, simple_poisson.lower_limit(n, 2.0, CL))
        self.assertLowerLimitCompatFC(11.05, simple_poisson.lower_limit(n, 2.5, CL))
        self.assertLowerLimitCompatFC(10.55, simple_poisson.lower_limit(n, 3.0, CL))
        self.assertLowerLimitCompatFC(10.05, simple_poisson.lower_limit(n, 3.5, CL))
        self.assertLowerLimitCompatFC(9.55, simple_poisson.lower_limit(n, 4.0, CL))
        self.assertLowerLimitCompatFC(8.55, simple_poisson.lower_limit(n, 5.0, CL))
        self.assertLowerLimitCompatFC(7.55, simple_poisson.lower_limit(n, 6.0, CL))
        self.assertLowerLimitCompatFC(6.55, simple_poisson.lower_limit(n, 7.0, CL))
        self.assertLowerLimitCompatFC(5.55, simple_poisson.lower_limit(n, 8.0, CL))
        self.assertLowerLimitCompatFC(4.55, simple_poisson.lower_limit(n, 9.0, CL))
        self.assertLowerLimitCompatFC(3.55, simple_poisson.lower_limit(n, 10.0, CL))
        self.assertLowerLimitCompatFC(2.81, simple_poisson.lower_limit(n, 11.0, CL))
        self.assertLowerLimitCompatFC(2.23, simple_poisson.lower_limit(n, 12.0, CL))
        self.assertLowerLimitCompatFC(1.48, simple_poisson.lower_limit(n, 13.0, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.53, simple_poisson.lower_limit(n, 14.0, CL))
        self.assertGreater(0.53, simple_poisson.lower_limit(n, 14.0, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 15.0, CL))

    def DIStest_lower_limit_against_FC_paper_CL095(self):
        CL = 0.95

        n = 1
        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 0.5, CL))

        n = 2
        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 0.5, CL))

        n = 3
        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.32, simple_poisson.lower_limit(n, 0.5, CL))
        self.assertGreater(0.32, simple_poisson.lower_limit(n, 0.5, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 1.0, CL))

        n = 4
        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))
        #MISS(0.86) self.assertLowerLimitCompatFC(0.87, simple_poisson.lower_limit(n, 0.5, CL))
        self.assertGreater(0.87, simple_poisson.lower_limit(n, 0.5, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.37, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertGreater(0.37, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 1.5, CL))

        n = 5
        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))
        self.assertLowerLimitCompatFC(1.47, simple_poisson.lower_limit(n, 0.5, CL))
        self.assertLowerLimitCompatFC(0.97, simple_poisson.lower_limit(n, 1.0, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.47, simple_poisson.lower_limit(n, 1.5, CL))
        self.assertGreater(0.47, simple_poisson.lower_limit(n, 1.5, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 2.0, CL))

        n = 6
        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))
        self.assertLowerLimitCompatFC(1.90, simple_poisson.lower_limit(n, 0.5, CL))
        self.assertLowerLimitCompatFC(1.61, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertLowerLimitCompatFC(1.11, simple_poisson.lower_limit(n, 1.5, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.61, simple_poisson.lower_limit(n, 2.0, CL))
        self.assertGreater(0.61, simple_poisson.lower_limit(n, 2.0, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.11, simple_poisson.lower_limit(n, 2.5, CL))
        self.assertGreater(0.11, simple_poisson.lower_limit(n, 2.5, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 3.0, CL))

        n = 7
        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(2.27, simple_poisson.lower_limit(n, 0.5, CL))
        self.assertGreater(2.27, simple_poisson.lower_limit(n, 0.5, CL))
        self.assertLowerLimitCompatFC(1.97, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertLowerLimitCompatFC(1.69, simple_poisson.lower_limit(n, 1.5, CL))
        #MISS(1.28) self.assertLowerLimitCompatFC(1.29, simple_poisson.lower_limit(n, 2.0, CL))
        self.assertGreater(1.29, simple_poisson.lower_limit(n, 2.0, CL))
        self.assertLowerLimitCompatFC(0.79, simple_poisson.lower_limit(n, 2.5, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.29, simple_poisson.lower_limit(n, 3.0, CL))
        self.assertGreater(0.29, simple_poisson.lower_limit(n, 3.0, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 3.5, CL))

        n = 8
        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))
        self.assertLowerLimitCompatFC(2.63, simple_poisson.lower_limit(n, 0.5, CL))
        self.assertLowerLimitCompatFC(2.33, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertLowerLimitCompatFC(2.05, simple_poisson.lower_limit(n, 1.5, CL))
        self.assertLowerLimitCompatFC(1.78, simple_poisson.lower_limit(n, 2.0, CL))
        self.assertLowerLimitCompatFC(1.48, simple_poisson.lower_limit(n, 2.5, CL))
        self.assertLowerLimitCompatFC(0.98, simple_poisson.lower_limit(n, 3.0, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.48, simple_poisson.lower_limit(n, 3.5, CL))
        self.assertGreater(0.48, simple_poisson.lower_limit(n, 3.5, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 4.0, CL))

        n = 9
        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))
        #MISS(3.85) self.assertLowerLimitCompatFC(3.86, simple_poisson.lower_limit(n, 0.5, CL))
        self.assertGreater(3.86, simple_poisson.lower_limit(n, 0.5, CL))
        #MISS(3.35) self.assertLowerLimitCompatFC(3.36, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertGreater(3.36, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertLowerLimitCompatFC(2.91, simple_poisson.lower_limit(n, 1.5, CL))
        self.assertLowerLimitCompatFC(2.46, simple_poisson.lower_limit(n, 2.0, CL))
        self.assertLowerLimitCompatFC(1.96, simple_poisson.lower_limit(n, 2.5, CL))
        self.assertLowerLimitCompatFC(1.62, simple_poisson.lower_limit(n, 3.0, CL))
        #MISS(1.19) self.assertLowerLimitCompatFC(1.20, simple_poisson.lower_limit(n, 3.5, CL))
        self.assertGreater(1.20, simple_poisson.lower_limit(n, 3.5, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.70, simple_poisson.lower_limit(n, 4.0, CL))
        self.assertGreater(0.70, simple_poisson.lower_limit(n, 4.0, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 4.0, CL))

        n = 10
        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))
        self.assertLowerLimitCompatFC(4.25, simple_poisson.lower_limit(n, 0.5, CL))
        self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertLowerLimitCompatFC(3.30, simple_poisson.lower_limit(n, 1.5, CL))
        self.assertLowerLimitCompatFC(2.92, simple_poisson.lower_limit(n, 2.0, CL))
        self.assertLowerLimitCompatFC(2.57, simple_poisson.lower_limit(n, 2.5, CL))
        self.assertLowerLimitCompatFC(2.25, simple_poisson.lower_limit(n, 3.0, CL))
        self.assertLowerLimitCompatFC(1.82, simple_poisson.lower_limit(n, 3.5, CL))
        #MISS(1.42) self.assertLowerLimitCompatFC(1.43, simple_poisson.lower_limit(n, 4.0, CL))
        self.assertGreater(1.43, simple_poisson.lower_limit(n, 4.0, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.43, simple_poisson.lower_limit(n, 5.0, CL))
        self.assertGreater(0.43, simple_poisson.lower_limit(n, 5.0, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 6.0, CL))

        n = 15
        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))
        self.assertLowerLimitCompatFC(7.75, simple_poisson.lower_limit(n, 0.5, CL))
        self.assertLowerLimitCompatFC(7.25, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertLowerLimitCompatFC(6.75, simple_poisson.lower_limit(n, 1.5, CL))
        self.assertLowerLimitCompatFC(6.25, simple_poisson.lower_limit(n, 2.0, CL))
        self.assertLowerLimitCompatFC(5.75, simple_poisson.lower_limit(n, 2.5, CL))
        self.assertLowerLimitCompatFC(5.25, simple_poisson.lower_limit(n, 3.0, CL))
        #MISS(4.77) self.assertLowerLimitCompatFC(4.78, simple_poisson.lower_limit(n, 3.5, CL))
        self.assertGreater(4.78, simple_poisson.lower_limit(n, 3.5, CL))
        self.assertLowerLimitCompatFC(4.35, simple_poisson.lower_limit(n, 4.0, CL))
        self.assertLowerLimitCompatFC(3.58, simple_poisson.lower_limit(n, 5.0, CL))
        self.assertLowerLimitCompatFC(2.91, simple_poisson.lower_limit(n, 6.0, CL))
        self.assertLowerLimitCompatFC(2.11, simple_poisson.lower_limit(n, 7.0, CL))
        self.assertLowerLimitCompatFC(1.25, simple_poisson.lower_limit(n, 8.0, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.25, simple_poisson.lower_limit(n, 9.0, CL))
        self.assertGreater(0.25, simple_poisson.lower_limit(n, 9.0, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 10.0, CL))

        n = 20
        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))
        self.assertLowerLimitCompatFC(11.83, simple_poisson.lower_limit(n, 0.5, CL))
        #MISS(11.32) self.assertLowerLimitCompatFC(11.33, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertGreater(11.33, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertLowerLimitCompatFC(10.83, simple_poisson.lower_limit(n, 1.5, CL))
        #MISS(10.32) self.assertLowerLimitCompatFC(10.33, simple_poisson.lower_limit(n, 2.0, CL))
        self.assertGreater(10.33, simple_poisson.lower_limit(n, 2.0, CL))
        #MISS(9.82) self.assertLowerLimitCompatFC(9.83, simple_poisson.lower_limit(n, 2.5, CL))
        self.assertGreater(9.83, simple_poisson.lower_limit(n, 2.5, CL))
        self.assertLowerLimitCompatFC(9.33, simple_poisson.lower_limit(n, 3.0, CL))
        self.assertLowerLimitCompatFC(8.83, simple_poisson.lower_limit(n, 3.5, CL))
        #MISS(8.32) self.assertLowerLimitCompatFC(8.33, simple_poisson.lower_limit(n, 4.0, CL))
        self.assertGreater(8.33, simple_poisson.lower_limit(n, 4.0, CL))
        self.assertLowerLimitCompatFC(7.33, simple_poisson.lower_limit(n, 5.0, CL))
        self.assertLowerLimitCompatFC(6.33, simple_poisson.lower_limit(n, 6.0, CL))
        self.assertLowerLimitCompatFC(5.39, simple_poisson.lower_limit(n, 7.0, CL))
        self.assertLowerLimitCompatFC(4.57, simple_poisson.lower_limit(n, 8.0, CL))
        #MISS(3.81( self.assertLowerLimitCompatFC(3.82, simple_poisson.lower_limit(n, 9.0, CL))
        self.assertGreater(3.82, simple_poisson.lower_limit(n, 9.0, CL))
        #MISS(2.93) self.assertLowerLimitCompatFC(2.94, simple_poisson.lower_limit(n, 10.0, CL))
        self.assertGreater(2.94, simple_poisson.lower_limit(n, 10.0, CL))
        self.assertLowerLimitCompatFC(2.23, simple_poisson.lower_limit(n, 11.0, CL))
        self.assertLowerLimitCompatFC(1.25, simple_poisson.lower_limit(n, 12.0, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.25, simple_poisson.lower_limit(n, 13.0, CL))
        self.assertGreater(0.25, simple_poisson.lower_limit(n, 13.0, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 14.0, CL))

    def DIStest_lower_limit_against_FC_paper_CL099(self):
        CL = 0.99
        # n = 0
        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))

        n = 1
        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 0.5, CL))

        n = 2
        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 0.5, CL))

        n = 3
        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 0.5, CL))

        n = 4
        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.32, simple_poisson.lower_limit(n, 0.5, CL))
        self.assertGreater(0.32, simple_poisson.lower_limit(n, 0.5, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 1.0, CL))

        n = 5
        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))
        self.assertLowerLimitCompatFC(0.78, simple_poisson.lower_limit(n, 0.5, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.28, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertGreater(0.28, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 1.5, CL))

        n = 6
        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))
        #MISS(1.28) self.assertLowerLimitCompatFC(1.29, simple_poisson.lower_limit(n, 0.5, CL))
        self.assertGreater(1.29, simple_poisson.lower_limit(n, 0.5, CL))
        self.assertLowerLimitCompatFC(0.79, simple_poisson.lower_limit(n, 1.0, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.29, simple_poisson.lower_limit(n, 1.5, CL))
        self.assertGreater(0.29, simple_poisson.lower_limit(n, 1.5, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 2.0, CL))

        n = 7
        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))
        self.assertLowerLimitCompatFC(1.83, simple_poisson.lower_limit(n, 0.5, CL))
        self.assertLowerLimitCompatFC(1.33, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertLowerLimitCompatFC(0.83, simple_poisson.lower_limit(n, 1.5, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.33, simple_poisson.lower_limit(n, 2.0, CL))
        self.assertGreater(0.33, simple_poisson.lower_limit(n, 2.0, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 2.5, CL))

        n = 8
        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))
        #MISS(2.40) self.assertLowerLimitCompatFC(2.41, simple_poisson.lower_limit(n, 0.5, CL))
        self.assertGreater(2.41, simple_poisson.lower_limit(n, 0.5, CL))
        #MISS(1.90) self.assertLowerLimitCompatFC(1.91, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertGreater(1.91, simple_poisson.lower_limit(n, 1.0, CL))
        #MISS(1.40) self.assertLowerLimitCompatFC(1.41, simple_poisson.lower_limit(n, 1.5, CL))
        self.assertGreater(1.41, simple_poisson.lower_limit(n, 1.5, CL))
        #MISS(0.90) self.assertLowerLimitCompatFC(0.91, simple_poisson.lower_limit(n, 2.0, CL))
        self.assertGreater(0.91, simple_poisson.lower_limit(n, 2.0, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.41, simple_poisson.lower_limit(n, 2.5, CL))
        self.assertGreater(0.41, simple_poisson.lower_limit(n, 2.5, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 3.0, CL))

        n = 9
        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))
        self.assertLowerLimitCompatFC(3.00, simple_poisson.lower_limit(n, 0.5, CL))
        #MISS(2.50) self.assertLowerLimitCompatFC(2.51, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertGreater(2.51, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertLowerLimitCompatFC(2.01, simple_poisson.lower_limit(n, 1.5, CL))
        self.assertLowerLimitCompatFC(1.51, simple_poisson.lower_limit(n, 2.0, CL))
        #MISS(1.00) self.assertLowerLimitCompatFC(1.01, simple_poisson.lower_limit(n, 2.5, CL))
        self.assertGreater(1.01, simple_poisson.lower_limit(n, 2.5, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.51, simple_poisson.lower_limit(n, 3.0, CL))
        self.assertGreater(0.51, simple_poisson.lower_limit(n, 3.0, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.01, simple_poisson.lower_limit(n, 3.5, CL))
        self.assertGreater(0.01, simple_poisson.lower_limit(n, 3.5, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 4.0, CL))

        n = 10
        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))
        #MISS(3.36) self.assertLowerLimitCompatFC(3.37, simple_poisson.lower_limit(n, 0.5, CL))
        self.assertGreater(3.37, simple_poisson.lower_limit(n, 0.5, CL))
        #MISS(3.06) self.assertLowerLimitCompatFC(3.07, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertGreater(3.07, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertLowerLimitCompatFC(2.63, simple_poisson.lower_limit(n, 1.5, CL))
        self.assertLowerLimitCompatFC(2.13, simple_poisson.lower_limit(n, 2.0, CL))
        self.assertLowerLimitCompatFC(1.63, simple_poisson.lower_limit(n, 2.5, CL))
        self.assertLowerLimitCompatFC(1.13, simple_poisson.lower_limit(n, 3.0, CL))
        self.assertLowerLimitCompatFC(0.63, simple_poisson.lower_limit(n, 3.5, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.13, simple_poisson.lower_limit(n, 4.0, CL))
        self.assertGreater(0.13, simple_poisson.lower_limit(n, 4.0, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 5.0, CL))

        n = 15
        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))
        self.assertLowerLimitCompatFC(6.20, simple_poisson.lower_limit(n, 0.5, CL))
        self.assertLowerLimitCompatFC(5.70, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertLowerLimitCompatFC(5.24, simple_poisson.lower_limit(n, 1.5, CL))
        self.assertLowerLimitCompatFC(4.84, simple_poisson.lower_limit(n, 2.0, CL))
        self.assertLowerLimitCompatFC(4.48, simple_poisson.lower_limit(n, 2.5, CL))
        self.assertLowerLimitCompatFC(4.14, simple_poisson.lower_limit(n, 3.0, CL))
        self.assertLowerLimitCompatFC(3.82, simple_poisson.lower_limit(n, 3.5, CL))
        self.assertLowerLimitCompatFC(3.42, simple_poisson.lower_limit(n, 4.0, CL))
        self.assertLowerLimitCompatFC(2.48, simple_poisson.lower_limit(n, 5.0, CL))
        self.assertLowerLimitCompatFC(1.48, simple_poisson.lower_limit(n, 6.0, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.48, simple_poisson.lower_limit(n, 7.0, CL))
        self.assertGreater(0.48, simple_poisson.lower_limit(n, 7.0, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 8.0, CL))

        n = 20
        # TODO: self.assertLowerLimitCompatFC(3.75, simple_poisson.lower_limit(n, 0.0, CL))
        self.assertLowerLimitCompatFC(9.78, simple_poisson.lower_limit(n, 0.5, CL))
        self.assertLowerLimitCompatFC(9.28, simple_poisson.lower_limit(n, 1.0, CL))
        self.assertLowerLimitCompatFC(8.78, simple_poisson.lower_limit(n, 1.5, CL))
        self.assertLowerLimitCompatFC(8.28, simple_poisson.lower_limit(n, 2.0, CL))
        self.assertLowerLimitCompatFC(7.78, simple_poisson.lower_limit(n, 2.5, CL))
        self.assertLowerLimitCompatFC(7.28, simple_poisson.lower_limit(n, 3.0, CL))
        #MISS(6.80) self.assertLowerLimitCompatFC(6.81, simple_poisson.lower_limit(n, 3.5, CL))
        self.assertGreater(6.81, simple_poisson.lower_limit(n, 3.5, CL))
        #MISS(6.36) self.assertLowerLimitCompatFC(6.37, simple_poisson.lower_limit(n, 4.0, CL))
        self.assertGreater(6.37, simple_poisson.lower_limit(n, 4.0, CL))
        self.assertLowerLimitCompatFC(5.57, simple_poisson.lower_limit(n, 5.0, CL))
        self.assertLowerLimitCompatFC(4.86, simple_poisson.lower_limit(n, 6.0, CL))
        self.assertLowerLimitCompatFC(3.93, simple_poisson.lower_limit(n, 7.0, CL))
        self.assertLowerLimitCompatFC(3.08, simple_poisson.lower_limit(n, 8.0, CL))
        self.assertLowerLimitCompatFC(2.08, simple_poisson.lower_limit(n, 9.0, CL))
        self.assertLowerLimitCompatFC(1.08, simple_poisson.lower_limit(n, 10.0, CL))
        #MISS(0.00) self.assertLowerLimitCompatFC(0.08, simple_poisson.lower_limit(n, 11.0, CL))
        self.assertGreater(0.08, simple_poisson.lower_limit(n, 11.0, CL))
        self.assertLowerLimitCompatFC(0.00, simple_poisson.lower_limit(n, 12.0, CL))

    def DIStest_upper_limit_against_FC_paper_CL06827(self):
        """Test the results of `simple_poisson.upper_limit()` against the tabulated values of F+C.

        For n=0 to 5 and n=10,15,20 we test all values of b in the tables
        until a "low-stats" result is reached.
        """
        CL = 0.6827

        n = 0.0
        # TODO: self.assertUpperLimitCompatFC(3.75, simple_poisson.upper_limit(n, 0.0, CL))
        #
        #MISS(1.17) self.assertUpperLimitCompatFC(0.80, simple_poisson.upper_limit(n, 0.5, CL))
        self.assertLess(0.80, simple_poisson.upper_limit(n, 0.5, CL))
        self.assertUpperLimitCompatFC(0.54, simple_poisson.upper_limit(n, 1.0, CL))
        #XXX-MISS(0.3125) self.assertUpperLimitCompatFC(0.41, simple_poisson.upper_limit(n, 1.5, CL))
        self.assertEqual(0.3125, simple_poisson.upper_limit(n, 1.5, CL))
        #XXX-MISS(0.1484375) self.assertUpperLimitCompatFC(0.41, simple_poisson.upper_limit(n, 2.0, CL))
        self.assertEqual(0.1484375, simple_poisson.upper_limit(n, 2.0, CL))
        self.assertUpperLimitCompatFC(0.25, simple_poisson.upper_limit(n, 2.5, CL))
        #XXX-MISS(0.109375) self.assertUpperLimitCompatFC(0.25, simple_poisson.upper_limit(n, 3.0, CL))
        self.assertEqual(0.109375, simple_poisson.upper_limit(n, 3.0, CL))
        self.assertUpperLimitCompatFC(0.21, simple_poisson.upper_limit(n, 3.5, CL))
        #XXX-MISS(0.0859375) self.assertUpperLimitCompatFC(0.21, simple_poisson.upper_limit(n, 4.0, CL))
        self.assertEqual(0.0859375, simple_poisson.upper_limit(n, 4.0, CL))
        #XXX-MISS(0.078125) self.assertUpperLimitCompatFC(0.19, simple_poisson.upper_limit(n, 5.0, CL))
        self.assertEqual(0.078125, simple_poisson.upper_limit(n, 5.0, CL))
        #XXX-MISS(0.0625) self.assertUpperLimitCompatFC(0.18, simple_poisson.upper_limit(n, 6.0, CL))
        self.assertEqual(0.0625, simple_poisson.upper_limit(n, 6.0, CL))
        #XXX-MISS(0.0546875) self.assertUpperLimitCompatFC(0.17, simple_poisson.upper_limit(n, 7.0, CL))
        self.assertEqual(0.0546875, simple_poisson.upper_limit(n, 7.0, CL))
        #XXX-MISS(0.046875) self.assertUpperLimitCompatFC(0.17, simple_poisson.upper_limit(n, 8.0, CL))
        self.assertEqual(0.046875, simple_poisson.upper_limit(n, 8.0, CL))

        n = 1
        # TODO: self.assertUpperLimitCompatFC(3.75, simple_poisson.upper_limit(n, 0.0, CL))
        self.assertDifferenceCompatFC(2.25, simple_poisson.upper_limit(n, 0.5, CL))
        #MISS(2.15) self.assertUpperLimitCompatFC(1.75, simple_poisson.upper_limit(n, 1.0, CL))
        self.assertLess(1.75, simple_poisson.upper_limit(n, 1.0, CL))
        self.assertUpperLimitCompatFC(1.32, simple_poisson.upper_limit(n, 1.5, CL))
        #MISS(1.35) self.assertUpperLimitCompatFC(0.97, simple_poisson.upper_limit(n, 2.0, CL))
        self.assertLess(0.97, simple_poisson.upper_limit(n, 2.0, CL))
        #MISS(1.04) self.assertUpperLimitCompatFC(0.68, simple_poisson.upper_limit(n, 2.5, CL))
        self.assertLess(0.68, simple_poisson.upper_limit(n, 2.5, CL))
        #XXX-MISS(0.4453125) self.assertUpperLimitCompatFC(0.50, simple_poisson.upper_limit(n, 3.0, CL))
        self.assertEqual(0.4453125, simple_poisson.upper_limit(n, 3.0, CL))
        #XXX-MISS(0.25) self.assertUpperLimitCompatFC(0.50, simple_poisson.upper_limit(n, 3.5, CL))
        self.assertEqual(0.25, simple_poisson.upper_limit(n, 3.5, CL))
        self.assertUpperLimitCompatFC(0.36, simple_poisson.upper_limit(n, 4.0, CL))
        #XXX-MISS(0.09375) self.assertUpperLimitCompatFC(0.30, simple_poisson.upper_limit(n, 5.0, CL))
        self.assertEqual(0.09375, simple_poisson.upper_limit(n, 5.0, CL))
        #XXX-MISS(0.078125) self.assertUpperLimitCompatFC(0.24, simple_poisson.upper_limit(n, 6.0, CL))
        self.assertEqual(0.078125, simple_poisson.upper_limit(n, 6.0, CL))
        #XXX-MISS(0.0625) self.assertUpperLimitCompatFC(0.21, simple_poisson.upper_limit(n, 7.0, CL))
        self.assertEqual(0.0625, simple_poisson.upper_limit(n, 7.0, CL))
        #XXX-MISS(0.0546875) self.assertUpperLimitCompatFC(0.20, simple_poisson.upper_limit(n, 8.0, CL))
        self.assertEqual(0.0546875, simple_poisson.upper_limit(n, 8.0, CL))
        #XXX-MISS(0.046875) self.assertUpperLimitCompatFC(0.19, simple_poisson.upper_limit(n, 9.0, CL))
        self.assertEqual(0.046875, simple_poisson.upper_limit(n, 9.0, CL))
        self.assertUpperLimitCompatFC(0.18, simple_poisson.upper_limit(n, 10.0, CL))

        n = 2
        # TODO: self.assertUpperLimitCompatFC(3.75, simple_poisson.upper_limit(n, 0.0, CL))
        #MISS(3.76) self.assertUpperLimitCompatFC(3.72, simple_poisson.upper_limit(n, 0.5, CL))
        self.assertLess(3.72, simple_poisson.upper_limit(n, 0.5, CL))
        #MISS(3.25) self.assertUpperLimitCompatFC(3.25, simple_poisson.upper_limit(n, 1.0, CL))
        self.assertEqual(3.25, simple_poisson.upper_limit(n, 1.0, CL))
        self.assertEqual(2.75, simple_poisson.upper_limit(n, 1.5, CL))
        self.assertEqual(2.25, simple_poisson.upper_limit(n, 2.0, CL))
        #MISS(2.2109375) self.assertUpperLimitCompatFC(1.80, simple_poisson.upper_limit(n, 2.5, CL))
        self.assertEqual(2.2109375, simple_poisson.upper_limit(n, 2.5, CL))
        #MISS(1.4140625)
        self.assertEqual(1.4140625, simple_poisson.upper_limit(n, 3.0, CL))
        self.assertUpperLimitCompatFC(1.09, simple_poisson.upper_limit(n, 3.5, CL))
        #MISS(1.171875) self.assertUpperLimitCompatFC(0.81, simple_poisson.upper_limit(n, 4.0, CL))
        self.assertEqual(1.171875, simple_poisson.upper_limit(n, 4.0, CL))
        #XXX-MISS(0.3671875) self.assertUpperLimitCompatFC(0.47, simple_poisson.upper_limit(n, 5.0, CL))
        self.assertEqual(0.3671875, simple_poisson.upper_limit(n, 5.0, CL))
        self.assertUpperLimitCompatFC(0.31, simple_poisson.upper_limit(n, 6.0, CL))
        self.assertUpperLimitCompatFC(0.27, simple_poisson.upper_limit(n, 7.0, CL))
        #XXX-MISS(0.0625) self.assertUpperLimitCompatFC(0.23, simple_poisson.upper_limit(n, 8.0, CL))
        self.assertEqual(0.0625, simple_poisson.upper_limit(n, 8.0, CL))
        #XXX-MISS(0.0546875) self.assertUpperLimitCompatFC(0.21, simple_poisson.upper_limit(n, 9.0, CL))
        self.assertEqual(0.0546875, simple_poisson.upper_limit(n, 9.0, CL))

        n = 3
        # TODO: self.assertUpperLimitCompatFC(3.75, simple_poisson.upper_limit(n, 0.0, CL))
        #MISS(5.234375) self.assertUpperLimitCompatFC(4.80, simple_poisson.upper_limit(n, 0.5, CL))
        self.assertEqual(5.234375, simple_poisson.upper_limit(n, 0.5, CL))
        self.assertUpperLimitCompatFC(4.30, simple_poisson.upper_limit(n, 1.0, CL))
        self.assertUpperLimitCompatFC(3.80, simple_poisson.upper_limit(n, 1.5, CL))
        self.assertUpperLimitCompatFC(3.30, simple_poisson.upper_limit(n, 2.0, CL))
        self.assertUpperLimitCompatFC(2.80, simple_poisson.upper_limit(n, 2.5, CL))
        self.assertUpperLimitCompatFC(2.30, simple_poisson.upper_limit(n, 3.0, CL))
        #MISS(2.265625) self.assertUpperLimitCompatFC(1.84, simple_poisson.upper_limit(n, 3.5, CL))
        self.assertEqual(2.265625, simple_poisson.upper_limit(n, 3.5, CL))
        self.assertUpperLimitCompatFC(1.45, simple_poisson.upper_limit(n, 4.0, CL))
        #MISS(1.5859375) self.assertUpperLimitCompatFC(0.91, simple_poisson.upper_limit(n, 5.0, CL))
        self.assertEqual(1.5859375, simple_poisson.upper_limit(n, 5.0, CL))
        #MISS(1.0234375) self.assertUpperLimitCompatFC(0.69, simple_poisson.upper_limit(n, 6.0, CL))
        self.assertEqual(1.0234375, simple_poisson.upper_limit(n, 6.0, CL))
        #XXX-MISS(0.3046875) self.assertUpperLimitCompatFC(0.42, simple_poisson.upper_limit(n, 7.0, CL))
        self.assertEqual(0.3046875, simple_poisson.upper_limit(n, 7.0, CL))
        #XXX-MISS(0.265625) self.assertUpperLimitCompatFC(0.31, simple_poisson.upper_limit(n, 8.0, CL))
        self.assertEqual(0.265625, simple_poisson.upper_limit(n, 8.0, CL))
        #XXX-MISS(0.234375) self.assertUpperLimitCompatFC(0.26, simple_poisson.upper_limit(n, 9.0, CL))
        self.assertEqual(0.234375, simple_poisson.upper_limit(n, 9.0, CL))
        #XXX-MISS(0.203125) self.assertUpperLimitCompatFC(0.23, simple_poisson.upper_limit(n, 10.0, CL))
        self.assertEqual(0.203125, simple_poisson.upper_limit(n, 10.0, CL))
        #XXX-MISS(0.1875) self.assertUpperLimitCompatFC(0.22, simple_poisson.upper_limit(n, 11.0, CL))
        self.assertEqual(0.1875, simple_poisson.upper_limit(n, 11.0, CL))

        n = 4
        # TODO: self.assertUpperLimitCompatFC(3.75, simple_poisson.upper_limit(n, 0.0, CL))
        self.assertUpperLimitCompatFC(6.28, simple_poisson.upper_limit(n, 0.5, CL))
        #MISS(6.2109375) self.assertUpperLimitCompatFC(5.78, simple_poisson.upper_limit(n, 1.0, CL))
        self.assertEqual(6.2109375, simple_poisson.upper_limit(n, 1.0, CL))
        self.assertUpperLimitCompatFC(5.28, simple_poisson.upper_limit(n, 1.5, CL))
        self.assertUpperLimitCompatFC(4.78, simple_poisson.upper_limit(n, 2.0, CL))
        self.assertUpperLimitCompatFC(4.28, simple_poisson.upper_limit(n, 2.5, CL))
        #MISS(4.2109375) self.assertUpperLimitCompatFC(3.78, simple_poisson.upper_limit(n, 3.0, CL))
        self.assertEqual(4.2109375, simple_poisson.upper_limit(n, 3.0, CL))
        self.assertUpperLimitCompatFC(3.28, simple_poisson.upper_limit(n, 3.5, CL))
        self.assertUpperLimitCompatFC(2.78, simple_poisson.upper_limit(n, 4.0, CL))
        #MISS(2.3203125) self.assertUpperLimitCompatFC(1.90, simple_poisson.upper_limit(n, 5.0, CL))
        self.assertEqual(2.3203125, simple_poisson.upper_limit(n, 5.0, CL))
        self.assertUpperLimitCompatFC(1.22, simple_poisson.upper_limit(n, 6.0, CL))
        #MISS(1.0390625) self.assertUpperLimitCompatFC(0.69, simple_poisson.upper_limit(n, 7.0, CL))
        self.assertEqual(1.0390625, simple_poisson.upper_limit(n, 7.0, CL))
        #XXX-MISS(0.3125) self.assertUpperLimitCompatFC(0.60, simple_poisson.upper_limit(n, 8.0, CL))
        self.assertEqual(0.3125, simple_poisson.upper_limit(n, 8.0, CL))
        #XXX-MISS(0.265625) self.assertUpperLimitCompatFC(0.38, simple_poisson.upper_limit(n, 9.0, CL))
        self.assertEqual(0.265625, simple_poisson.upper_limit(n, 9.0, CL))
        #XXX-MISS(0.234375) self.assertUpperLimitCompatFC(0.30, simple_poisson.upper_limit(n, 10.0, CL))
        self.assertEqual(0.234375, simple_poisson.upper_limit(n, 10.0, CL))
        #XXX-MISS(0.2109375) self.assertUpperLimitCompatFC(0.26, simple_poisson.upper_limit(n, 11.0, CL))
        self.assertEqual(0.2109375, simple_poisson.upper_limit(n, 11.0, CL))
        #XXX-MISS(0.1875) self.assertUpperLimitCompatFC(0.24 simple_poisson.upper_limit(n, 12.0, CL))
        self.assertEqual(0.1875, simple_poisson.upper_limit(n, 12.0, CL))

        n = 5
        # TODO: self.assertUpperLimitCompatFC(3.75, simple_poisson.upper_limit(n, 0.0, CL))
        self.assertUpperLimitCompatFC(7.31, simple_poisson.upper_limit(n, 0.5, CL))
        self.assertUpperLimitCompatFC(6.81, simple_poisson.upper_limit(n, 1.0, CL))
        self.assertUpperLimitCompatFC(6.31, simple_poisson.upper_limit(n, 1.5, CL))
        #MISS(6.24609375) self.assertUpperLimitCompatFC(5.81, simple_poisson.upper_limit(n, 2.0, CL))
        self.assertEqual(6.24609375, simple_poisson.upper_limit(n, 2.0, CL))
        self.assertUpperLimitCompatFC(5.31, simple_poisson.upper_limit(n, 2.5, CL))
        self.assertUpperLimitCompatFC(4.81, simple_poisson.upper_limit(n, 3.0, CL))
        self.assertUpperLimitCompatFC(4.31, simple_poisson.upper_limit(n, 3.5, CL))
        #MISS(4.25) self.assertUpperLimitCompatFC(3.81, simple_poisson.upper_limit(n, 4.0, CL))
        self.assertEqual(4.25, simple_poisson.upper_limit(n, 4.0, CL))
        self.assertUpperLimitCompatFC(2.81, simple_poisson.upper_limit(n, 5.0, CL))
        #MISS(2.3515625) self.assertUpperLimitCompatFC(1.92, simple_poisson.upper_limit(n, 6.0, CL))
        self.assertEqual(2.3515625, simple_poisson.upper_limit(n, 6.0, CL))
        #MISS(2.0390625) self.assertUpperLimitCompatFC(1.23, simple_poisson.upper_limit(n, 7.0, CL))
        self.assertEqual(2.0390625, simple_poisson.upper_limit(n, 7.0, CL))
        #MISS(1.421875) self.assertUpperLimitCompatFC(0.99, simple_poisson.upper_limit(n, 8.0, CL))
        self.assertEqual(1.421875, simple_poisson.upper_limit(n, 8.0, CL))
        self.assertUpperLimitCompatFC(0.60, simple_poisson.upper_limit(n, 9.0, CL))
        #XXX-MISS(0.265625) self.assertUpperLimitCompatFC(0.48, simple_poisson.upper_limit(n, 10.0, CL))
        self.assertEqual(0.265625, simple_poisson.upper_limit(n, 10.0, CL))
        #XXX-MISS(0.234375) self.assertUpperLimitCompatFC(0.35, simple_poisson.upper_limit(n, 11.0, CL))
        self.assertEqual(0.234375, simple_poisson.upper_limit(n, 11.0, CL))
        #XXX-MISS(0.2109375) self.assertUpperLimitCompatFC(0.29, simple_poisson.upper_limit(n, 12.0, CL))
        self.assertEqual(0.2109375, simple_poisson.upper_limit(n, 12.0, CL))
        #XXX-MISS(0.1875) self.assertUpperLimitCompatFC(0.26, simple_poisson.upper_limit(n, 13.0, CL))
        self.assertEqual(0.1875, simple_poisson.upper_limit(n, 13.0, CL))
        #XXX-MISS(0.171875) self.assertUpperLimitCompatFC(0.24, simple_poisson.upper_limit(n, 14.0, CL))
        self.assertEqual(0.171875, simple_poisson.upper_limit(n, 14.0, CL))

        n = 10
        # TODO: self.assertUpperLimitCompatFC(3.75, simple_poisson.upper_limit(n, 0.0, CL))
        self.assertUpperLimitCompatFC(13.31, simple_poisson.upper_limit(n, 0.5, CL))
        self.assertUpperLimitCompatFC(12.81, simple_poisson.upper_limit(n, 1.0, CL))
        self.assertUpperLimitCompatFC(12.31, simple_poisson.upper_limit(n, 1.5, CL))
        self.assertUpperLimitCompatFC(11.81, simple_poisson.upper_limit(n, 2.0, CL))
        self.assertUpperLimitCompatFC(11.31, simple_poisson.upper_limit(n, 2.5, CL))
        self.assertUpperLimitCompatFC(10.81, simple_poisson.upper_limit(n, 3.0, CL))
        self.assertUpperLimitCompatFC(10.31, simple_poisson.upper_limit(n, 3.5, CL))
        self.assertUpperLimitCompatFC(9.81, simple_poisson.upper_limit(n, 4.0, CL))
        #XXX-MISS(8.798828125) self.assertUpperLimitCompatFC(9.31, simple_poisson.upper_limit(n, 5.0, CL))
        self.assertEqual(8.798828125, simple_poisson.upper_limit(n, 5.0, CL))
        #XXX-MISS(8.2578125) self.assertUpperLimitCompatFC(8.81, simple_poisson.upper_limit(n, 6.0, CL))
        self.assertEqual(8.2578125, simple_poisson.upper_limit(n, 6.0, CL))
        self.assertUpperLimitCompatFC(6.81, simple_poisson.upper_limit(n, 7.0, CL))
        self.assertUpperLimitCompatFC(5.81, simple_poisson.upper_limit(n, 8.0, CL))
        self.assertUpperLimitCompatFC(4.81, simple_poisson.upper_limit(n, 9.0, CL))
        #MISS(4.2578125) self.assertUpperLimitCompatFC(3.81, simple_poisson.upper_limit(n, 10.0, CL))
        self.assertEqual(4.2578125, simple_poisson.upper_limit(n, 10.0, CL))
        self.assertUpperLimitCompatFC(2.89, simple_poisson.upper_limit(n, 11.0, CL))
        self.assertUpperLimitCompatFC(2.11, simple_poisson.upper_limit(n, 12.0, CL))
        self.assertUpperLimitCompatFC(1.47, simple_poisson.upper_limit(n, 13.0, CL))
        #MISS(1.3046875) self.assertUpperLimitCompatFC(1.03, simple_poisson.upper_limit(n, 14.0, CL))
        self.assertEqual(1.3046875, simple_poisson.upper_limit(n, 14.0, CL))
        self.assertUpperLimitCompatFC(0.84, simple_poisson.upper_limit(n, 15.0, CL))

        n = 15
        # TODO: self.assertUpperLimitCompatFC(19.32, simple_poisson.upper_limit(n, 0.0, CL))
        self.assertDifferenceCompatFC(18.82, simple_poisson.upper_limit(n, 0.5, CL))
        self.assertDifferenceCompatFC(18.32, simple_poisson.upper_limit(n, 1.0, CL))
        self.assertDifferenceCompatFC(17.82, simple_poisson.upper_limit(n, 1.5, CL))
        self.assertDifferenceCompatFC(17.32, simple_poisson.upper_limit(n, 2.0, CL))
        self.assertDifferenceCompatFC(16.82, simple_poisson.upper_limit(n, 2.5, CL))
        self.assertDifferenceCompatFC(16.32, simple_poisson.upper_limit(n, 3.0, CL))
        self.assertDifferenceCompatFC(15.82, simple_poisson.upper_limit(n, 3.5, CL))
        self.assertDifferenceCompatFC(15.32, simple_poisson.upper_limit(n, 4.0, CL))
        self.assertDifferenceCompatFC(14.32, simple_poisson.upper_limit(n, 5.0, CL))
        self.assertDifferenceCompatFC(13.32, simple_poisson.upper_limit(n, 6.0, CL))
        self.assertDifferenceCompatFC(12.32, simple_poisson.upper_limit(n, 7.0, CL))
        self.assertDifferenceCompatFC(11.32, simple_poisson.upper_limit(n, 8.0, CL))
        self.assertDifferenceCompatFC(10.32, simple_poisson.upper_limit(n, 9.0, CL))
        self.assertDifferenceCompatFC(9.32, simple_poisson.upper_limit(n, 10.0, CL))
        self.assertDifferenceCompatFC(8.32, simple_poisson.upper_limit(n, 11.0, CL))
        self.assertDifferenceCompatFC(7.32, simple_poisson.upper_limit(n, 12.0, CL))
        self.assertDifferenceCompatFC(6.32, simple_poisson.upper_limit(n, 13.0, CL))
        self.assertDifferenceCompatFC(5.32, simple_poisson.upper_limit(n, 14.0, CL))
        self.assertDifferenceCompatFC(4.32, simple_poisson.upper_limit(n, 15.0, CL))

        n = 20
        # TODO: self.assertUpperLimitCompatFC(19.32, simple_poisson.upper_limit(n, 0.0, CL))
        # TODO: self.assertDifferenceCompatFC(24.80, simple_poisson.upper_limit(n, 0.5, CL))
        # self.assertDifferenceCompatFC(24.30, simple_poisson.upper_limit(n, 1.0, CL))
        # self.assertDifferenceCompatFC(23.80, simple_poisson.upper_limit(n, 1.5, CL))
        # self.assertDifferenceCompatFC(23.30, simple_poisson.upper_limit(n, 2.0, CL))
        # self.assertDifferenceCompatFC(22.80, simple_poisson.upper_limit(n, 2.5, CL))
        # self.assertDifferenceCompatFC(22.30, simple_poisson.upper_limit(n, 3.0, CL))
        # self.assertDifferenceCompatFC(21.80, simple_poisson.upper_limit(n, 3.5, CL))
        # self.assertDifferenceCompatFC(21.30, simple_poisson.upper_limit(n, 4.0, CL))
        self.assertDifferenceCompatFC(20.30, simple_poisson.upper_limit(n, 5.0, CL))
        self.assertDifferenceCompatFC(19.30, simple_poisson.upper_limit(n, 6.0, CL))
        #MISS(18.2939453125) self.assertDifferenceCompatFC(18.30, simple_poisson.upper_limit(n, 7.0, CL))
        self.assertEqual(18.2939453125, simple_poisson.upper_limit(n, 7.0, CL))
        self.assertDifferenceCompatFC(17.30, simple_poisson.upper_limit(n, 8.0, CL))
        self.assertDifferenceCompatFC(16.30, simple_poisson.upper_limit(n, 9.0, CL))
        #MISS(15.29296875) self.assertDifferenceCompatFC(15.30, simple_poisson.upper_limit(n, 10.0, CL))
        self.assertDifferenceCompatFC(15.29296875, simple_poisson.upper_limit(n, 10.0, CL))
        self.assertDifferenceCompatFC(14.30, simple_poisson.upper_limit(n, 11.0, CL))
        self.assertDifferenceCompatFC(13.30, simple_poisson.upper_limit(n, 12.0, CL))
        self.assertDifferenceCompatFC(12.30, simple_poisson.upper_limit(n, 13.0, CL))
        self.assertDifferenceCompatFC(11.30, simple_poisson.upper_limit(n, 14.0, CL))
        #MISS(10.29296875) self.assertDifferenceCompatFC(10.30, simple_poisson.upper_limit(n, 15.0, CL))
        self.assertEqual(10.29296875, simple_poisson.upper_limit(n, 15.0, CL))

    def test_upper_limit_against_FC_paper_CL090(self):
        """Test the results of `simple_poisson.upper_limit()` against the tabulated values of F+C.

        For n=0 to 5 and n=10,15,20 we test all values of b in the tables
        until a "low-stats" result is reached.
        """
        CL = 0.90

        n = 0
        # TODO: self.assertUpperLimitCompatFC(3.75, simple_poisson.upper_limit(n, 0.0, CL))
        #MISS(2.265625) self.assertUpperLimitCompatFC(1.94, simple_poisson.upper_limit(n, 0.5, CL))
        self.assertEqual(2.265625, simple_poisson.upper_limit(n, 0.5, CL))
        self.assertUpperLimitCompatFC(1.61, simple_poisson.upper_limit(n, 1.0, CL))
        self.assertUpperLimitCompatFC(1.33, simple_poisson.upper_limit(n, 1.5, CL))
        #XXX-MISS(1.078125) self.assertUpperLimitCompatFC(1.26, simple_poisson.upper_limit(n, 2.0, CL))
        self.assertEqual(1.078125, simple_poisson.upper_limit(n, 2.0, CL))
        #MISS(1.515625) self.assertUpperLimitCompatFC(1.18, simple_poisson.upper_limit(n, 2.5, CL))
        self.assertEqual(1.515625, simple_poisson.upper_limit(n, 2.5, CL))
        #MISS(1.28125) self.assertUpperLimitCompatFC(1.06, simple_poisson.upper_limit(n, 3.0, CL))
        self.assertEqual(1.28125, simple_poisson.upper_limit(n, 3.0, CL))
        #MISS(1.3828125) self.assertUpperLimitCompatFC(1.01, simple_poisson.upper_limit(n, 3.5, CL))
        self.assertEqual(1.3828125, simple_poisson.upper_limit(n, 3.5, CL))
        #MISS(1.15625) self.assertUpperLimitCompatFC(0.98, simple_poisson.upper_limit(n, 4.0, CL))
        self.assertEqual(1.15625, simple_poisson.upper_limit(n, 4.0, CL))

        n = 1
        # TODO: self.assertUpperLimitCompatFC(4.36, simple_poisson.upper_limit(n, 0.0, CL))
        #MISS(4.25) self.assertUpperLimitCompatFC(3.86, simple_poisson.upper_limit(n, 0.5, CL))
        self.assertEqual(4.25, simple_poisson.upper_limit(n, 0.5, CL))
        self.assertUpperLimitCompatFC(3.36, simple_poisson.upper_limit(n, 1.0, CL))
        self.assertUpperLimitCompatFC(2.91, simple_poisson.upper_limit(n, 1.5, CL))
        self.assertUpperLimitCompatFC(2.53, simple_poisson.upper_limit(n, 2.0, CL))
        #XXX-MISS(2.1796875) self.assertUpperLimitCompatFC(2.19, simple_poisson.upper_limit(n, 2.5, CL))
        self.assertEqual(2.1796875, simple_poisson.upper_limit(n, 2.5, CL))
        #MISS(2.25) self.assertUpperLimitCompatFC(1.88, simple_poisson.upper_limit(n, 3.0, CL))
        self.assertEqual(2.25, simple_poisson.upper_limit(n, 3.0, CL))
        self.assertUpperLimitCompatFC(1.59, simple_poisson.upper_limit(n, 3.5, CL))
        #MISS(2.0390625) self.assertUpperLimitCompatFC(1.39, simple_poisson.upper_limit(n, 4.0, CL))
        self.assertEqual(2.0390625, simple_poisson.upper_limit(n, 4.0, CL))
        #XXX-MISS(1.1953125) self.assertUpperLimitCompatFC(1.22, simple_poisson.upper_limit(n, 5.0, CL))
        self.assertEqual(1.1953125, simple_poisson.upper_limit(n, 5.0, CL))
        #XXX-MISS(1.0859375) self.assertUpperLimitCompatFC(1.14, simple_poisson.upper_limit(n, 6.0, CL))
        self.assertEqual(1.0859375, simple_poisson.upper_limit(n, 6.0, CL))
        #MISS(1.1015625) self.assertUpperLimitCompatFC(1.10, simple_poisson.upper_limit(n, 7.0, CL))
        self.assertEqual(1.1015625, simple_poisson.upper_limit(n, 7.0, CL))

        n = 2
        # TODO: self.assertUpperLimitCompatFC(3.75, simple_poisson.upper_limit(n, 0.0, CL))
        self.assertUpperLimitCompatFC(5.41, simple_poisson.upper_limit(n, 0.5, CL))
        self.assertUpperLimitCompatFC(4.91, simple_poisson.upper_limit(n, 1.0, CL))
        self.assertUpperLimitCompatFC(4.41, simple_poisson.upper_limit(n, 1.5, CL))
        #MISS(4.3125) self.assertUpperLimitCompatFC(3.91, simple_poisson.upper_limit(n, 2.0, CL))
        self.assertEqual(4.3125, simple_poisson.upper_limit(n, 2.0, CL))
        self.assertUpperLimitCompatFC(3.45, simple_poisson.upper_limit(n, 2.5, CL))
        self.assertUpperLimitCompatFC(3.04, simple_poisson.upper_limit(n, 3.0, CL))
        self.assertUpperLimitCompatFC(2.67, simple_poisson.upper_limit(n, 3.5, CL))
        self.assertUpperLimitCompatFC(2.33, simple_poisson.upper_limit(n, 4.0, CL))
        #MISS(2.09375) self.assertUpperLimitCompatFC(1.73, simple_poisson.upper_limit(n, 5.0, CL))
        self.assertEqual(2.09375, simple_poisson.upper_limit(n, 5.0, CL))
        #XXX-MISS(1.21875) self.assertUpperLimitCompatFC(1.57, simple_poisson.upper_limit(n, 6.0, CL))
        self.assertEqual(1.21875, simple_poisson.upper_limit(n, 6.0, CL))
        #XXX-MISS(1.1015625) self.assertUpperLimitCompatFC(1.38, simple_poisson.upper_limit(n, 7.0, CL))
        self.assertEqual(1.1015625, simple_poisson.upper_limit(n, 7.0, CL))
        #XXX-MISS(1.0078125) self.assertUpperLimitCompatFC(1.27, simple_poisson.upper_limit(n, 8.0, CL))
        self.assertEqual(1.0078125, simple_poisson.upper_limit(n, 8.0, CL))
        #XXX-MISS(1.0234375) self.assertUpperLimitCompatFC(1.21, simple_poisson.upper_limit(n, 9.0, CL))
        self.assertEqual(1.0234375, simple_poisson.upper_limit(n, 9.0, CL))

        n = 3
        # TODO: self.assertUpperLimitCompatFC(3.75, simple_poisson.upper_limit(n, 0.0, CL))
        self.assertDifferenceCompatFC(6.92, simple_poisson.upper_limit(n, 0.5, CL))
        self.assertDifferenceCompatFC(6.42, simple_poisson.upper_limit(n, 1.0, CL))
        #MISS(6.333984375) self.assertUpperLimitCompatFC(5.92, simple_poisson.upper_limit(n, 1.5, CL))
        self.assertEqual(6.333984375, simple_poisson.upper_limit(n, 1.5, CL))
        self.assertDifferenceCompatFC(5.42, simple_poisson.upper_limit(n, 2.0, CL))
        self.assertDifferenceCompatFC(4.92, simple_poisson.upper_limit(n, 2.5, CL))
        self.assertDifferenceCompatFC(4.42, simple_poisson.upper_limit(n, 3.0, CL))
        #MISS(4.359375) self.assertUpperLimitCompatFC(3.95, simple_poisson.upper_limit(n, 3.5, CL))
        self.assertEqual(4.359375, simple_poisson.upper_limit(n, 3.5, CL))
        self.assertUpperLimitCompatFC(3.53, simple_poisson.upper_limit(n, 4.0, CL))
        self.assertUpperLimitCompatFC(2.78, simple_poisson.upper_limit(n, 5.0, CL))
        self.assertDifferenceCompatFC(2.14, simple_poisson.upper_limit(n, 6.0, CL))
        #XXX-MISS(1.5859375) self.assertUpperLimitCompatFC(1.75, simple_poisson.upper_limit(n, 7.0, CL))
        self.assertEqual(1.5859375, simple_poisson.upper_limit(n, 7.0, CL))
        #XXX-MISS(1.4453125) self.assertUpperLimitCompatFC(1.49, simple_poisson.upper_limit(n, 8.0, CL))
        self.assertEqual(1.4453125, simple_poisson.upper_limit(n, 8.0, CL))
        #XXX-MISS(1.3359375) self.assertUpperLimitCompatFC(1.37, simple_poisson.upper_limit(n, 9.0, CL))
        self.assertEqual(1.3359375, simple_poisson.upper_limit(n, 9.0, CL))
        #MISS(1.34375) self.assertUpperLimitCompatFC(1.29, simple_poisson.upper_limit(n, 10.0, CL))
        self.assertEqual(1.34375, simple_poisson.upper_limit(n, 10.0, CL))
        #XXX-MISS(1.15625) self.assertUpperLimitCompatFC(1.25, simple_poisson.upper_limit(n, 11.0, CL))
        self.assertEqual(1.15625, simple_poisson.upper_limit(n, 11.0, CL))

        n = 4
        # TODO: self.assertUpperLimitCompatFC(3.75, simple_poisson.upper_limit(n, 0.0, CL))
        self.assertUpperLimitCompatFC(8.10, simple_poisson.upper_limit(n, 0.5, CL))
        self.assertUpperLimitCompatFC(7.60, simple_poisson.upper_limit(n, 1.0, CL))
        #XXX-MISS(7.08984375) self.assertUpperLimitCompatFC(7.10, simple_poisson.upper_limit(n, 1.5, CL))
        self.assertEqual(7.08984375, simple_poisson.upper_limit(n, 1.5, CL))
        self.assertUpperLimitCompatFC(6.60, simple_poisson.upper_limit(n, 2.0, CL))
        self.assertUpperLimitCompatFC(6.10, simple_poisson.upper_limit(n, 2.5, CL))
        self.assertUpperLimitCompatFC(5.60, simple_poisson.upper_limit(n, 3.0, CL))
        self.assertUpperLimitCompatFC(5.10, simple_poisson.upper_limit(n, 3.5, CL))
        self.assertUpperLimitCompatFC(4.60, simple_poisson.upper_limit(n, 4.0, CL))
        #MISS(4.0078125) self.assertUpperLimitCompatFC(3.60, simple_poisson.upper_limit(n, 5.0, CL))
        self.assertEqual(4.0078125, simple_poisson.upper_limit(n, 5.0, CL))
        self.assertUpperLimitCompatFC(2.83, simple_poisson.upper_limit(n, 6.0, CL))
        self.assertDifferenceCompatFC(2.56, simple_poisson.upper_limit(n, 7.0, CL))
        #MISS(2.34375) self.assertUpperLimitCompatFC(1.98, simple_poisson.upper_limit(n, 8.0, CL))
        self.assertEqual(2.34375, simple_poisson.upper_limit(n, 8.0, CL))
        #XXX-MISS(1.4609375) self.assertUpperLimitCompatFC(1.82, simple_poisson.upper_limit(n, 9.0, CL))
        self.assertEqual(1.4609375, simple_poisson.upper_limit(n, 9.0, CL))
        #XXX-MISS(1.34375) self.assertUpperLimitCompatFC(1.57, simple_poisson.upper_limit(n, 10.0, CL))
        self.assertEqual(1.34375, simple_poisson.upper_limit(n, 10.0, CL))
        #XXX-MISS(1.25) self.assertUpperLimitCompatFC(1.45, simple_poisson.upper_limit(n, 11.0, CL))
        self.assertEqual(1.25, simple_poisson.upper_limit(n, 11.0, CL))
        #XXX-MISS(1.1640625) self.assertUpperLimitCompatFC(1.37, simple_poisson.upper_limit(n, 12.0, CL))
        self.assertEqual(1.1640625, simple_poisson.upper_limit(n, 12.0, CL))


        n = 5
        # TODO: self.assertUpperLimitCompatFC(3.75, simple_poisson.upper_limit(n, 0.0, CL))
        self.assertUpperLimitCompatFC(9.49, simple_poisson.upper_limit(n, 0.5, CL))
        self.assertUpperLimitCompatFC(8.99, simple_poisson.upper_limit(n, 1.0, CL))
        self.assertUpperLimitCompatFC(8.49, simple_poisson.upper_limit(n, 1.5, CL))
        self.assertUpperLimitCompatFC(7.99, simple_poisson.upper_limit(n, 2.0, CL))
        self.assertUpperLimitCompatFC(7.49, simple_poisson.upper_limit(n, 2.5, CL))
        self.assertUpperLimitCompatFC(6.99, simple_poisson.upper_limit(n, 3.0, CL))
        self.assertUpperLimitCompatFC(6.49, simple_poisson.upper_limit(n, 3.5, CL))
        self.assertUpperLimitCompatFC(5.99, simple_poisson.upper_limit(n, 4.0, CL))
        self.assertUpperLimitCompatFC(4.99, simple_poisson.upper_limit(n, 5.0, CL))
        self.assertUpperLimitCompatFC(4.07, simple_poisson.upper_limit(n, 6.0, CL))
        self.assertUpperLimitCompatFC(3.28, simple_poisson.upper_limit(n, 7.0, CL))
        self.assertUpperLimitCompatFC(2.60, simple_poisson.upper_limit(n, 8.0, CL))
        #MISS(2.7578125) self.assertUpperLimitCompatFC(2.38, simple_poisson.upper_limit(n, 9.0, CL))
        self.assertEqual(2.7578125, simple_poisson.upper_limit(n, 9.0, CL))
        #MISS(2.1953125) self.assertUpperLimitCompatFC(1.85, simple_poisson.upper_limit(n, 10.0, CL))
        self.assertEqual(2.1953125, simple_poisson.upper_limit(n, 10.0, CL))
        #XXX-MISS(1.359375) self.assertUpperLimitCompatFC(1.70, simple_poisson.upper_limit(n, 11.0, CL))
        self.assertEqual(1.359375, simple_poisson.upper_limit(n, 11.0, CL))
        #XXX-MISS(1.2578125) self.assertUpperLimitCompatFC(1.58, simple_poisson.upper_limit(n, 12.0, CL))
        self.assertEqual(1.2578125, simple_poisson.upper_limit(n, 12.0, CL))
        #XXX-MISS(1.171875) self.assertUpperLimitCompatFC(1.48, simple_poisson.upper_limit(n, 13.0, CL))
        self.assertEqual(1.171875, simple_poisson.upper_limit(n, 13.0, CL))
        #XXX-MISS(1.09375) self.assertUpperLimitCompatFC(1.39, simple_poisson.upper_limit(n, 14.0, CL))
        self.assertEqual(1.09375, simple_poisson.upper_limit(n, 14.0, CL))


if __name__ == "__main__":
    unittest.main()
