import unittest
from random import shuffle

from unified_ci import tools, simple_gaussian

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

    def DISABLEDtest_conservative_quantile(self):
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
    def test_lower_limit_against_FC_paper(self):
        """
        Test that difference is below 0.005 (= 0.5 * precision of F+C table)
        """
        def test(a, b):
            self.assertGreater(0.005, abs(a - b))

        test(0.0, simple_gaussian.lower_limit(-2.3, 1.0, 0.6827))
        test(0.0, simple_gaussian.lower_limit(-2.3, 1.0, 0.90))
        test(0.0, simple_gaussian.lower_limit(-2.3, 1.0, 0.95))
        test(0.0, simple_gaussian.lower_limit(-2.3, 1.0, 0.99))

        test(0.02, simple_gaussian.lower_limit(0.5, 1.0, 0.6827))
        test(0.00, simple_gaussian.lower_limit(0.5, 1.0, 0.90))
        test(0.00, simple_gaussian.lower_limit(0.5, 1.0, 0.95))
        test(0.00, simple_gaussian.lower_limit(0.5, 1.0, 0.99))

        test(0.42, simple_gaussian.lower_limit(1.3, 1.0, 0.6827))
        test(0.02, simple_gaussian.lower_limit(1.3, 1.0, 0.90))
        test(0.00, simple_gaussian.lower_limit(1.3, 1.0, 0.95))
        test(0.00, simple_gaussian.lower_limit(1.3, 1.0, 0.99))

        test(0.72, simple_gaussian.lower_limit(1.7, 1.0, 0.6827))
        test(0.38, simple_gaussian.lower_limit(1.7, 1.0, 0.90))
        test(0.06, simple_gaussian.lower_limit(1.7, 1.0, 0.95))
        test(0.00, simple_gaussian.lower_limit(1.7, 1.0, 0.99))

        test(1.40, simple_gaussian.lower_limit(2.4, 1.0, 0.6827))
        test(0.87, simple_gaussian.lower_limit(2.4, 1.0, 0.90))
        test(0.69, simple_gaussian.lower_limit(2.4, 1.0, 0.95))
        test(0.07, simple_gaussian.lower_limit(2.4, 1.0, 0.99))

if __name__ == "__main__":
    unittest.main()
