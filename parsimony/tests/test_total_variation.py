# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 16:17:15 2014

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import numpy as np

from parsimony.tests import TestCase


class TestTotalVariation(TestCase):

    def test_nonsmooth(self):

        from parsimony.functions import CombinedFunction
        import parsimony.algorithms.explicit as explicit
        import parsimony.functions as functions
        import parsimony.functions.nesterov.tv as tv
        import parsimony.datasets.simulated.l1_l2_tv as l1_l2_tv
        import parsimony.utils.start_vectors as start_vectors

        np.random.seed(42)

        px = 4
        py = 4
        pz = 4
        shape = (pz, py, px)
        n, p = 50, np.prod(shape)

        l = 0.0
        k = 0.0
        g = 1.1

        start_vector = start_vectors.RandomStartVector(normalise=True)
        beta = start_vector.get_vector(p)

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        snr = 100.0

        A, _ = tv.A_from_shape(shape)
        X, y, beta_star = l1_l2_tv.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        eps = 1e-8
        max_iter = 12500

        beta_start = start_vector.get_vector(p)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = explicit.FISTA(eps=eps, max_iter=max_iter / len(mus))

        beta_parsimony = beta_start
        for mu in mus:
#            function = functions.LinearRegressionL1L2GL(X, y, l, k, g,
#                                                        A=A, mu=mu,
#                                                        penalty_start=0)

            function = CombinedFunction()
            function.add_function(functions.losses.LinearRegression(X, y,
                                                                   mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))

            beta_parsimony = fista.run(function, beta_parsimony)

        berr = np.linalg.norm(beta_parsimony - beta_star)
#        print "berr:", berr
        assert berr < 5e-2

        f_parsimony = function.f(beta_parsimony)
        f_star = function.f(beta_star)
        ferr = abs(f_parsimony - f_star)
#        print "ferr:", ferr
        assert ferr < 5e-4

    def test_smooth(self):

        from parsimony.functions import CombinedFunction
        import parsimony.algorithms.explicit as explicit
        import parsimony.functions as functions
        import parsimony.functions.nesterov.tv as tv
        import parsimony.datasets.simulated.l1_l2_tvmu as l1_l2_tvmu
        import parsimony.utils.start_vectors as start_vectors

        np.random.seed(1337)

        px = 4
        py = 4
        pz = 4
        shape = (pz, py, px)
        n, p = 50, np.prod(shape)

        l = 0.0
        k = 0.0
        g = 0.9

        start_vector = start_vectors.RandomStartVector(normalise=True)
        beta = start_vector.get_vector(p)

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        snr = 100.0

        A, _ = tv.A_from_shape(shape)
        mu_min = 5e-8
        X, y, beta_star = l1_l2_tvmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                          A=A, mu=mu_min, snr=snr)

        eps = 1e-8
        max_iter = 17700

        beta_start = start_vector.get_vector(p)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = explicit.FISTA(eps=eps, max_iter=max_iter / len(mus))

        beta_parsimony = beta_start
        for mu in mus:
#            function = functions.LinearRegressionL1L2GL(X, y, l, k, g,
#                                                        A=A, mu=mu,
#                                                        penalty_start=0)

            function = CombinedFunction()
            function.add_function(functions.losses.LinearRegression(X, y,
                                                               mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))

            beta_parsimony = fista.run(function, beta_parsimony)

        berr = np.linalg.norm(beta_parsimony - beta_star)
#        print "berr:", berr
        assert berr < 5e-2

        f_parsimony = function.f(beta_parsimony)
        f_star = function.f(beta_star)
        ferr = abs(f_parsimony - f_star)
#        print "ferr:", ferr
        assert ferr < 5e-5

    def test_combo_nonsmooth(self):

        from parsimony.functions import CombinedFunction
        import parsimony.algorithms.explicit as explicit
        import parsimony.functions as functions
        import parsimony.functions.nesterov.tv as tv
        import parsimony.datasets.simulated.l1_l2_tv as l1_l2_tv
        import parsimony.utils.start_vectors as start_vectors

        np.random.seed(42)

        px = 4
        py = 4
        pz = 4
        shape = (pz, py, px)
        n, p = 50, np.prod(shape)

        l = 0.618
        k = 1.0 - l
        g = 1.1

        start_vector = start_vectors.RandomStartVector(normalise=True)
        beta = start_vector.get_vector(p)

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        snr = 100.0

        A, _ = tv.A_from_shape(shape)
        X, y, beta_star = l1_l2_tv.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        eps = 1e-8
        max_iter = 5300

        beta_start = start_vector.get_vector(p)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = explicit.FISTA(eps=eps, max_iter=max_iter / len(mus))

        beta_parsimony = beta_start
        for mu in mus:
#            function = functions.LinearRegressionL1L2GL(X, y, l, k, g,
#                                                        A=A, mu=mu,
#                                                        penalty_start=0)

            function = CombinedFunction()
            function.add_function(functions.losses.LinearRegression(X, y,
                                                                   mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            function.add_penalty(functions.penalties.L2(l=k))
            function.add_prox(functions.penalties.L1(l=l))

            beta_parsimony = fista.run(function, beta_parsimony)

        berr = np.linalg.norm(beta_parsimony - beta_star)
#        print "berr:", berr
        assert berr < 5e-3

        f_parsimony = function.f(beta_parsimony)
        f_star = function.f(beta_star)
        ferr = abs(f_parsimony - f_star)
#        print "ferr:", ferr
        assert ferr < 5e-5

    def test_combo_smooth(self):

        from parsimony.functions import CombinedFunction
        import parsimony.algorithms.explicit as explicit
        import parsimony.functions as functions
        import parsimony.functions.nesterov.tv as tv
        import parsimony.datasets.simulated.l1_l2_tvmu as l1_l2_tvmu
        import parsimony.utils.start_vectors as start_vectors

        np.random.seed(42)

        px = 4
        py = 4
        pz = 4
        shape = (pz, py, px)
        n, p = 50, np.prod(shape)

        l = 0.618
        k = 1.0 - l
        g = 1.1

        start_vector = start_vectors.RandomStartVector(normalise=True)
        beta = start_vector.get_vector(p)

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        snr = 100.0

        A, _ = tv.A_from_shape(shape)
        mu_min = 5e-8
        X, y, beta_star = l1_l2_tvmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                          A=A, mu=mu_min, snr=snr)

        eps = 1e-8
        max_iter = 5300

        beta_start = start_vector.get_vector(p)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = explicit.FISTA(eps=eps, max_iter=max_iter / len(mus))

        beta_parsimony = beta_start
        for mu in mus:
#            function = functions.LinearRegressionL1L2GL(X, y, l, k, g,
#                                                        A=A, mu=mu,
#                                                        penalty_start=0)

            function = CombinedFunction()
            function.add_function(functions.losses.LinearRegression(X, y,
                                                               mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            function.add_penalty(functions.penalties.L2(l=k))
            function.add_prox(functions.penalties.L1(l=l))

            beta_parsimony = fista.run(function, beta_parsimony)

        berr = np.linalg.norm(beta_parsimony - beta_star)
#        print "berr:", berr
        assert berr < 5e-3

        f_parsimony = function.f(beta_parsimony)
        f_star = function.f(beta_star)
        ferr = abs(f_parsimony - f_star)
#        print "ferr:", ferr
        assert ferr < 5e-5

if __name__ == "__main__":
    import unittest
    unittest.main()
