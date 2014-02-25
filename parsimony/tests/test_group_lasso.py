# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 11:03:30 2014

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import unittest

import numpy as np

from tests import TestCase


class TestGroupLasso(TestCase):

    def test_nonoverlapping_nonsmooth(self):
        """
        Reference: http://spams-devel.gforge.inria.fr/doc-python/doc_spams.pdf
        """
        from parsimony.functions import CombinedFunction
        import parsimony.algorithms.explicit as explicit
        import parsimony.functions as functions
        import parsimony.functions.nesterov.gl as gl
        import parsimony.datasets.simulated.l1_l2_gl as l1_l2_gl
        import parsimony.start_vectors as start_vectors

        np.random.seed(42)

        # Note that p must be even!
        n, p = 25, 20
        groups = [range(0, p / 2), range(p / 2, p)]
#        weights = [1.5, 0.5]

        A = gl.A_from_groups(p, groups=groups)  # , weights=weights)

        l = 0.0
        k = 0.0
        g = 1.0

        start_vector = start_vectors.RandomStartVector(normalise=True)
        beta = start_vector.get_vector((p, 1))

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        snr = 100.0

        X, y, beta_star = l1_l2_gl.load(l, k, g, beta, M, e, A, snr=snr)

        eps = 1e-8
        max_iter = 10000

        beta_start = start_vector.get_vector((p, 1))

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = explicit.FISTA(eps=eps, max_iter=max_iter / len(mus))

        beta_parsimony = beta_start
        for mu in mus:
#            function = functions.RR_L1_GL(X, y, k, l, g, A=A, mu=mu,
#                                          penalty_start=0)

            function = CombinedFunction()
            function.add_function(functions.losses.LinearRegression(X, y,
                                                               mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))

            beta_parsimony = fista.run(function, beta_parsimony)

        try:
            import spams

            params = {"loss": "square",
                      "regul": "group-lasso-l2",
                      "groups": np.array([1] * (p / 2) + [2] * (p / 2),
                                         dtype=np.int32),
                      "lambda1": g,
                      "max_it": max_iter,
                      "tol": eps,
                      "ista": False,
                      "numThreads": -1,
                     }
            beta_spams, optim_info = \
                    spams.fistaFlat(Y=np.asfortranarray(y),
                                    X=np.asfortranarray(X),
                                    W0=np.asfortranarray(beta_start),
                                    return_optim_info=True,
                                    **params)

        except ImportError:
            beta_spams = np.asarray([[14.01111427],
                                     [35.56508563],
                                     [27.38245962],
                                     [22.39716553],
                                     [5.835744940],
                                     [5.841502910],
                                     [2.172209350],
                                     [32.40227785],
                                     [22.48364756],
                                     [26.48822401],
                                     [0.770391500],
                                     [36.28288883],
                                     [31.14118214],
                                     [7.938279340],
                                     [6.800713150],
                                     [6.862914540],
                                     [11.38161678],
                                     [19.63087584],
                                     [16.15855845],
                                     [10.89356615]])

        mse = (np.linalg.norm(beta_parsimony - beta_spams) ** 2.0) / p
        assert mse < 1e-5

        f_parsimony = function.f(beta_parsimony)
        f_spams = function.f(beta_spams)
        assert abs(f_parsimony - f_spams) < 1e-5

    def test_nonoverlapping_smooth(self):
        """
        Reference: http://spams-devel.gforge.inria.fr/doc-python/doc_spams.pdf
        """
        from parsimony.functions import CombinedFunction
        import parsimony.algorithms.explicit as explicit
        import parsimony.functions as functions
        import parsimony.functions.nesterov.gl as gl
        import parsimony.datasets.simulated.l1_l2_glmu as l1_l2_glmu
        import parsimony.start_vectors as start_vectors

        np.random.seed(314)

        # Note that p must be even!
        n, p = 25, 20
        groups = [range(0, p / 2), range(p / 2, p)]
#        weights = [1.5, 0.5]

        A = gl.A_from_groups(p, groups=groups)  # , weights=weights)

        l = 0.0
        k = 0.0
        g = 0.9

        start_vector = start_vectors.RandomStartVector(normalise=True)
        beta = start_vector.get_vector((p, 1))

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        snr = 100.0

        mu_min = 5e-8
        X, y, beta_star = l1_l2_glmu.load(l, k, g, beta, M, e, A,
                                          mu=mu_min, snr=snr)

        eps = 1e-8
        max_iter = 10000

        beta_start = start_vector.get_vector((p, 1))

        mus = [5e-0, 5e-2, 5e-4, 5e-6, 5e-8]
        fista = explicit.FISTA(eps=eps, max_iter=max_iter / len(mus))

        beta_parsimony = beta_start
        for mu in mus:
#            function = functions.RR_L1_GL(X, y, k, l, g, A=A, mu=mu,
#                                          penalty_start=0)

            function = CombinedFunction()
            function.add_function(functions.losses.LinearRegression(X, y,
                                                               mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))

            beta_parsimony = fista.run(function, beta_parsimony)

        try:
            import spams

            params = {"loss": "square",
                      "regul": "group-lasso-l2",
                      "groups": np.array([1] * (p / 2) + [2] * (p / 2),
                                         dtype=np.int32),
                      "lambda1": g,
                      "max_it": max_iter,
                      "tol": eps,
                      "ista": False,
                      "numThreads": -1,
                     }
            beta_spams, optim_info = \
                    spams.fistaFlat(Y=np.asfortranarray(y),
                                    X=np.asfortranarray(X),
                                    W0=np.asfortranarray(beta_start),
                                    return_optim_info=True,
                                    **params)

        except ImportError:
            beta_spams = np.asarray([[9.92737817],
                                     [6.25741002],
                                     [2.85462422],
                                     [8.45021308],
                                     [9.85959465],
                                     [8.90571615],
                                     [7.77263765],
                                     [2.87114577],
                                     [9.79103766],
                                     [2.78660721],
                                     [8.20420015],
                                     [2.81858990],
                                     [1.30444549],
                                     [4.10358283],
                                     [9.05604300],
                                     [2.97987576],
                                     [0.71923705],
                                     [6.83698462],
                                     [6.29995241],
                                     [6.25209606]])

        mse = (np.linalg.norm(beta_parsimony - beta_spams) ** 2.0) / p
        assert mse < 0.005

        f_parsimony = function.f(beta_parsimony)
        f_spams = function.f(beta_spams)
        assert abs(f_parsimony - f_spams) < 0.005

    def test_overlapping_nonsmooth(self):

        from parsimony.functions import CombinedFunction
        import parsimony.algorithms.explicit as explicit
        import parsimony.functions as functions
        import parsimony.functions.nesterov.gl as gl
        import parsimony.datasets.simulated.l1_l2_gl as l1_l2_gl
        import parsimony.start_vectors as start_vectors

        np.random.seed(42)

        # Note that p should be divisible by 3!
        n, p = 25, 30
        groups = [range(0, 2 * p / 3), range(p / 3, p)]
        weights = [1.5, 0.5]

        A = gl.A_from_groups(p, groups=groups, weights=weights)

        l = 0.0
        k = 0.0
        g = 1.1

        start_vector = start_vectors.RandomStartVector(normalise=True)
        beta = start_vector.get_vector((p, 1))

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        snr = 100.0

        X, y, beta_star = l1_l2_gl.load(l, k, g, beta, M, e, A, snr=snr)

        eps = 1e-8
        max_iter = 5000

        beta_start = start_vector.get_vector((p, 1))

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = explicit.FISTA(eps=eps, max_iter=max_iter / len(mus))

        beta_parsimony = beta_start
        for mu in mus:
#            function = functions.RR_L1_GL(X, y, k, l, g, A=A, mu=mu,
#                                          penalty_start=0)

            function = CombinedFunction()
            function.add_function(functions.losses.LinearRegression(X, y,
                                                               mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))

            beta_parsimony = fista.run(function, beta_parsimony)

        mse = (np.linalg.norm(beta_parsimony - beta_star) ** 2.0) / p
        assert mse < 0.005

        f_parsimony = function.f(beta_parsimony)
        f_star = function.f(beta_star)
        assert abs(f_parsimony - f_star) < 0.005

    def test_overlapping_smooth(self):

        from parsimony.functions import CombinedFunction
        import parsimony.algorithms.explicit as explicit
        import parsimony.functions as functions
        import parsimony.functions.nesterov.gl as gl
        import parsimony.datasets.simulated.l1_l2_glmu as l1_l2_glmu
        import parsimony.start_vectors as start_vectors

        np.random.seed(314)

        # Note that p must be even!
        n, p = 25, 30
        groups = [range(0, 2 * p / 3), range(p / 3, p)]
        weights = [1.5, 0.5]

        A = gl.A_from_groups(p, groups=groups, weights=weights)

        l = 0.0
        k = 0.0
        g = 0.9

        start_vector = start_vectors.RandomStartVector(normalise=True)
        beta = start_vector.get_vector((p, 1))

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        snr = 100.0

        mu_min = 5e-8
        X, y, beta_star = l1_l2_glmu.load(l, k, g, beta, M, e, A,
                                          mu=mu_min, snr=snr)

        eps = 1e-8
        max_iter = 5000

        beta_start = start_vector.get_vector((p, 1))

        mus = [5e-0, 5e-2, 5e-4, 5e-6, 5e-8]
        fista = explicit.FISTA(eps=eps, max_iter=max_iter / len(mus))

        beta_parsimony = beta_start
        for mu in mus:
#            function = functions.RR_L1_GL(X, y, k, l, g, A=A, mu=mu,
#                                          penalty_start=0)

            function = CombinedFunction()
            function.add_function(functions.losses.LinearRegression(X, y,
                                                               mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))

            beta_parsimony = fista.run(function, beta_parsimony)

        mse = (np.linalg.norm(beta_parsimony - beta_star) ** 2.0) / p
        assert mse < 1e-7

        f_parsimony = function.f(beta_parsimony)
        f_star = function.f(beta_star)
        assert abs(f_parsimony - f_star) < 1e-7

    def test_combo_overlapping_smooth(self):

        from parsimony.functions import CombinedFunction
        import parsimony.algorithms.explicit as explicit
        import parsimony.functions as functions
        import parsimony.functions.nesterov.gl as gl
        import parsimony.datasets.simulated.l1_l2_glmu as l1_l2_glmu
        import parsimony.start_vectors as start_vectors

        np.random.seed(314)

        # Note that p must be even!
        n, p = 25, 30
        groups = [range(0, 2 * p / 3), range(p / 3, p)]
        weights = [1.5, 0.5]

        A = gl.A_from_groups(p, groups=groups, weights=weights)

        l = 0.618
        k = 1.0 - l
        g = 2.718

        start_vector = start_vectors.RandomStartVector(normalise=True)
        beta = start_vector.get_vector((p, 1))

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        snr = 100.0

        mu_min = 5e-8
        X, y, beta_star = l1_l2_glmu.load(l, k, g, beta, M, e, A,
                                          mu=mu_min, snr=snr)

        eps = 1e-8
        max_iter = 2000

        beta_start = start_vector.get_vector((p, 1))

        mus = [5e-0, 5e-2, 5e-4, 5e-6, 5e-8]
        fista = explicit.FISTA(eps=eps, max_iter=max_iter / len(mus))

        beta_parsimony = beta_start
        for mu in mus:
#            function = functions.RR_L1_GL(X, y, k, l, g, A=A, mu=mu,
#                                          penalty_start=0)

            function = CombinedFunction()
            function.add_function(functions.losses.LinearRegression(X, y,
                                                               mean=False))
            function.add_penalty(functions.penalties.L2(l=k))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            function.add_prox(functions.penalties.L1(l=l))

            beta_parsimony = fista.run(function, beta_parsimony)

        mse = (np.linalg.norm(beta_parsimony - beta_star) ** 2.0) / p
        assert mse < 0.0005

        f_parsimony = function.f(beta_parsimony)
        f_star = function.f(beta_star)
        assert abs(f_parsimony - f_star) < 0.001

    def test_combo_overlapping_nonsmooth(self):

        from parsimony.functions import CombinedFunction
        import parsimony.algorithms.explicit as explicit
        import parsimony.functions as functions
        import parsimony.functions.nesterov.gl as gl
        import parsimony.datasets.simulated.l1_l2_gl as l1_l2_gl
        import parsimony.start_vectors as start_vectors

        np.random.seed(42)

        # Note that p must be even!
        n, p = 25, 30
        groups = [range(0, 2 * p / 3), range(p / 3, p)]
        weights = [1.5, 0.5]

        A = gl.A_from_groups(p, groups=groups, weights=weights)

        l = 0.618
        k = 1.0 - l
        g = 2.718

        start_vector = start_vectors.RandomStartVector(normalise=True)
        beta = start_vector.get_vector((p, 1))

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        snr = 100.0

        X, y, beta_star = l1_l2_gl.load(l, k, g, beta, M, e, A, snr=snr)

        eps = 1e-8
        max_iter = 5000

        beta_start = start_vector.get_vector((p, 1))

        mus = [5e-0, 5e-2, 5e-4, 5e-6, 5e-8]
        fista = explicit.FISTA(eps=eps, max_iter=max_iter / len(mus))

        beta_parsimony = beta_start
        for mu in mus:
#            function = functions.RR_L1_GL(X, y, k, l, g, A=A, mu=mu,
#                                          penalty_start=0)

            function = CombinedFunction()
            function.add_function(functions.losses.LinearRegression(X, y,
                                                               mean=False))
            function.add_penalty(functions.penalties.L2(l=k))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            function.add_prox(functions.penalties.L1(l=l))

            beta_parsimony = fista.run(function, beta_parsimony)

        mse = (np.linalg.norm(beta_parsimony - beta_star) ** 2.0) / p
        assert mse < 1e-4

        f_parsimony = function.f(beta_parsimony)
        f_star = function.f(beta_star)
        assert abs(f_parsimony - f_star) < 0.005

if __name__ == "__main__":
    unittest.main()