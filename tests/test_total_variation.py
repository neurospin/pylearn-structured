# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 16:17:15 2014

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import numpy as np

from tests import TestCase


class TestTotalVariation(TestCase):

    def test_nonsmooth(self):

        from parsimony.functions import CombinedFunction
        import parsimony.algorithms.proximal as proximal
        import parsimony.functions as functions
        import parsimony.functions.nesterov.tv as tv
        import parsimony.datasets.simulate.l1_l2_tv as l1_l2_tv
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
        fista = proximal.FISTA(eps=eps, max_iter=max_iter / len(mus))

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
        import parsimony.algorithms.proximal as proximal
        import parsimony.functions as functions
        import parsimony.functions.nesterov.tv as tv
        import parsimony.datasets.simulate.l1_l2_tvmu as l1_l2_tvmu
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
        fista = proximal.FISTA(eps=eps, max_iter=max_iter / len(mus))

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
        import parsimony.algorithms.proximal as proximal
        import parsimony.functions as functions
        import parsimony.functions.nesterov.tv as tv
        import parsimony.datasets.simulate.l1_l2_tv as l1_l2_tv
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
        fista = proximal.FISTA(eps=eps, max_iter=max_iter / len(mus))

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
            function.add_penalty(functions.penalties.L2Squared(l=k))
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
        import parsimony.algorithms.proximal as proximal
        import parsimony.functions as functions
        import parsimony.functions.nesterov.tv as tv
        import parsimony.datasets.simulate.l1_l2_tvmu as l1_l2_tvmu
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
        fista = proximal.FISTA(eps=eps, max_iter=max_iter / len(mus))

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
            function.add_penalty(functions.penalties.L2Squared(l=k))
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

    def test_tvhelper_A_from_mask(self):
        import parsimony.functions.nesterov.tv as tv
        ## Simple mask with offset
        shape = (5, 4)
        mask = np.zeros(shape)
        mask[1:(shape[0] - 1), 0:(shape[1] - 1)] = 1
        Ax_ = np.matrix(
        [[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [ 0, -1, 0, 0, 1, 0, 0, 0, 0, 0],
         [ 0, 0, -1, 0, 0, 1, 0, 0, 0, 0],
         [ 0, 0, 0, -1, 0, 0, 1, 0, 0, 0],
         [ 0, 0, 0, 0, -1, 0, 0, 1, 0, 0],
         [ 0, 0, 0, 0, 0, -1, 0, 0, 1, 0],
         [ 0, 0, 0, 0, 0, 0, -1, 0, 0, 1],
         [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        Ay_ = np.matrix(
        [[ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [ 0, -1, 1, 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, -1, 1, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, -1, 1, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, -1, 1, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0, -1, 1, 0],
         [ 0, 0, 0, 0, 0, 0, 0, 0, -1, 1],
         [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        A,_ = tv.A_from_mask(mask, offset=1)
        Ax, Ay, Az = A
        assert np.all(Ax.todense() == Ax_)
        assert np.all(Ay.todense() == Ay_)
        assert np.sum(Az.todense() == 0)

        ############################################################################
        ## GROUP TV
        shape = (6,4)
        mask = np.zeros(shape,dtype=int)
        mask[:3,:3] = 1
        mask[3:6,1:4] = 2
        Ax_ = np.matrix(
        [[-1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [ 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0],
         [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 1],
         [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        Ay_ = np.matrix(
        [[-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [ 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
         [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1, 0],
         [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 1],
         [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        A,_ = tv.A_from_mask(mask)
        Ax, Ay, Az = A
        assert np.all(Ax.todense() == Ax_)
        assert np.all(Ay.todense() == Ay_)
        assert np.sum(Az.todense() == 0)

if __name__ == "__main__":
    import unittest
    unittest.main()
