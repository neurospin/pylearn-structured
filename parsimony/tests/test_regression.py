# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 09:21:23 2014

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import unittest
from nose.tools import assert_less, assert_equal, assert_almost_equal

import numpy as np

from tests import TestCase
import parsimony.utils.consts as consts


class TestLinearRegression():#TestCase):

    def test_linear_regression_overdetermined(self):

        from parsimony.functions.losses import LinearRegression
        import parsimony.algorithms.explicit as explicit
        import parsimony.start_vectors as start_vectors

        np.random.seed(42)

        n, p = 100, 50

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        X = np.random.multivariate_normal(mean, Sigma, n)

        start_vector = start_vectors.RandomStartVector(normalise=True)
        beta_star = start_vector.get_vector((p, 1))

        y = np.dot(X, beta_star)

        eps = 1e-8
        max_iter = 150
        gd = explicit.GradientDescent(eps=eps, max_iter=max_iter)
        linear_regression = LinearRegression(X, y, mean=False)
        beta_start = start_vector.get_vector((p, 1))

        beta_parsimony = gd.run(linear_regression, beta_start)

        mse = np.linalg.norm(beta_parsimony - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_parsimony = linear_regression.f(beta_parsimony)
        f_star = linear_regression.f(beta_star)
        if abs(f_star) > consts.TOLERANCE:
            err = abs(f_parsimony - f_star) / f_star
        else:
            err = abs(f_parsimony - f_star)
#        print "err:", err
        assert_less(err, 1e-6, "The found regression vector does not give " \
                               "the correct function value.")

    def test_linear_regression_underdetermined(self):

        from parsimony.functions.losses import LinearRegression
        import parsimony.algorithms.explicit as explicit
        import parsimony.start_vectors as start_vectors

        np.random.seed(42)

        n, p = 60, 90

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        X = np.random.multivariate_normal(mean, Sigma, n)

        start_vector = start_vectors.RandomStartVector(normalise=True)
        beta_star = start_vector.get_vector((p, 1))

        y = np.dot(X, beta_star)

        eps = 1e-8
        max_iter = 300
        gd = explicit.GradientDescent(eps=eps, max_iter=max_iter)
        linear_regression = LinearRegression(X, y, mean=False)
        beta_start = start_vector.get_vector((p, 1))

        beta_parsimony = gd.run(linear_regression, beta_start)

        mse = np.linalg.norm(beta_parsimony - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 5e-1, "The found regression vector is not correct.")

        f_parsimony = linear_regression.f(beta_parsimony)
        f_star = linear_regression.f(beta_star)
        if abs(f_star) > consts.TOLERANCE:
            err = abs(f_parsimony - f_star) / f_star
        else:
            err = abs(f_parsimony - f_star)
#        print "err:", err
        assert_less(err, 1e-4, "The found regression vector does not give " \
                               "the correct function value.")

    def test_linear_regression_determined(self):

        from parsimony.functions.losses import LinearRegression
        import parsimony.algorithms.explicit as explicit
        import parsimony.start_vectors as start_vectors

        np.random.seed(42)

        n, p = 50, 50

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        X = np.random.multivariate_normal(mean, Sigma, n)

        start_vector = start_vectors.RandomStartVector(normalise=True)
        beta_star = start_vector.get_vector((p, 1))

        y = np.dot(X, beta_star)

        eps = 1e-8
        max_iter = 19000
        gd = explicit.GradientDescent(eps=eps, max_iter=max_iter)
        linear_regression = LinearRegression(X, y, mean=False)
        beta_start = start_vector.get_vector((p, 1))

        beta_parsimony = gd.run(linear_regression, beta_start)

        mse = np.linalg.norm(beta_parsimony - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-2, "The found regression vector is not correct.")

        f_parsimony = linear_regression.f(beta_parsimony)
        f_star = linear_regression.f(beta_star)
        if abs(f_star) > consts.TOLERANCE:
            err = abs(f_parsimony - f_star) / f_star
        else:
            err = abs(f_parsimony - f_star)
#        print "err:", err
        assert_less(err, 1e-5, "The found regression vector does not give " \
                               "the correct function value.")

    def test_linear_regression_l1(self):

        from parsimony.functions.losses import LinearRegression
        from parsimony.functions.penalties import L1
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulated.l1_l2_gl as l1_l2_gl
        import parsimony.algorithms.explicit as explicit
        import parsimony.start_vectors as start_vectors

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        n, p = 60, 90

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector((p, 1))
        beta[beta < 0.1] = 0.0

        l = 0.618
        k = 0.0
        g = 0.0

        A = np.eye(p)
        A = [A, A, A]
        snr = 100.0
        X, y, beta_star = l1_l2_gl.load(l, k, g, beta, M, e, A, snr=snr)

        eps = 1e-8
        max_iter = 9000
        fista = explicit.FISTA(eps=eps, max_iter=max_iter)
        linear_regression = LinearRegression(X, y, mean=False)
        l1 = L1(l=l)
        function = CombinedFunction()
        function.add_function(linear_regression)
        function.add_prox(l1)

        beta_start = start_vector.get_vector((p, 1))

        beta_parsimony = fista.run(function, beta_start)

        mse = np.linalg.norm(beta_parsimony - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_parsimony = function.f(beta_parsimony)
        f_star = function.f(beta_star)
        err = abs(f_parsimony - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-4, "The found regression vector does not give " \
                               "the correct function value.")

    def test_linear_regression_l2(self):

        from parsimony.functions.losses import LinearRegression
        from parsimony.functions.losses import RidgeRegression
        from parsimony.functions.penalties import L2
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulated.l1_l2_gl as l1_l2_gl
        import parsimony.algorithms.explicit as explicit
        import parsimony.start_vectors as start_vectors

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        n, p = 60, 90

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector((p, 1))
#        beta[beta < 0.1] = 0.0

        l = 0.0
        k = 0.618
        g = 0.0

        A = np.eye(p)
        A = [A, A, A]
        snr = 100.0
        X, y, beta_star = l1_l2_gl.load(l, k, g, beta, M, e, A, snr=snr)

        eps = 1e-8
        max_iter = 6000

        fista = explicit.FISTA(eps=eps, max_iter=max_iter)
        beta_start = start_vector.get_vector((p, 1))

        function = CombinedFunction()
        function.add_function(LinearRegression(X, y, mean=False))
        function.add_penalty(L2(k))
        beta_penalty = fista.run(function, beta_start)

        mse = np.linalg.norm(beta_penalty - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_parsimony = function.f(beta_penalty)
        err = abs(f_parsimony - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-4, "The found regression vector does not give " \
                               "the correct function value.")

        function = CombinedFunction()
        function.add_function(LinearRegression(X, y, mean=False))
        function.add_prox(L2(k))
        beta_prox = fista.run(function, beta_start)

        mse = np.linalg.norm(beta_prox - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_prox = function.f(beta_prox)
        err = abs(f_prox - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-4, "The found regression vector does not give " \
                               "the correct function value.")

        function = CombinedFunction()
        function.add_function(RidgeRegression(X, y, k))
        beta_rr = fista.run(function, beta_start)

        mse = np.linalg.norm(beta_rr - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_rr = function.f(beta_rr)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-4, "The found regression vector does not give " \
                               "the correct function value.")

    def test_linear_regression_tv(self):

        from parsimony.functions.losses import LinearRegression
#        from parsimony.functions.losses import RidgeRegression
#        from parsimony.functions.penalties import L1
#        from parsimony.functions.penalties import L2
        import parsimony.functions.nesterov.tv as tv
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulated.l1_l2_tv as l1_l2_tv
        import parsimony.datasets.simulated.l1_l2_tvmu as l1_l2_tvmu
        import parsimony.algorithms.explicit as explicit
        import parsimony.start_vectors as start_vectors

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        px = 4
        py = 4
        pz = 4
        shape = (pz, py, px)
        n, p = 50, np.prod(shape)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector((p, 1))
        beta = np.sort(beta, axis=0)
#        beta[beta < 0.1] = 0.0

        l = 0.0
        k = 0.0
        g = 1.618

        A, _ = tv.A_from_shape(shape)
        snr = 20.0
        eps = 1e-8
        max_iter = 2500

        X, y, beta_star = l1_l2_tv.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = explicit.FISTA(eps=eps, max_iter=max_iter / len(mus))
        beta_start = start_vector.get_vector((p, 1))

        beta_nonsmooth = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            beta_nonsmooth = fista.run(function, beta_nonsmooth)

        mse = np.linalg.norm(beta_nonsmooth - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_nonsmooth = function.f(beta_nonsmooth)
        err = abs(f_nonsmooth - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-8, "The found regression vector does not give " \
                               "the correct function value.")

        mu_min = mus[-1]
        X, y, beta_star = l1_l2_tvmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                          A=A, mu=mu_min, snr=snr)
        beta_smooth = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            beta_smooth = fista.run(function, beta_smooth)

        mse = np.linalg.norm(beta_smooth - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_smooth = function.f(beta_smooth)
        err = abs(f_smooth - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-8, "The found regression vector does not give " \
                               "the correct function value.")

    def test_linear_regression_gl(self):

        from parsimony.functions.losses import LinearRegression
#        from parsimony.functions.losses import RidgeRegression
#        from parsimony.functions.penalties import L1
#        from parsimony.functions.penalties import L2
        import parsimony.functions.nesterov.gl as gl
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulated.l1_l2_gl as l1_l2_gl
        import parsimony.datasets.simulated.l1_l2_glmu as l1_l2_glmu
        import parsimony.algorithms.explicit as explicit
        import parsimony.start_vectors as start_vectors

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        n, p = 60, 90
        groups = [range(0, 2 * p / 3), range(p / 3, p)]
        weights = [1.5, 0.5]

        A = gl.A_from_groups(p, groups=groups, weights=weights)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector((p, 1))
        beta = np.sort(beta, axis=0)
#        beta[beta < 0.1] = 0.0

        l = 0.0
        k = 0.0
        g = 1.618

        snr = 20.0
        eps = 1e-8
        max_iter = 3000

        X, y, beta_star = l1_l2_gl.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = explicit.FISTA(eps=eps, max_iter=max_iter / len(mus))
        beta_start = start_vector.get_vector((p, 1))

        beta_nonsmooth = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            beta_nonsmooth = fista.run(function, beta_nonsmooth)

        mse = np.linalg.norm(beta_nonsmooth - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_parsimony = function.f(beta_nonsmooth)
        err = abs(f_parsimony - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-8, "The found regression vector does not give " \
                               "the correct function value.")

        mu_min = mus[-1]
        X, y, beta_star = l1_l2_glmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                          A=A, mu=mu_min, snr=snr)
        beta_smooth = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            beta_smooth = fista.run(function, beta_smooth)

        mse = np.linalg.norm(beta_smooth - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_rr = function.f(beta_smooth)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-8, "The found regression vector does not give " \
                               "the correct function value.")

    def test_linear_regression_l1_l2(self):

        from parsimony.functions.losses import LinearRegression
        from parsimony.functions.losses import RidgeRegression
        from parsimony.functions.penalties import L1
        from parsimony.functions.penalties import L2
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulated.l1_l2_gl as l1_l2_gl
        import parsimony.algorithms.explicit as explicit
        import parsimony.start_vectors as start_vectors

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        n, p = 60, 90

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector((p, 1))
        beta[beta < 0.1] = 0.0

        l = 0.618
        k = 1.0 - l
        g = 0.0

        A = np.eye(p)
        A = [A, A, A]
        snr = 100.0
        X, y, beta_star = l1_l2_gl.load(l, k, g, beta, M, e, A, snr=snr)

        eps = 1e-8
        max_iter = 600

        fista = explicit.FISTA(eps=eps, max_iter=max_iter)
        beta_start = start_vector.get_vector((p, 1))

        function = CombinedFunction()
        function.add_function(LinearRegression(X, y, mean=False))
        function.add_penalty(L2(k))
        function.add_prox(L1(l))
        beta_penalty = fista.run(function, beta_start)

        mse = np.linalg.norm(beta_penalty - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_parsimony = function.f(beta_penalty)
        err = abs(f_parsimony - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-5, "The found regression vector does not give " \
                               "the correct function value.")

        function = CombinedFunction()
        function.add_function(RidgeRegression(X, y, k))
        function.add_prox(L1(l))
        beta_rr = fista.run(function, beta_start)

        mse = np.linalg.norm(beta_rr - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_rr = function.f(beta_rr)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-5, "The found regression vector does not give " \
                               "the correct function value.")

    def test_linear_regression_l1_tv(self):

        from parsimony.functions.losses import LinearRegression
#        from parsimony.functions.losses import RidgeRegression
        from parsimony.functions.penalties import L1
#        from parsimony.functions.penalties import L2
        import parsimony.functions.nesterov.tv as tv
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulated.l1_l2_tv as l1_l2_tv
        import parsimony.datasets.simulated.l1_l2_tvmu as l1_l2_tvmu
        import parsimony.algorithms.explicit as explicit
        import parsimony.start_vectors as start_vectors

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        px = 4
        py = 4
        pz = 4
        shape = (pz, py, px)
        n, p = 50, np.prod(shape)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector((p, 1))
        beta = np.sort(beta, axis=0)
        beta[beta < 0.1] = 0.0

        l = 0.618
        k = 0.0
        g = 1.618

        A, _ = tv.A_from_shape(shape)
        snr = 20.0
        eps = 1e-8
        max_iter = 1800

        X, y, beta_star = l1_l2_tv.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = explicit.FISTA(eps=eps, max_iter=max_iter / len(mus))
        beta_start = start_vector.get_vector((p, 1))

        beta_nonsmooth_penalty = beta_start
        function = None
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            function.add_prox(L1(l))
            beta_nonsmooth_penalty = \
                    fista.run(function, beta_nonsmooth_penalty)

        mse = np.linalg.norm(beta_nonsmooth_penalty - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_nonsmooth_star = function.f(beta_star)
        f_nonsmooth_penalty = function.f(beta_nonsmooth_penalty)
        err = abs(f_nonsmooth_penalty - f_nonsmooth_star) / f_nonsmooth_star
#        print "err:", err
        assert_less(err, 1e-4, "The found regression vector does not give " \
                               "the correct function value.")

        mu_min = mus[-1]
        X, y, beta_star = l1_l2_tvmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                          A=A, mu=mu_min, snr=snr)
        beta_smooth_penalty = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            function.add_prox(L1(l))
            beta_smooth_penalty = \
                    fista.run(function, beta_smooth_penalty)

        mse = np.linalg.norm(beta_smooth_penalty - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_smooth_star = function.f(beta_star)
        f_smooth_penalty = function.f(beta_smooth_penalty)
        err = abs(f_smooth_penalty - f_smooth_star) / f_smooth_star
#        print "err:", err
        assert_less(err, 1e-4, "The found regression vector does not give " \
                               "the correct function value.")

    def test_linear_regression_l1_gl(self):

        from parsimony.functions.losses import LinearRegression
#        from parsimony.functions.losses import RidgeRegression
        from parsimony.functions.penalties import L1
#        from parsimony.functions.penalties import L2
        import parsimony.functions.nesterov.gl as gl
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulated.l1_l2_gl as l1_l2_gl
        import parsimony.datasets.simulated.l1_l2_glmu as l1_l2_glmu
        import parsimony.algorithms.explicit as explicit
        import parsimony.start_vectors as start_vectors

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        n, p = 60, 90
        groups = [range(0, 2 * p / 3), range(p / 3, p)]
        weights = [1.5, 0.5]

        A = gl.A_from_groups(p, groups=groups, weights=weights)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector((p, 1))
        beta = np.sort(beta, axis=0)
        beta[beta < 0.1] = 0.0

        l = 0.618
        k = 0.0
        g = 1.618

        snr = 20.0
        eps = 1e-8
        max_iter = 1600

        X, y, beta_star = l1_l2_gl.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = explicit.FISTA(eps=eps, max_iter=max_iter / len(mus))
        beta_start = start_vector.get_vector((p, 1))

        beta_nonsmooth = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            function.add_prox(L1(l))
            beta_nonsmooth = fista.run(function, beta_nonsmooth)

        mse = np.linalg.norm(beta_nonsmooth - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_parsimony = function.f(beta_nonsmooth)
        err = abs(f_parsimony - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-4, "The found regression vector does not give " \
                               "the correct function value.")

        mu_min = mus[-1]
        X, y, beta_star = l1_l2_glmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                          A=A, mu=mu_min, snr=snr)
        beta_smooth = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            function.add_prox(L1(l))
            beta_smooth = fista.run(function, beta_smooth)

        mse = np.linalg.norm(beta_smooth - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_rr = function.f(beta_smooth)
        err = abs(f_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-4, "The found regression vector does not give " \
                               "the correct function value.")

    def test_linear_regression_l2_tv(self):

        from parsimony.functions.losses import LinearRegression
        from parsimony.functions.losses import RidgeRegression
#        from parsimony.functions.penalties import L1
        from parsimony.functions.penalties import L2
        import parsimony.functions.nesterov.tv as tv
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulated.l1_l2_tv as l1_l2_tv
        import parsimony.datasets.simulated.l1_l2_tvmu as l1_l2_tvmu
        import parsimony.algorithms.explicit as explicit
        import parsimony.start_vectors as start_vectors

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        px = 4
        py = 4
        pz = 4
        shape = (pz, py, px)
        n, p = 50, np.prod(shape)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector((p, 1))
        beta = np.sort(beta, axis=0)
#        beta[beta < 0.1] = 0.0

        l = 0.0
        k = 0.618
        g = 1.618

        A, _ = tv.A_from_shape(shape)
        snr = 20.0
        eps = 1e-8
        max_iter = 900

        X, y, beta_star = l1_l2_tv.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = explicit.FISTA(eps=eps, max_iter=max_iter / len(mus))
        beta_start = start_vector.get_vector((p, 1))

        beta_nonsmooth_penalty = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            function.add_penalty(L2(k))
            beta_nonsmooth_penalty = \
                    fista.run(function, beta_nonsmooth_penalty)

        mse = np.linalg.norm(beta_nonsmooth_penalty - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_nonsmooth_penalty = function.f(beta_nonsmooth_penalty)
        err = abs(f_nonsmooth_penalty - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-7, "The found regression vector does not give " \
                               "the correct function value.")

        beta_nonsmooth_prox = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            function.add_prox(L2(k))
            beta_nonsmooth_prox = fista.run(function, beta_nonsmooth_prox)

        mse = np.linalg.norm(beta_nonsmooth_prox - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_nonsmooth_prox = function.f(beta_nonsmooth_prox)
        err = abs(f_nonsmooth_prox - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-7, "The found regression vector does not give " \
                               "the correct function value.")

        beta_nonsmooth_rr = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(RidgeRegression(X, y, k))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            beta_nonsmooth_rr = fista.run(function, beta_nonsmooth_rr)

        mse = np.linalg.norm(beta_nonsmooth_rr - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_nonsmooth_rr = function.f(beta_nonsmooth_rr)
        err = abs(f_nonsmooth_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-7, "The found regression vector does not give " \
                               "the correct function value.")

        mu_min = mus[-1]
        X, y, beta_star = l1_l2_tvmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                          A=A, mu=mu_min, snr=snr)
        beta_smooth_penalty = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            function.add_penalty(L2(k))
            beta_smooth_penalty = \
                    fista.run(function, beta_smooth_penalty)

        mse = np.linalg.norm(beta_smooth_penalty - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_smooth_penalty = function.f(beta_smooth_penalty)
        err = abs(f_smooth_penalty - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-7, "The found regression vector does not give " \
                               "the correct function value.")

        beta_smooth_prox = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            function.add_prox(L2(k))
            beta_smooth_prox = fista.run(function, beta_smooth_prox)

        mse = np.linalg.norm(beta_smooth_prox - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_smooth_prox = function.f(beta_smooth_prox)
        err = abs(f_smooth_prox - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-7, "The found regression vector does not give " \
                               "the correct function value.")

        beta_smooth_rr = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(RidgeRegression(X, y, k))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            beta_smooth_rr = fista.run(function, beta_smooth_rr)

        mse = np.linalg.norm(beta_smooth_rr - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_smooth_rr = function.f(beta_smooth_rr)
        err = abs(f_smooth_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-7, "The found regression vector does not give " \
                               "the correct function value.")

    def test_linear_regression_l2_gl(self):

        from parsimony.functions.losses import LinearRegression
        from parsimony.functions.losses import RidgeRegression
#        from parsimony.functions.penalties import L1
        from parsimony.functions.penalties import L2
        import parsimony.functions.nesterov.gl as gl
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulated.l1_l2_gl as l1_l2_gl
        import parsimony.datasets.simulated.l1_l2_glmu as l1_l2_glmu
        import parsimony.algorithms.explicit as explicit
        import parsimony.start_vectors as start_vectors

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        n, p = 60, 90
        groups = [range(0, 2 * p / 3), range(p / 3, p)]
        weights = [1.5, 0.5]

        A = gl.A_from_groups(p, groups=groups, weights=weights)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector((p, 1))
        beta = np.sort(beta, axis=0)
#        beta[beta < 0.1] = 0.0

        l = 0.0
        k = 0.618
        g = 1.618

        snr = 20.0
        eps = 1e-8
        max_iter = 950

        X, y, beta_star = l1_l2_gl.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = explicit.FISTA(eps=eps, max_iter=max_iter / len(mus))
        beta_start = start_vector.get_vector((p, 1))

        beta_nonsmooth_penalty = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            function.add_penalty(L2(k))
            beta_nonsmooth_penalty = fista.run(function,
                                               beta_nonsmooth_penalty)

        mse = np.linalg.norm(beta_nonsmooth_penalty - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_nonsmooth_penalty = function.f(beta_nonsmooth_penalty)
        err = abs(f_nonsmooth_penalty - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-6, "The found regression vector does not give " \
                               "the correct function value.")

        beta_nonsmooth_prox = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            function.add_prox(L2(k))
            beta_nonsmooth_prox = fista.run(function, beta_nonsmooth_prox)

        mse = np.linalg.norm(beta_nonsmooth_prox - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_nonsmooth_prox = function.f(beta_nonsmooth_prox)
        err = abs(f_nonsmooth_prox - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-6, "The found regression vector does not give " \
                               "the correct function value.")

        beta_nonsmooth_rr = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(RidgeRegression(X, y, k))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            beta_nonsmooth_rr = fista.run(function, beta_nonsmooth_rr)

        mse = np.linalg.norm(beta_nonsmooth_rr - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_nonsmooth_rr = function.f(beta_nonsmooth_rr)
        err = abs(f_nonsmooth_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-6, "The found regression vector does not give " \
                               "the correct function value.")

        mu_min = mus[-1]
        X, y, beta_star = l1_l2_glmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                          A=A, mu=mu_min, snr=snr)

        beta_smooth_penalty = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            function.add_penalty(L2(k))
            beta_smooth_penalty = fista.run(function, beta_smooth_penalty)

        mse = np.linalg.norm(beta_smooth_penalty - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_smooth_penalty = function.f(beta_smooth_penalty)
        err = abs(f_smooth_penalty - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-6, "The found regression vector does not give " \
                               "the correct function value.")

        beta_smooth_prox = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            function.add_prox(L2(k))
            beta_smooth_prox = fista.run(function, beta_smooth_prox)

        mse = np.linalg.norm(beta_smooth_prox - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_smooth_prox = function.f(beta_smooth_prox)
        err = abs(f_smooth_prox - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-6, "The found regression vector does not give " \
                               "the correct function value.")

        beta_smooth_rr = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(RidgeRegression(X, y, k))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            beta_smooth_rr = fista.run(function, beta_smooth_rr)

        mse = np.linalg.norm(beta_smooth_rr - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_smooth_rr = function.f(beta_smooth_rr)
        err = abs(f_smooth_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-6, "The found regression vector does not give " \
                               "the correct function value.")

    def test_linear_regression_l1_l2_tv(self):

        from parsimony.functions.losses import LinearRegression
        from parsimony.functions.losses import RidgeRegression
        from parsimony.functions.penalties import L1
        from parsimony.functions.penalties import L2
        import parsimony.functions.nesterov.tv as tv
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulated.l1_l2_tv as l1_l2_tv
        import parsimony.datasets.simulated.l1_l2_tvmu as l1_l2_tvmu
        import parsimony.algorithms.explicit as explicit
        import parsimony.start_vectors as start_vectors

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        px = 4
        py = 4
        pz = 4
        shape = (pz, py, px)
        n, p = 50, np.prod(shape)

        alpha = 1.0
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector((p, 1))
        beta = np.sort(beta, axis=0)
        beta[beta < 0.1] = 0.0

        l = 0.618
        k = 1.0 - l
        g = 1.618

        A, _ = tv.A_from_shape(shape)
        snr = 100.0
        eps = 1e-8
        max_iter = 600

        X, y, beta_star = l1_l2_tv.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = explicit.FISTA(eps=eps, max_iter=max_iter / len(mus))
        beta_start = start_vector.get_vector((p, 1))

        beta_nonsmooth_penalty = beta_start
        function = None
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            function.add_penalty(L2(k))
            function.add_prox(L1(l))
            beta_nonsmooth_penalty = \
                    fista.run(function, beta_nonsmooth_penalty)

        mse = np.linalg.norm(beta_nonsmooth_penalty - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_nonsmooth_star = function.f(beta_star)
        f_nonsmooth_penalty = function.f(beta_nonsmooth_penalty)
        err = abs(f_nonsmooth_penalty - f_nonsmooth_star) / f_nonsmooth_star
#        print "err:", err
        assert_less(err, 1e-4, "The found regression vector does not give " \
                               "the correct function value.")

        beta_nonsmooth_rr = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(RidgeRegression(X, y, k))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            function.add_prox(L1(l))
            beta_nonsmooth_rr = fista.run(function, beta_nonsmooth_rr)

        mse = np.linalg.norm(beta_nonsmooth_rr - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_nonsmooth_star = function.f(beta_star)
        f_nonsmooth_rr = function.f(beta_nonsmooth_rr)
        err = abs(f_nonsmooth_rr - f_nonsmooth_star) / f_nonsmooth_star
#        print "err:", err
        assert_less(err, 1e-4, "The found regression vector does not give " \
                               "the correct function value.")

        mu_min = mus[-1]
        X, y, beta_star = l1_l2_tvmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                          A=A, mu=mu_min, snr=snr)
        beta_smooth_penalty = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            function.add_penalty(L2(k))
            function.add_prox(L1(l))
            beta_smooth_penalty = \
                    fista.run(function, beta_smooth_penalty)

        mse = np.linalg.norm(beta_smooth_penalty - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_smooth_star = function.f(beta_star)
        f_smooth_penalty = function.f(beta_smooth_penalty)
        err = abs(f_smooth_penalty - f_smooth_star) / f_smooth_star
#        print "err:", err
        assert_less(err, 1e-4, "The found regression vector does not give " \
                               "the correct function value.")

        beta_smooth_rr = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(RidgeRegression(X, y, k))
            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
                                                   penalty_start=0))
            function.add_prox(L1(l))
            beta_smooth_rr = fista.run(function, beta_smooth_rr)

        mse = np.linalg.norm(beta_smooth_rr - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_smooth_star = function.f(beta_star)
        f_smooth_rr = function.f(beta_smooth_rr)
        err = abs(f_smooth_rr - f_smooth_star) / f_smooth_star
#        print "err:", err
        assert_less(err, 1e-4, "The found regression vector does not give " \
                               "the correct function value.")

    def test_linear_regression_l1_l2_gl(self):

        from parsimony.functions.losses import LinearRegression
        from parsimony.functions.losses import RidgeRegression
        from parsimony.functions.penalties import L1
        from parsimony.functions.penalties import L2
        import parsimony.functions.nesterov.gl as gl
        from parsimony.functions import CombinedFunction
        import parsimony.datasets.simulated.l1_l2_gl as l1_l2_gl
        import parsimony.datasets.simulated.l1_l2_glmu as l1_l2_glmu
        import parsimony.algorithms.explicit as explicit
        import parsimony.start_vectors as start_vectors

        start_vector = start_vectors.RandomStartVector(normalise=True)

        np.random.seed(42)

        n, p = 60, 90
        groups = [range(0, 2 * p / 3), range(p / 3, p)]
        weights = [1.5, 0.5]

        A = gl.A_from_groups(p, groups=groups, weights=weights)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)

        beta = start_vector.get_vector((p, 1))
        beta = np.sort(beta, axis=0)
        beta[beta < 0.1] = 0.0

        l = 0.618
        k = 1.0 - l
        g = 1.618

        snr = 20.0
        eps = 1e-8
        max_iter = 750

        X, y, beta_star = l1_l2_gl.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        mus = [5e-2, 5e-4, 5e-6, 5e-8]
        fista = explicit.FISTA(eps=eps, max_iter=max_iter / len(mus))
        beta_start = start_vector.get_vector((p, 1))

        beta_nonsmooth_penalty = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            function.add_penalty(L2(k))
            function.add_prox(L1(l))
            beta_nonsmooth_penalty = fista.run(function,
                                               beta_nonsmooth_penalty)

        mse = np.linalg.norm(beta_nonsmooth_penalty - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_nonsmooth_penalty = function.f(beta_nonsmooth_penalty)
        err = abs(f_nonsmooth_penalty - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-6, "The found regression vector does not give " \
                               "the correct function value.")

        beta_nonsmooth_rr = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(RidgeRegression(X, y, k))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            function.add_prox(L1(l))
            beta_nonsmooth_rr = fista.run(function, beta_nonsmooth_rr)

        mse = np.linalg.norm(beta_nonsmooth_rr - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_nonsmooth_rr = function.f(beta_nonsmooth_rr)
        err = abs(f_nonsmooth_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-6, "The found regression vector does not give " \
                               "the correct function value.")

        mu_min = mus[-1]
        X, y, beta_star = l1_l2_glmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                          A=A, mu=mu_min, snr=snr)
        beta_smooth_penalty = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(LinearRegression(X, y, mean=False))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            function.add_penalty(L2(k))
            function.add_prox(L1(l))
            beta_smooth_penalty = fista.run(function, beta_smooth_penalty)

        mse = np.linalg.norm(beta_smooth_penalty - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_smooth_penalty = function.f(beta_smooth_penalty)
        err = abs(f_smooth_penalty - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-6, "The found regression vector does not give " \
                               "the correct function value.")

        beta_smooth_rr = beta_start
        for mu in mus:
            function = CombinedFunction()
            function.add_function(RidgeRegression(X, y, k))
            function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                      penalty_start=0))
            function.add_prox(L1(l))
            beta_smooth_rr = fista.run(function, beta_smooth_rr)

        mse = np.linalg.norm(beta_smooth_rr - beta_star) \
                / np.linalg.norm(beta_star)
#        print "mse:", mse
        assert_less(mse, 1e-3, "The found regression vector is not correct.")

        f_star = function.f(beta_star)
        f_smooth_rr = function.f(beta_smooth_rr)
        err = abs(f_smooth_rr - f_star) / f_star
#        print "err:", err
        assert_less(err, 1e-6, "The found regression vector does not give " \
                               "the correct function value.")

    def test_estimators(self):

        import numpy as np
        import parsimony.estimators as estimators
        import parsimony.algorithms.explicit as explicit
        import parsimony.functions.nesterov.tv as tv
        import parsimony.datasets.simulated.l1_l2_tv as l1_l2_tv

        np.random.seed(42)

        shape = (4, 4, 4)
        A, n_compacts = tv.A_from_shape(shape)

        n, p = 64, np.prod(shape)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)
        beta = np.random.rand(p, 1)
        snr = 100.0

        l = 0.0  # L1 coefficient
        k = 0.1  # Ridge coefficient
        g = 0.0  # TV coefficient
        np.random.seed(42)
        X, y, beta_star = l1_l2_tv.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        lr = estimators.LinearRegression_L1_L2_TV(l, k, g, A,
                                       algorithm=explicit.FISTA(max_iter=1000))
        lr.fit(X, y)
        score = lr.score(X, y)
        print "score:", score
        assert_almost_equal(score, 1.299125,
                            msg="The found regression vector does not give " \
                                "a low enough score value.",
                            places=5)

        n, p = 50, np.prod(shape)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)
        beta = np.random.rand(p, 1)
        snr = 100.0

        l = 0.0  # L1 coefficient
        k = 0.1  # Ridge coefficient
        g = 0.0  # TV coefficient
        np.random.seed(42)
        X, y, beta_star = l1_l2_tv.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        lr = estimators.LinearRegression_L1_L2_TV(l, k, g, A,
                                       algorithm=explicit.FISTA(max_iter=1000))
        lr.fit(X, y)
        score = lr.score(X, y)
        print "score:", score
        assert_almost_equal(score, 0.969570,
                            msg="The found regression vector does not give " \
                                "a low enough score value.",
                            places=5)

        n, p = 100, np.prod(shape)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)
        beta = np.random.rand(p, 1)
        snr = 100.0

        l = 0.0  # L1 coefficient
        k = 0.1  # Ridge coefficient
        g = 0.0  # TV coefficient
        np.random.seed(42)
        X, y, beta_star = l1_l2_tv.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        lr = estimators.LinearRegression_L1_L2_TV(l, k, g, A,
                                       algorithm=explicit.FISTA(max_iter=1000))
        lr.fit(X, y)
        score = lr.score(X, y)
        print "score:", score
        assert_almost_equal(score, 1.154561,
                            msg="The found regression vector does not give " \
                                "a low enough score value.",
                            places=5)

        n, p = 100, np.prod(shape)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        M = np.random.multivariate_normal(mean, Sigma, n)
        e = np.random.randn(n, 1)
        beta = np.random.rand(p, 1)
        beta = np.sort(beta, axis=0)
        beta[:10, :] = 0.0
        snr = 100.0

        l = 0.618  # L1 coefficient
        k = 1.0 - l  # Ridge coefficient
        g = 2.718  # TV coefficient
        np.random.seed(42)
        X, y, beta_star = l1_l2_tv.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        l = 0.0
        k = 0.0
        g = 0.0
        lr = estimators.LinearRegression_L1_L2_TV(l, k, g, A,
                                       algorithm=explicit.FISTA(max_iter=1000))
        lr.fit(X, y)
        score = lr.score(X, y)
        print "score:", score
        assert_almost_equal(score, 1.019992,
                            msg="The found regression vector does not give " \
                                "a low enough score value.",
                            places=5)

        l = 0.618
        k = 0.0
        g = 0.0
        lr = estimators.LinearRegression_L1_L2_TV(l, k, g, A,
                                       algorithm=explicit.FISTA(max_iter=1000))
        lr.fit(X, y)
        score = lr.score(X, y)
        print "score:", score
        assert_almost_equal(score, 1.064312,
                            msg="The found regression vector does not give " \
                                "a low enough score value.",
                            places=5)

        l = 0.0
        k = 1.0 - 0.618
        g = 0.0
        lr = estimators.LinearRegression_L1_L2_TV(l, k, g, A,
                                       algorithm=explicit.FISTA(max_iter=1000))
        lr.fit(X, y)
        score = lr.score(X, y)
        print "score:", score
        assert_almost_equal(score, 1.024532,
                            msg="The found regression vector does not give " \
                                "a low enough score value.",
                            places=5)

        l = 0.0
        k = 0.0
        g = 2.718
        lr = estimators.LinearRegression_L1_L2_TV(l, k, g, A,
                                       algorithm=explicit.FISTA(max_iter=1000))
        lr.fit(X, y)
        score = lr.score(X, y)
        print "score:", score
        assert_almost_equal(score, 14.631501,
                            msg="The found regression vector does not give " \
                                "a low enough score value.",
                            places=5)

        l = 0.618
        k = 1.0 - l
        g = 0.0
        lr = estimators.LinearRegression_L1_L2_TV(l, k, g, A,
                                       algorithm=explicit.FISTA(max_iter=1000))
        lr.fit(X, y)
        score = lr.score(X, y)
        print "score:", score
        assert_almost_equal(score, 1.070105,
                            msg="The found regression vector does not give " \
                                "a low enough score value.",
                            places=5)

        l = 0.618
        k = 0.0
        g = 2.718
        lr = estimators.LinearRegression_L1_L2_TV(l, k, g, A,
                                       algorithm=explicit.FISTA(max_iter=1000))
        lr.fit(X, y)
        score = lr.score(X, y)
        print "score:", score
        assert_almost_equal(score, 14.458926,
                            msg="The found regression vector does not give " \
                                "a low enough score value.",
                            places=5)

        l = 0.0
        k = 1.0 - 0.618
        g = 2.718
        lr = estimators.LinearRegression_L1_L2_TV(l, k, g, A,
                                       algorithm=explicit.FISTA(max_iter=1000))
        lr.fit(X, y)
        score = lr.score(X, y)
        print "score:", score
        assert_almost_equal(score, 13.982838,
                            msg="The found regression vector does not give " \
                                "a low enough score value.",
                            places=5)

        l = 0.618
        k = 1.0 - l
        g = 2.718
        lr = estimators.LinearRegression_L1_L2_TV(l, k, g, A,
                                        algorithm=explicit.ISTA(max_iter=1000))
        lr.fit(X, y)
        score = lr.score(X, y)
        print "score:", score
        assert_almost_equal(score, 1041.254962,
                            msg="The found regression vector does not give " \
                                "the correct score value.",
                            places=5)

        l = 0.618
        k = 1.0 - l
        g = 2.718
        lr = estimators.LinearRegression_L1_L2_TV(l, k, g, A,
                                      algorithm=explicit.FISTA(max_iter=1000))
        lr.fit(X, y)
        score = lr.score(X, y)
        print "score:", score
        assert_almost_equal(score, 13.112947,
                            msg="The found regression vector does not give " \
                                "the correct score value.",
                            places=5)

        l = 0.618
        k = 1.0 - l
        g = 2.718
        rr = estimators.RidgeRegression_L1_TV(k, l, g, A,
                                        algorithm=explicit.ISTA(max_iter=1000))
        rr.fit(X, y)
        score = rr.score(X, y)
        print "score:", score
        assert_almost_equal(score, 7.106687,
                            msg="The found regression vector does not give " \
                                "the correct score value.",
                            places=5)

        l = 0.618
        k = 1.0 - l
        g = 2.718
        rr = estimators.RidgeRegression_L1_TV(k, l, g, A,
                                       algorithm=explicit.FISTA(max_iter=1000))
        rr.fit(X, y)
        score = rr.score(X, y)
        print "score:", score
        assert_almost_equal(score, 1.081411,
                            msg="The found regression vector does not give " \
                                "the correct score value.",
                            places=5)

        l = 0.618
        k = 1.0 - l
        g = 2.718
        rr = estimators.RidgeRegression_L1_TV(k, l, g, A,
            algorithm=explicit.DynamicCONESTA(continuations=10, max_iter=1000))
        rr.fit(X, y)
        score = rr.score(X, y)
        print "score:", score
        assert_almost_equal(score, 1.061931,
                            msg="The found regression vector does not give " \
                                "the correct score value.",
                            places=5)

        l = 0.618
        k = 1.0 - l
        g = 2.718
        rr = estimators.RidgeRegression_L1_TV(k, l, g, A,
             algorithm=explicit.StaticCONESTA(continuations=10, max_iter=1000))
        rr.fit(X, y)
        score = rr.score(X, y)
        print "score:", score
        assert_almost_equal(score, 1.062117,
                            msg="The found regression vector does not give " \
                                "the correct score value.",
                            places=5)

    def test_linear_regression_large(self):

        import parsimony.algorithms.explicit as explicit
        import parsimony.estimators as estimators
        import parsimony.functions.nesterov.tv as tv

        np.random.seed(42)

        px = 10
        py = 10
        pz = 10
        shape = (pz, py, px)
        n, p = 100, np.prod(shape)

        A, _ = tv.A_from_shape(shape)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        X = np.random.multivariate_normal(mean, Sigma, n)

        beta = np.random.rand(p, 1) * 2.0 - 1.0
        beta = np.sort(beta, axis=0)
        beta[np.abs(beta) < 0.1] = 0.0

        y = np.dot(X, beta)

        eps = 1e-8
        max_iter = 500

        k = 0.618
        l = 1.0 - k
        g = 1.618

        mu = None
        logreg_static = estimators.RidgeRegression_L1_TV(
                           k=k,
                           l=l,
                           g=g,
                           A=A,
                           mu=mu,
                           output=False,
                           algorithm=explicit.StaticCONESTA(eps=eps,
                                                            continuations=20,
                                                            max_iter=max_iter))
        logreg_static.fit(X, y)
        err = logreg_static.score(X, y)
#        print err
        assert_almost_equal(err, 0.025976,
                     msg="The found regression vector is not correct.",
                     places=5)

        mu = None
        logreg_dynamic = estimators.RidgeRegression_L1_TV(
                          k=k,
                          l=l,
                          g=g,
                          A=A,
                          mu=mu,
                          output=False,
                          algorithm=explicit.DynamicCONESTA(eps=eps,
                                                            continuations=20,
                                                            max_iter=max_iter))
        logreg_dynamic.fit(X, y)
        err = logreg_dynamic.score(X, y)
#        print err
        assert_almost_equal(err, 0.025976,
                     msg="The found regression vector is not correct.",
                     places=5)

        mu = 5e-4
        logreg_fista = estimators.RidgeRegression_L1_TV(
                          k=k,
                          l=l,
                          g=g,
                          A=A,
                          mu=mu,
                          output=False,
                          algorithm=explicit.FISTA(eps=eps,
                                                   max_iter=10000))
        logreg_fista.fit(X, y)
        err = logreg_fista.score(X, y)
#        print err
        assert_almost_equal(err, 0.025868,
                     msg="The found regression vector is not correct.",
                     places=5)

        mu = 5e-4
        logreg_ista = estimators.RidgeRegression_L1_TV(
                          k=k,
                          l=l,
                          g=g,
                          A=A,
                          mu=mu,
                          output=False,
                          algorithm=explicit.ISTA(eps=eps,
                                                  max_iter=10000))
        logreg_ista.fit(X, y)
        err = logreg_ista.score(X, y)
#        print err
        assert_almost_equal(err, 0.034949,
                     msg="The found regression vector is not correct.",
                     places=5)

if __name__ == "__main__":
    unittest.main()