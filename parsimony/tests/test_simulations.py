# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 15:28:08 2014

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import unittest
from nose.tools import assert_less

import numpy as np
import matplotlib.pyplot as plot

from tests import TestCase


class TestSimulations(TestCase):

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
        pz = 1
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
        beta[0:5, :] = 0.0

#        print beta

        l = 0.618
        k = 1.0 - l
        g = 1.618

        A, _ = tv.A_from_shape(shape)
        snr = 100.0
        eps = 1e-8
        max_iter = 200

        X, y, beta_star = l1_l2_tv.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
                                        A=A, snr=snr)

        errs = []
        effs = []
        gammas = np.linspace(g * 0.5, g, 10).tolist()[:-1] \
               + np.linspace(g, g * 2.0, 10).tolist()
        for gamma in gammas:
            mus = [5e-2, 5e-4, 5e-6, 5e-8]
            fista = explicit.FISTA(eps=eps, max_iter=max_iter / len(mus))
            beta_start = start_vector.get_vector((p, 1))

            beta_nonsmooth_penalty = beta_start
            function = None
            for mu in mus:
                function = CombinedFunction()
                function.add_function(LinearRegression(X, y, mean=False))
                function.add_penalty(tv.TotalVariation(l=gamma, A=A, mu=mu,
                                                       penalty_start=0))
                function.add_penalty(L2(k))
                function.add_prox(L1(l))
                beta_nonsmooth_penalty = \
                        fista.run(function, beta_nonsmooth_penalty)

            mse = np.linalg.norm(beta_nonsmooth_penalty - beta_star) \
                    / np.linalg.norm(beta_star)
            errs.append(mse)

            f_nonsmooth_star = function.f(beta_star)
            f_nonsmooth_penalty = function.f(beta_nonsmooth_penalty)
            eff = abs(f_nonsmooth_penalty - f_nonsmooth_star) / f_nonsmooth_star
            effs.append(eff)

#        plot.subplot(2, 1, 1)
#        plot.plot(gammas, errs)
#        print np.min(errs)
#        plot.subplot(2, 1, 2)
#        plot.plot(gammas, effs)
#        print np.min(effs)
#        plot.show()

#        mse = np.linalg.norm(beta_nonsmooth_penalty - beta_star) \
#                    / np.linalg.norm(beta_star)
#        print "mse:", mse
#        assert_less(mse, 1e-5, "The found regression vector is not correct.")
#
#        f_nonsmooth_star = function.f(beta_star)
#        f_nonsmooth_penalty = function.f(beta_nonsmooth_penalty)
#        err = abs(f_nonsmooth_penalty - f_nonsmooth_star) / f_nonsmooth_star
#        print "err:", err
#        assert_less(err, 1e-3, "The found regression vector does not give " \
#                               "the correct function value.")

#        beta_nonsmooth_rr = beta_start
#        for mu in mus:
#            function = CombinedFunction()
#            function.add_function(RidgeRegression(X, y, k))
#            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
#                                                   penalty_start=0))
#            function.add_prox(L1(l))
#            beta_nonsmooth_rr = fista.run(function, beta_nonsmooth_rr)
#
#        mse = (np.linalg.norm(beta_nonsmooth_rr - beta_star) ** 2.0) / p
##        print "mse:", mse
#        assert_less(mse, 1e-5, "The found regression vector is not correct.")
#
#        f_nonsmooth_star = function.f(beta_star)
#        f_nonsmooth_rr = function.f(beta_nonsmooth_rr)
#        err = abs(f_nonsmooth_rr - f_nonsmooth_star)
##        print "err:", err
#        assert_less(err, 1e-3, "The found regression vector does not give " \
#                               "the correct function value.")
#
#        mu_min = mus[-1]
#        X, y, beta_star = l1_l2_tvmu.load(l=l, k=k, g=g, beta=beta, M=M, e=e,
#                                          A=A, mu=mu_min, snr=snr)
#        beta_smooth_penalty = beta_start
#        for mu in mus:
#            function = CombinedFunction()
#            function.add_function(LinearRegression(X, y, mean=False))
#            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
#                                                   penalty_start=0))
#            function.add_penalty(L2(k))
#            function.add_prox(L1(l))
#            beta_smooth_penalty = \
#                    fista.run(function, beta_smooth_penalty)
#
#        mse = (np.linalg.norm(beta_smooth_penalty - beta_star) ** 2.0) / p
##        print "mse:", mse
#        assert_less(mse, 1e-5, "The found regression vector is not correct.")
#
#        f_smooth_star = function.f(beta_star)
#        f_smooth_penalty = function.f(beta_smooth_penalty)
#        err = abs(f_smooth_penalty - f_smooth_star)
##        print "err:", err
#        assert_less(err, 1e-3, "The found regression vector does not give " \
#                               "the correct function value.")
#
#        beta_smooth_rr = beta_start
#        for mu in mus:
#            function = CombinedFunction()
#            function.add_function(RidgeRegression(X, y, k))
#            function.add_penalty(tv.TotalVariation(l=g, A=A, mu=mu,
#                                                   penalty_start=0))
#            function.add_prox(L1(l))
#            beta_smooth_rr = fista.run(function, beta_smooth_rr)
#
#        mse = (np.linalg.norm(beta_smooth_rr - beta_star) ** 2.0) / p
##        print "mse:", mse
#        assert_less(mse, 1e-5, "The found regression vector is not correct.")
#
#        f_smooth_star = function.f(beta_star)
#        f_smooth_rr = function.f(beta_smooth_rr)
#        err = abs(f_smooth_rr - f_smooth_star)
##        print "err:", err
#        assert_less(err, 1e-3, "The found regression vector does not give " \
#                               "the correct function value.")

if __name__ == "__main__":
    unittest.main()