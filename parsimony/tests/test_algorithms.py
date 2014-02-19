# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:42:07 2013

@author:  Jinpeng Li
@email:   jinpeng.li@cea.fr
@license: BSD 3-clause.
"""
import unittest

from parsimony.tests.spamsdata import SpamsGenerator
from tests import TestCase


class TestAlgorithms(TestCase):

    def test_algorithms(self):
        # Compares three algorithms (FISTA, conesta_static, and
        # conesta_dynamic) to the SPAMS FISTA algorithm.

        import numpy as np
        import parsimony.estimators as estimators
        import parsimony.algorithms as algorithms
        import parsimony.functions.nesterov.tv as tv
        spams_generator = SpamsGenerator()
        ret_data = spams_generator.get_x_y_estimated_beta()
        weight_l1_spams = ret_data['weight_l1']
        shape = ret_data["shape"]
        X = ret_data["X"]
        y = ret_data["y"]
        # WARNING: We must have a non-zero ridge parameter!
        k = 5e-8  # ridge regression coefficient
        l = 0.05  # l1 coefficient
        # WARNING: We must have a non-zero TV parameter!
        g = 5e-8  # tv coefficient

        Atv, n_compacts = tv.A_from_shape(shape)
        tvl1l2_algorithms = []
        # Al1 = sparse.eye(num_ft, num_ft)
        tvl1l2_fista = estimators.RidgeRegression_L1_TV(
                                k, l, g,
                                Atv,
                                algorithm=algorithms.FISTA())
        tvl1l2_conesta_static = estimators.RidgeRegression_L1_TV(
                                k, l, g,
                                Atv,
                                algorithm=algorithms.CONESTA(dynamic=False))
        tvl1l2_conesta_dynamic = estimators.RidgeRegression_L1_TV(
                                k, l, g,
                                Atv,
                                algorithm=algorithms.CONESTA(dynamic=True))

        tvl1l2_algorithms.append(tvl1l2_fista)
        tvl1l2_algorithms.append(tvl1l2_conesta_static)
        tvl1l2_algorithms.append(tvl1l2_conesta_dynamic)

        for tvl1l2_algorithm in tvl1l2_algorithms:
            print str(tvl1l2_algorithm.algorithm)
            tvl1l2_algorithm.fit(X, y)
            ## sometimes betas are different
            ## but lead to the same error (err1 and err2)
            # error = np.sum(np.absolute(tvl1l2_algorithm.beta - W))
            # self.assertTrue(error < 0.01)
            err1 = np.sum(np.absolute(
                          np.dot(X, tvl1l2_algorithm.beta) - y))
            err2 = np.sum(np.absolute(np.dot(X, weight_l1_spams) - y))
            self.assertTrue(np.absolute(err1 - err2) < 0.01,
                            np.absolute(err1 - err2))
            weight_err = np.linalg.norm(
                            tvl1l2_algorithm.beta - weight_l1_spams)
            self.assertTrue(weight_err < 0.01, weight_err)

if __name__ == "__main__":
    unittest.main()