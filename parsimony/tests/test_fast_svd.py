# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:23:53 2013

@author: jinpeng
"""

import unittest
import numpy as np
from parsimony.algorithms import FastSVD
from parsimony.algorithms import FastSparseSVD
import parsimony.utils as utils


def generate_sparse_matrix(shape, density=0.10):
    '''
    Example
    -------
    shape = (5, 5)
    density = 0.2
    print generate_sparse_matrix(shape, density)
    '''
    # shape = (5, 5)
    # density = 0.1
    num_elements = 1
    for i in xrange(len(shape)):
        num_elements = num_elements * shape[i]
    zero_vec = np.zeros(num_elements, dtype=float)
    indices = np.random.random_integers(0,
                                        num_elements - 1,
                                        int(density * num_elements))
    zero_vec[indices] = np.random.random_sample(len(indices))
    sparse_mat = np.reshape(zero_vec, shape)
    return sparse_mat


class TestSVD(unittest.TestCase):

    def get_err_by_np_linalg_svd(self, computed_v, X):
        # svd from numpy array
        U, s_np, V = np.linalg.svd(X)
        np_v = V[[0], :].T
        err = np.sum(
            np.absolute(computed_v / np_v) - np.ones((np_v.shape[0], 1),
            dtype=computed_v.dtype))
        return err

    def get_err_fast_svd(self, nrow, ncol):
        X = np.random.random((nrow, ncol))
        # svd from parsimony
        parsimony_v = FastSVD(X, max_iter=1000)
        return self.get_err_by_np_linalg_svd(parsimony_v, X)

    def test_fast_svd(self):
        err = self.get_err_fast_svd(50, 50)
        self.assertTrue(err < utils.TOLERANCE)
        err = self.get_err_fast_svd(5000, 5)
        self.assertTrue(err < utils.TOLERANCE)
        err = self.get_err_fast_svd(5, 5000)
        self.assertTrue(err < utils.TOLERANCE)

    def get_err_fast_sparse_svd(self, nrow, ncol, density):
        X = generate_sparse_matrix(shape=(nrow, ncol),
                                   density=density)
        # svd from parsimony
        parsimony_v = FastSparseSVD(X, max_iter=1000)
        return self.get_err_by_np_linalg_svd(parsimony_v, X)

    def test_fast_sparse_svd(self):
        err = self.get_err_fast_sparse_svd(50, 50, density=0.1)
        self.assertTrue(err < (utils.TOLERANCE * 100))
        err = self.get_err_fast_sparse_svd(500, 5000, density=0.1)
        self.assertTrue(err < (utils.TOLERANCE * 100))
        err = self.get_err_fast_sparse_svd(5000, 500, density=0.1)
        self.assertTrue(err < (utils.TOLERANCE * 100))


if __name__ == '__main__':
    unittest.main()
