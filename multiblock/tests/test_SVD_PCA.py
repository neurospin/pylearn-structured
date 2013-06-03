# -*- coding: utf-8 -*-
"""
Created on Thu May 23 09:16:42 2013

TODO: Add Sparse SVD.

@author: Tommy Löfstedt
"""

import numpy as np
import preprocess
import algorithms
import prox_ops
import methods
from utils import direct, norm
from utils.testing import assert_array_almost_equal
from math import log
from time import time


def test():

    # Assure same answer every time
    np.random.seed(42)

    # Compare SVD with and without sparsity constraint to numpy.linalg.svd
    Xtr = np.random.rand(6, 6)
    #Xte = np.random.rand(2,6)
    num_comp = 3
    tol = 5e-12

    preproc = preprocess.PreprocessQueue([preprocess.Center(),
                                          preprocess.Scale()], Xtr)
    Xtr = preproc.process(Xtr)

    for st in [0.1, 0.01, 0.001, 0.0001, 0]:
        svd = methods.SVD(num_comp=num_comp)
        svd.set_prox_op(prox_ops.L1(st))
        svd.set_tolerance(tol)
        svd.set_max_iter(1000)
        svd.fit(Xtr)

        U, S, V = np.linalg.svd(Xtr)
        V = V.T
        S = np.diag(S)
        U = U[:, 0:num_comp]
        S = S[:, 0:num_comp]
        V = V[:, 0:num_comp]

        if st < tol:
            num_decimals = 5
        else:
            num_decimals = int(log(1. / st, 10) + 0.5)
        svd.V, V = direct(svd.V, V, compare=True)
        assert_array_almost_equal(svd.V, V, decimal=num_decimals - 2,
                err_msg="sklearn.NIPALS.SVD and numpy.linalg.svd " \
                "implementations lead to different loadings")

    # Compare PCA with sparsity constraint to numpy.linalg.svd
    Xtr = np.random.rand(5, 5)
    num_comp = 5
    tol = 5e-9

    preproc = preprocess.PreprocessQueue([preprocess.Center(),
                                          preprocess.Scale()], Xtr)
    Xtr = preproc.process(Xtr)

    for st in [0.1, 0.01, 0.001, 0.0001, 0]:
        pca = methods.PCA(num_comp=num_comp)
        svd.set_prox_op(prox_ops.L1(st))
        svd.set_tolerance(tol)
        svd.set_max_iter(1000)
        pca.fit(Xtr)

        Tte = pca.transform(Xtr)
        U, S, V = np.linalg.svd(Xtr)
        V = V.T
        US = np.dot(U, np.diag(S))
        US = US[:, 0:num_comp]
        V = V[:, 0:num_comp]

        if st < tol:
            num_decimals = 5
        else:
            num_decimals = int(log(1. / st, 10) + 0.5)
        assert_array_almost_equal(Xtr, np.dot(pca.T, pca.P.T),
                                  decimal=num_decimals - 1,
                                  err_msg="Model does not equal the matrices")

    # Compare PCA without the sparsity constraint to numpy.linalg.svd
    Xtr = np.random.rand(50, 50)
    Xte = np.random.rand(20, 50)
    num_comp = 3

    preproc = preprocess.PreprocessQueue([preprocess.Center(),
                                          preprocess.Scale()], Xtr)
    Xtr = preproc.process(Xtr)
    Xte = preproc.process(Xte)

    pca = methods.PCA(num_comp=num_comp)
    pca.set_tolerance(tol)
    pca.set_max_iter(1000)
    pca.fit(Xtr)

    pca.P, pca.T = direct(pca.P, pca.T)
    Tte = pca.transform(Xte)

    U, S, V = np.linalg.svd(Xtr)
    V = V.T
    US = np.dot(U, np.diag(S))
    US = US[:, 0:num_comp]
    V = V[:, 0:num_comp]
    V, US = direct(V, US)
    SVDte = np.dot(Xte, V)

    assert_array_almost_equal(pca.P, V, decimal=2, err_msg="NIPALS PCA and "
            "numpy.linalg.svd implementations lead to different loadings")

    assert_array_almost_equal(pca.T, US, decimal=2, err_msg="NIPALS PCA and "
            "numpy.linalg.svd implementations lead to different scores")

    assert_array_almost_equal(Tte, SVDte, decimal=2, err_msg="NIPALS PCA and "
            "numpy.linalg.svd implementations lead to different scores")

    # Compare PCA without the sparsity constraint to numpy.linalg.svd
    X = np.random.rand(50, 100)
    num_comp = 50

    preproc = preprocess.PreprocessQueue([preprocess.Center(),
                                          preprocess.Scale()], X)
    X = preproc.process(X)

    pca = methods.PCA(num_comp=num_comp)
    pca.set_tolerance(tol)
    pca.set_max_iter(1000)
    pca.fit(X)

    Xhat_1 = np.dot(pca.T, pca.P.T)

    U, S, V = np.linalg.svd(X, full_matrices=False)
    Xhat_2 = np.dot(U, np.dot(np.diag(S), V))

    assert_array_almost_equal(X, Xhat_1, decimal=2, err_msg="PCA performs "
            " a faulty reconstruction of X")

    assert_array_almost_equal(Xhat_1, Xhat_2, decimal=2, err_msg="PCA and " \
            "numpy.linalg.svd implementations lead to different " \
            "reconstructions")

    # Compare PCA without the sparsity constraint to numpy.linalg.svd
    X = np.random.rand(100, 50)
    num_comp = 50

    preproc = preprocess.PreprocessQueue([preprocess.Center(),
                                          preprocess.Scale()], X)
    X = preproc.process(X)

    pca = methods.PCA(num_comp=num_comp)
    pca.set_prox_op(prox_ops.L1(st))
    pca.set_tolerance(tol)
    pca.set_max_iter(1500)
    pca.fit(X)
    Xhat_1 = np.dot(pca.T, pca.P.T)

    U, S, V = np.linalg.svd(X, full_matrices=False)
    Xhat_2 = np.dot(U, np.dot(np.diag(S), V))

    assert_array_almost_equal(X, Xhat_1, decimal=2, err_msg="PCA performs a " \
        "faulty reconstruction of X")

    assert_array_almost_equal(Xhat_1, Xhat_2, decimal=2, err_msg="PCA and "
            "numpy.linalg.svd implementations lead to different " \
            "reconstructions")

    # Assure TuckerFactorAnalysis gives the same answer as SVD
    tol = 5e-10
    miter = 1500
    X = np.random.rand(50, 100)
    Y = np.random.rand(50, 100)
    num_comp = 10

    preprocX = preprocess.PreprocessQueue([preprocess.Center(),
                                           preprocess.Scale()], X)
    preprocY = preprocess.PreprocessQueue([preprocess.Center(),
                                           preprocess.Scale()], Y)
    X = preprocX.process(X)
    Y = preprocY.process(Y)

    tfa = methods.TuckerFactorAnalysis(num_comp=num_comp)
    tfa.set_max_iter(miter)
    tfa.set_tolerance(tol)
    tfa.fit(X, Y)

    svd = methods.SVD(num_comp=num_comp)
    svd.set_max_iter(miter)
    svd.set_tolerance(tol)
    svd.fit(np.dot(X.T, Y))

    tfa.W, svd.U = direct(tfa.W, svd.U, compare=True)
    assert_array_almost_equal(tfa.W, svd.U, decimal=5, err_msg="Tucker's " \
        "inner-battery factor analysis gives different X weights when " \
        "compared to SVD")

    tfa.C, svd.V = direct(tfa.C, svd.V, compare=True)
    assert_array_almost_equal(tfa.C, svd.V, decimal=5, err_msg="Tucker's " \
        "inner-battery factor analysis gives different Y weights when " \
        "compared to SVD")

    # Compare the accuracy and speed of the FastSVD
    tol = 5e-10
    miter = 2000
    X = np.random.rand(100, 1000)

    preproc = preprocess.PreprocessQueue([preprocess.Center(),
                                          preprocess.Scale()], X)
    X = preproc.process(X)

    start = time()
    svd = algorithms.FastSVD()
    svd.tolerance = tol
    svd.max_iter = miter
    p = svd.run(X)
    time_fast = time() - start

    Xhat_1 = np.dot(np.dot(X, p), p.T)

    start = time()
    U, S, V = np.linalg.svd(X, full_matrices=True)
    time_svd = time() - start
    Xhat_2 = np.dot(U[:, [0]], np.dot(np.diag(S[[0]]), V[[0], :]))

    v, p = direct(V[[0], :].T, p, compare=True)

    assert_array_almost_equal(v, p, decimal=3, err_msg="FastSVD does not "
            " find the correct loading vector.")

    assert_array_almost_equal(Xhat_1, Xhat_2, decimal=2, err_msg="FastSVD " \
            "and numpy.linalg.svd implementations lead to different " \
            "reconstructions")

    assert time_fast < time_svd


if __name__ == "__main__":

    test()