# -*- coding: utf-8 -*-
"""
Created on Thu May 23 09:37:48 2013

TODO: Fix the proximal operators. Normalisation?

@author: Tommy Löfstedt
"""

import numpy as np
import start_vectors
import preprocess
import prox_ops
import models

import utils
from utils.testing import assert_array_almost_equal
from utils.testing import orth_matrix, fleiss_kappa

from sklearn.datasets import load_linnerud


def test():

    np.random.seed(1)

    d = load_linnerud()
    Xorig = d.data
    Yorig = d.target
#    print X.shape
#    print Y.shape
    tol = 5e-12
    miter = 1500
#    Xorig = X.copy()
#    Yorig = Y.copy()
    center = True
    scale = True
    inf = 2 ** 30
    SSY = np.sum(Yorig ** 2.0)
    num_comp = 2

    preprocX = preprocess.PreprocessQueue([], Xorig)
    preprocY = preprocess.PreprocessQueue([], Yorig)
    if center:
        preprocX.push(preprocess.Center())
        preprocY.push(preprocess.Center())
    if scale:
        preprocX.push(preprocess.Scale())
        preprocY.push(preprocess.Scale())

    # Test first with PLSR
    X = preprocX.process(Xorig)
    Y = preprocY.process(Yorig)
    pls = models.PLSR(num_comp=num_comp)
    pls.set_tolerance(tol)
    pls.set_max_iter(miter)
    pls.fit(X, Y)
    Yhat = pls.predict(X)
    Yhat = preprocY.revert(Yhat)
    SSYdiff = np.sum((Yorig - Yhat) ** 2.0)
    R2Yhat = (1.0 - (SSYdiff / SSY))
    utils.debug("PLS : R2Yhat = %.6f" % R2Yhat)

    # Test sPLS models when keeping all variables
    spls1 = models.PLSR(num_comp=num_comp)
    spls1.set_tolerance(tol)
    spls1.set_max_iter(miter)
    spls1.set_prox_op(prox_ops.L1(0., 0., normaliser=[utils.norm,
                                                      utils.normI]))
    spls1.fit(X, Y)
    Yhat1 = spls1.predict(X)
    Yhat1 = preprocY.revert(Yhat1)
    SSYdiff1 = np.sum((Yorig - Yhat1) ** 2.0)
    utils.debug("sPLS: R2Yhat = %.6f" % (1.0 - (SSYdiff1 / SSY)))
    assert abs(R2Yhat - (1.0 - (SSYdiff1 / SSY))) < utils.TOLERANCE
    assert_array_almost_equal(Yhat, Yhat1, decimal=5,
            err_msg="Sparse PLS with no thresholding does not give correct " \
                    "result")

#    spls2 = PLSR(num_comp=num_comp)
##    alg = spls2.get_algorithm()
#    spls2.set_tolerance(tol)
#    spls2.set_max_iter(miter)
#    spls2.set_prox_op(prox_ops.L1_binsearch(float('Inf'), float('Inf'),
#                      normaliser=[norm, normI]))
#    spls2.fit(X, Y)
#    Yhat2 = spls2.predict(X)
#    Yhat2 = preprocY.revert(Yhat2)
#    SSYdiff2 = np.sum((Yorig - Yhat2) ** 2)
#    utils.debug("sPLS: R2Yhat = %.6f" % (1 - (SSYdiff2 / SSY)))
#    assert abs(R2Yhat - (1 - (SSYdiff1 / SSY))) < TOLERANCE
#    assert_array_almost_equal(Yhat, Yhat2, decimal=5,
#            err_msg="Sparse PLS with no thresholding does not give correct " \
#                    "result")
#
#    spls3 = PLSR(num_comp=num_comp)
#    alg = spls3.get_algorithm()
#    alg.set_tolerance(tol)
#    alg.set_max_iter(miter)
#    alg.set_prox_op(prox_ops.L0_binsearch(inf, inf, normaliser=[norm, normI]))
#    spls3.fit(X, Y)
#    Yhat3 = spls3.predict(X)
#    Yhat3 = preprocY.revert(Yhat3)
#    SSYdiff3 = np.sum((Yorig - Yhat3) ** 2)
#    utils.debug("sPLS: R2Yhat = %.6f" % (1 - (SSYdiff3 / SSY)))
#    assert_array_almost_equal(Yhat, Yhat3, decimal=5,
#            err_msg="Sparse PLS with no thresholding does not give correct " \
#                    "result")
#
#    spls4 = PLSR(num_comp=num_comp)
#    alg = spls4.get_algorithm()
#    alg.set_tolerance(tol)
#    alg.set_max_iter(miter)
#    alg.set_prox_op(prox_ops.L0_by_count(inf, inf, normaliser=[norm, normI]))
#    spls4.fit(X, Y)
#    Yhat4 = spls4.predict(X)
#    Yhat4 = preprocY.revert(Yhat4)
#    SSYdiff4 = np.sum((Yorig - Yhat4) ** 2)
#    utils.debug("sPLS: R2Yhat = %.6f" % (1 - (SSYdiff4 / SSY)))
#    assert_array_almost_equal(Yhat, Yhat4, decimal=5,
#            err_msg="Sparse PLS with no thresholding does not give correct " \
#                    "result")

    # Create a matrix X (10,11) with variables with
    # correlation 1 throught 0 to a single y variable
#    np.random.seed(38) # 15, 22, 32, 38, 40
    n_sz = 10
    Xorig, Yorig = orth_matrix(n_sz)
    SSX = np.sum(Xorig ** 2.0)
    SSY = np.sum(Yorig ** 2.0)

    preprocX = preprocess.PreprocessQueue([], Xorig)
    preprocY = preprocess.PreprocessQueue([], Yorig)
    if center:
        preprocX.push(preprocess.Center())
        preprocY.push(preprocess.Center())
    if scale:
        preprocX.push(preprocess.Scale())
        preprocY.push(preprocess.Scale())
    X = preprocX.process(Xorig)
    Y = preprocY.process(Yorig)

    utils.debug()
    num_comp = n_sz - 1
    # Analyse with PLSR
    pls = models.PLSR(num_comp=num_comp)
    pls.set_tolerance(tol)
    pls.set_max_iter(miter)
    pls.fit(X, Y)
    Yhat = pls.predict(X)
    Yhat = preprocY.revert(Yhat)
    SSYdiff = np.sum((Yorig - Yhat) ** 2.0)
    R2Yhat = (1.0 - (SSYdiff / SSY))
    utils.debug("PLS :         R2Yhat=%.6f, num_comp=%d" % (R2Yhat, num_comp))

    num_comp = 1
    # Analyse with PLSR
    pls = models.PLSR(num_comp=num_comp)
    pls.set_tolerance(tol)
    pls.set_max_iter(miter)
    pls.fit(X, Y)
    Yhat = pls.predict(X)
    Yhat = preprocY.revert(Yhat)
    SSYdiff = np.sum((Yorig - Yhat) ** 2.0)
    R2Yhat = (1.0 - (SSYdiff / SSY))
    utils.debug("PLS :         R2Yhat=%.6f, num_comp=%d" % (R2Yhat, num_comp))

    # Analyse with Sparse PLSR (L1)
    nonzero = []
    for l in np.linspace(0, 0.9, 10).tolist():
        spls1 = models.PLSR(num_comp=num_comp)
#        alg = spls1.get_algorithm()
        spls1.set_tolerance(tol)
        spls1.set_max_iter(miter)
        spls1.set_prox_op(prox_ops.L1(l, 0, normaliser=[utils.norm,
                                                        utils.normI]))
        spls1.fit(X, Y)
        Yhat1 = spls1.predict(X)
        Yhat1 = preprocY.revert(Yhat1)
        SSYdiff1 = np.sum((Yorig - Yhat1) ** 2.0)
        R2Yhat1 = 1.0 - (SSYdiff1 / SSY)
        nonzero.append(np.count_nonzero(spls1.W))
        utils.debug("sPLS: l=%.2f, R2Yhat=%.6f, num_comp=%d, nonzero=%d" \
                % (l, R2Yhat1, num_comp, nonzero[-1]))

        assert all(x <= y for x, y in zip(np.abs(spls1.W)[:, 0],
                                          (np.abs(spls1.W)[:, 0])[1:]))
        assert all(x >= y for x, y in zip(nonzero, nonzero[1:]))
    assert abs(R2Yhat1 - 1.0) < utils.TOLERANCE

#    # Analyse with Sparse PLSR (L1_binsearch)
#    utils.debug()
#    utils.debug("PLS :         R2Yhat=%.6f, num_comp=%d" % (R2Yhat, num_comp))
#    nonzero = []
#    for s in [float('Inf'), 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.25, 0.125, 0.0625,
#              0.03125, 0]:
#        spls2 = PLSR(num_comp=num_comp)
#        alg = spls2.get_algorithm()
#        alg.set_tolerance(tol)
#        alg.set_max_iter(miter)
#        alg.set_prox_op(prox_ops.L1_binsearch(s, float('Inf'),
#                                              normaliser=[norm, normI]))
#        spls2.fit(X, Y)
#        Yhat2 = spls2.predict(X)
#        Yhat2 = preprocY.revert(Yhat2)
#        SSYdiff2 = np.sum((Yorig - Yhat2) ** 2)
#        R2Yhat2 = 1 - (SSYdiff2 / SSY)
#        nonzero.append(np.count_nonzero(spls2.W))
#        utils.debug("sPLS: s=%-4.2f, R2Yhat=%.6f, num_comp=%d, nonzero=%d" \
#                % (s, R2Yhat2, num_comp, nonzero[-1]))
#
#        assert all(x <= y for x, y in zip(np.abs(spls2.W)[:, 0],
#                                          (np.abs(spls2.W)[:, 0])[1:]))
#        assert all(x >= y for x, y in zip(nonzero, nonzero[1:]))
#    assert abs(R2Yhat2 - 1) < TOLERANCE
#
#    # Analyse with Sparse PLSR (L0_binsearch)
#    utils.debug()
#    utils.debug("PLS :        R2Yhat=%.6f, num_comp=%d" % (R2Yhat, num_comp))
#    nonzero = []
#    for n in [100, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]:
#        spls3 = PLSR(num_comp=num_comp)
#        alg = spls3.get_algorithm()
#        alg.set_tolerance(tol)
#        alg.set_max_iter(miter)
#        alg.set_prox_op(prox_ops.L0_binsearch(n, 100,
#                                              normaliser=[norm, normI]))
#        spls3.fit(X, Y)
#        Yhat3 = spls3.predict(X)
#        Yhat3 = preprocY.revert(Yhat3)
#        SSYdiff3 = np.sum((Yorig - Yhat3) ** 2)
#        R2Yhat3 = 1 - (SSYdiff3 / SSY)
#        nonzero.append(np.count_nonzero(spls3.W))
#        utils.debug("sPLS: n=%3d, R2Yhat=%.6f, num_comp=%d, nonzero=%d" \
#                % (n, R2Yhat3, num_comp, nonzero[-1]))
#
#        assert all(x <= y for x, y in zip(np.abs(spls3.W)[:, 0],
#                                          (np.abs(spls3.W)[:, 0])[1:]))
#        assert all(x >= y for x, y in zip(nonzero, nonzero[1:]))
#    assert abs(R2Yhat3 - 1) < TOLERANCE
#
#    # Analyse with Sparse PLSR (L0_by_count)
#    utils.debug()
#    utils.debug("PLS :        R2Yhat=%.6f, num_comp=%d" % (R2Yhat, num_comp))
#    nonzero = []
#    for n in [100] + range(11, -1, -1):
#        spls4 = PLSR(num_comp=num_comp)
#        alg = spls4.get_algorithm()
#        alg.set_tolerance(tol)
#        alg.set_max_iter(miter)
#        alg.set_prox_op(prox_ops.L0_by_count(n, 100, normaliser=[norm, normI]))
#        spls4.fit(X, Y)
#        Yhat4 = spls4.predict(X)
#        Yhat4 = preprocY.revert(Yhat4)
#        SSYdiff4 = np.sum((Yorig - Yhat4) ** 2)
#        R2Yhat4 = 1 - (SSYdiff4 / SSY)
#        nonzero.append(np.count_nonzero(spls4.W))
#        utils.debug("sPLS: n=%3d, R2Yhat=%.6f, num_comp=%d, nonzero=%d" \
#                % (n, R2Yhat4, num_comp, nonzero[-1]))
#
#        assert all(x <= y for x, y in zip(np.abs(spls4.W)[:, 0],
#                                          (np.abs(spls4.W)[:, 0])[1:]))
#        assert all(x >= y for x, y in zip(nonzero, nonzero[1:]))
#    assert abs(R2Yhat4 - 1) < TOLERANCE

    # Analyse with O2PLS
    o2pls = models.O2PLS(num_comp=[num_comp, 8, 0])
    o2pls.set_tolerance(tol)
    o2pls.set_max_iter(miter)
    o2pls.fit(X, Y)
    Yhat = o2pls.predict(X)
    Yhat = preprocY.revert(Yhat)
    SSYdiff = np.sum((Yorig - Yhat) ** 2)
    R2Yhat = (1 - (SSYdiff / SSY))
    utils.debug()
    utils.debug("O2PLS :         R2Yhat=%.5f, num_comp=%d, num_orth=%2d" \
            % (R2Yhat, num_comp, 8))

    # Analyse with Sparse O2PLS (L1)
    nonzeroW = []
    nonzeroWo = []
    n_cp = 2
    for l in np.linspace(0, 0.9, 19).tolist():
        num_orth = max(n_sz - n_cp, 0)
        so2pls1 = models.O2PLS(num_comp=[num_comp, num_orth, 0])
        so2pls1.set_tolerance(tol)
        so2pls1.set_max_iter(miter)
        so2pls1.set_prox_op(prox_ops.L1([l, l], [0, 0]))
        so2pls1.fit(X, Y)
        Yhat1 = so2pls1.predict(X)
        Yhat1 = preprocY.revert(Yhat1)
        SSYdiff1 = np.sum((Yorig - Yhat1) ** 2.0)
        R2Yhat1 = 1 - (SSYdiff1 / SSY)
        nonzeroW.append(np.count_nonzero(so2pls1.W[:, [0]]))
        if so2pls1.Wo.shape[1] > 0:
            nonzeroWo.append(np.count_nonzero(so2pls1.Wo[:, [0]]))
        else:
            nonzeroWo.append(0)
        utils.debug("sO2PLS: l=%.2f, R2Yhat=%.5f, num_comp=%d, num_orth=%2d, "\
              "nonzeroW=%2d, nonzeroWo=%2d" \
              % (l, R2Yhat1, num_comp, num_orth, nonzeroW[-1], nonzeroWo[-1]))
        n_cp += 1

#        assert all(x <= y for x, y in zip(np.abs(so2pls1.W)[:,0], (np.abs(so2pls1.W)[:,0])[1:]))
#        assert all(x >= y for x, y in zip(nonzeroW, nonzeroW[1:]))
#        assert all(x <= y for x, y in zip(nonzeroWo, nonzeroWo[1:]))
    assert abs(R2Yhat1 - 1.0) < 0.0005  # TOLERANCE

#    # Analyse with Sparse O2PLS (L1_binsearch)
#    utils.debug()
#    utils.debug("O2PLS :         R2Yhat=%.5f, num_comp=%d" \
#            % (R2Yhat, num_comp))
#    nonzeroW = []
#    nonzeroWo = []
#    n_cp = 0
#    for s in [float('Inf'), 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.25, 0.125, 0.0625,
#              0.03125, 0]:
#        num_orth = max(n_sz - n_cp, 0)
#        so2pls2 = O2PLS(num_comp=[num_comp, num_orth, 0])
#        alg = so2pls2.get_algorithm()
#        alg.set_tolerance(tol)
#        alg.set_max_iter(miter)
#        alg.set_prox_op(prox_ops.L1_binsearch([s, s], [float('Inf')] * 2))
#        so2pls2.fit(X, Y)
#        Yhat2 = so2pls2.predict(X)
#        Yhat2 = preprocY.revert(Yhat2)
#        SSYdiff2 = np.sum((Yorig - Yhat2) ** 2)
#        R2Yhat2 = 1 - (SSYdiff2 / SSY)
#        nonzeroW.append(np.count_nonzero(so2pls2.W[:, [0]]))
#        if so2pls2.Wo.shape[1] > 0:
#            nonzeroWo.append(np.count_nonzero(so2pls2.Wo[:, [0]]))
#        else:
#            nonzeroWo.append(0)
#        utils.debug("sO2PLS: s=%4.2f, R2Yhat=%.5f, num_comp=%d, " \
#                    "num_orth=%2d, nonzeroW=%2d, nonzeroWo=%2d" \
#                    % (s, R2Yhat2, num_comp, num_orth, nonzeroW[-1],
#                       nonzeroWo[-1]))
#        n_cp += 1
#
##        assert all(x <= y for x, y in zip(np.abs(so2pls2.W)[:,0], (np.abs(so2pls2.W)[:,0])[1:]))
##        assert all(x >= y for x, y in zip(nonzeroW, nonzeroW[1:]))
##        assert all(x <= y for x, y in zip(nonzeroWo, nonzeroWo[1:]))
#    assert abs(R2Yhat2 - 1) < 0.0005  # TOLERANCE
#
#    # Analyse with Sparse O2PLS (L0_binsearch)
#    utils.debug()
#    utils.debug("O2PLS :        R2Yhat=%.6f, num_comp=%d" % (R2Yhat, num_comp))
#    nonzeroW = []
#    nonzeroWo = []
#    n_cp = 0
#    for n in [100, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]:
#        num_orth = max(n_sz - n_cp, 0)
#        so2pls3 = O2PLS(num_comp=[num_comp, num_orth, 0])
#        alg = so2pls3.get_algorithm()
#        alg.set_tolerance(tol)
#        alg.set_max_iter(miter)
#        alg.set_prox_op(prox_ops.L0_binsearch([n, n], [100, 100]))
#        so2pls3.fit(X, Y)
#        Yhat3 = so2pls3.predict(X)
#        Yhat3 = preprocY.revert(Yhat3)
#        SSYdiff3 = np.sum((Yorig - Yhat3) ** 2)
#        R2Yhat3 = 1 - (SSYdiff3 / SSY)
#        nonzeroW.append(np.count_nonzero(so2pls3.W[:, 0]))
#        if so2pls3.Wo.shape[1] > 0:
#            nonzeroWo.append(np.count_nonzero(so2pls3.Wo[:, 0]))
#        else:
#            nonzeroWo.append(0)
#        utils.debug("sO2PLS: n=%3d, R2Yhat=%.6f, num_comp=%d, num_orth=%2d, " \
#                    "nonzeroW=%2d, nonzeroWo=%2d" \
#                    % (n, R2Yhat3, num_comp, num_orth, nonzeroW[-1],
#                       nonzeroWo[-1]))
#        n_cp += 1
#
##        assert all(x <= y for x, y in zip(np.abs(so2pls3.W)[:,0], (np.abs(so2pls3.W)[:,0])[1:]))
#        assert all(x >= y for x, y in zip(nonzeroW, nonzeroW[1:]))
##        assert all(x <= y for x, y in zip(nonzeroWo, nonzeroWo[1:]))
#    assert abs(R2Yhat3 - 1) < TOLERANCE
#
#    utils.debug()
#    utils.debug("O2PLS :          R2Yhat = %.6f, num_comp = %d" \
#            % (R2Yhat, num_comp))
#    nonzeroW = []
#    nonzeroWo = []
#    n_cp = 0
#    for n in [100, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]:
#        num_orth = max(n_sz - n_cp, 0)
#        so2pls4 = O2PLS(num_comp=[num_comp, num_orth, 0])
#        alg = so2pls4.get_algorithm()
#        alg.set_tolerance(tol)
#        alg.set_max_iter(miter)
#        alg.set_prox_op(prox_ops.L0_by_count([n, n], [100, 100],
#                                             normaliser=[norm, norm]))
#        so2pls4.fit(X, Y)
#        Yhat4 = so2pls4.predict(X)
#        Yhat4 = preprocY.revert(Yhat4)
#        SSYdiff4 = np.sum((Yorig - Yhat4) ** 2)
#        R2Yhat4 = 1 - (SSYdiff4 / SSY)
#        nonzeroW.append(np.count_nonzero(so2pls4.W[:, 0]))
#        if so2pls4.Wo.shape[1] > 0:
#            nonzeroWo.append(np.count_nonzero(so2pls4.Wo[:, 0]))
#        else:
#            nonzeroWo.append(0)
#        utils.debug("sO2PLS: n = %3d, R2Yhat = %.6f, num_comp = %d, " \
#                    "nonzeroW: %2d, nonzeroWo: %2d" \
#                    % (n, R2Yhat4, num_comp, nonzeroW[-1], nonzeroWo[-1]))
#        n_cp += 1
#
##        assert all(x <= y for x, y in zip(np.abs(so2pls4.W)[:,0], (np.abs(so2pls4.W)[:,0])[1:]))
#        assert all(x >= y for x, y in zip(nonzeroW, nonzeroW[1:]))
##        assert all(x <= y for x, y in zip(nonzeroWo, nonzeroWo[1:]))
#    assert abs(R2Yhat4 - 1) < TOLERANCE

    # Testing agreement of different runs by using Fleiss kappa
    utils.debug()
    np.random.seed(15)
    n = 10
    X, Y = orth_matrix(n)
    X = np.hstack((X, utils.rand(n, n)))
    Y = np.hstack((Y, utils.rand(n, n)))

    preprocX = preprocess.PreprocessQueue([], X)
    preprocY = preprocess.PreprocessQueue([], Y)
    if center:
        preprocX.push(preprocess.Center())
        preprocY.push(preprocess.Center())
    if scale:
        preprocX.push(preprocess.Scale())
        preprocY.push(preprocess.Scale())
    X = preprocX.process(X)
    Y = preprocY.process(Y)

    num_comp = 1
    nonzero = []
    for l in [0, 0.1, 0.2, 0.3]:
        num = 10
        A = utils.zeros(X.shape[1], num)
        B = utils.zeros(Y.shape[1], num)
        for run in xrange(num):
            spls1 = models.PLSR(num_comp=num_comp)
#            alg = spls1.get_algorithm()
            spls1.set_tolerance(tol)
            spls1.set_max_iter(miter)
            spls1.set_prox_op(prox_ops.L1(l, l, normaliser=[utils.norm,
                                                            utils.normI]))
            spls1.set_start_vector(start_vectors.RandomStartVector())
            spls1.fit(X, Y)
            Yhat1 = spls1.predict(X)
            Yhat1 = preprocY.revert(Yhat1)
            SSYdiff1 = np.sum((Yorig - Yhat1) ** 2)
            R2Yhat1 = 1 - (SSYdiff1 / SSY)
            nonzero.append(np.count_nonzero(spls1.W))
            utils.debug("sPLS: l = %.2f, R2Yhat = %.6f, num_comp = %d, " \
                        "nonzero: %d" % (l, R2Yhat1, num_comp, nonzero[-1]))

            A[:, run] = np.abs(spls1.W[:, 0])
            B[:, run] = np.abs(spls1.C[:, 0])

        A[A > 0.01] = 1
        B[B > 0.01] = 1
        kappaA = fleiss_kappa(A, 2)
        kappaB = fleiss_kappa(B, 2)
        utils.debug("Kappa X:", kappaA)
        utils.debug("Kappa Y:", kappaB)
        assert kappaA > 0.1
        assert kappaB > 0.1

    utils.debug()
    num_comp = 1
    nonzeroW = []
    nonzeroWo = []
    for l in [0, 0.1, 0.2, 0.3]:
        num = 10
        A = utils.zeros(X.shape[1], num)
        B = utils.zeros(Y.shape[1], num)
        Ao = utils.zeros(X.shape[1], num)
        Bo = utils.zeros(Y.shape[1], num)
        for run in xrange(num):
            so2pls1 = models.O2PLS(num_comp=[num_comp, 2, 2])
            so2pls1.set_tolerance(tol)
            so2pls1.set_max_iter(miter)
            so2pls1.set_prox_op(prox_ops.L1([l, l], [l, l]))
            so2pls1.set_start_vector(start_vectors.RandomStartVector())
            so2pls1.fit(X, Y)
            Yhat1 = so2pls1.predict(X)
            Yhat1 = preprocY.revert(Yhat1)
            SSYdiff1 = np.sum((Yorig - Yhat1) ** 2)
            R2Yhat1 = 1 - (SSYdiff1 / SSY)
            nonzeroW.append(np.count_nonzero(so2pls1.W[:, 0]))
            nonzeroWo.append(np.count_nonzero(so2pls1.Wo[:, [0]]))
            utils.debug("O2PLS: l = %.2f, R2Yhat = %.6f, num_comp = %d, " \
                  "nonzeroW: %d, nonzeroWo: %d" \
                  % (l, R2Yhat1, num_comp, nonzeroW[-1], nonzeroWo[-1]))

            A[:, run] = np.abs(so2pls1.W[:, 0])
            B[:, run] = np.abs(so2pls1.C[:, 0])
            Ao[:, run] = np.abs(so2pls1.Wo[:, 0])
            Bo[:, run] = np.abs(so2pls1.Co[:, 0])

        A[A > 0] = 1
        B[B > 0] = 1
        Ao[Ao > 0] = 1
        Bo[Bo > 0] = 1
        kappaA = fleiss_kappa(A, 2)
        kappaB = fleiss_kappa(B, 2)
        kappaAo = fleiss_kappa(Ao, 2)
        kappaBo = fleiss_kappa(Bo, 2)
        utils.debug("Kappa X:", kappaA)
        utils.debug("Kappa Y:", kappaB)
        utils.debug("Kappa Xo:", kappaAo)
        utils.debug("Kappa Yo:", kappaBo)
        assert kappaA > 0.1
        assert kappaB > 0.1
        assert kappaAo > 0.1
        assert kappaBo > 0.1


if __name__ == "__main__":

    test()
