# -*- coding: utf-8 -*-
"""
Created on Thu May 23 09:30:03 2013

TODO: Add linear regression and ridge regression.

@author: Tommy Löfstedt
"""

import numpy as np

import structured.preprocess as preprocess
import structured.models as models
import structured.utils as utils
from structured.utils.testing import assert_array_almost_equal

from sklearn.datasets import load_linnerud
from sklearn.pls import PLSRegression
from sklearn.pls import PLSCanonical


def test():

    d = load_linnerud()
    Xorig = d.data
    Yorig = d.target
    tol = 5e-12
    miter = 1500
    num_comp = 2
    SSY = np.sum(Yorig ** 2)
#    SSX = np.sum(Xorig ** 2)
#    center = True
    scale = False

    if scale:
        preprocX = preprocess.PreprocessQueue([preprocess.Center(),
                                               preprocess.Scale()], Xorig)
        preprocY = preprocess.PreprocessQueue([preprocess.Center(),
                                               preprocess.Scale()], Yorig)
    else:
        preprocX = preprocess.PreprocessQueue([preprocess.Center()], Xorig)
        preprocY = preprocess.PreprocessQueue([preprocess.Center()], Yorig)

    # Predict using PLSRegression (reference implementation)
    X = Xorig.copy()
    Y = Yorig.copy()
    pls1 = PLSRegression(n_components=num_comp, scale=scale,
                 tol=tol, max_iter=miter, copy=True)
    pls1.fit(X, Y)
    Yhat1 = pls1.predict(X)

    SSYdiff1 = np.sum((Y - Yhat1) ** 2.0)
    utils.debug("PLSRegression: R2Yhat = %.4f" % (1.0 - (SSYdiff1 / SSY)))

    # Predict using PLSR
    X = preprocX.process(Xorig)
    Y = preprocY.process(Yorig)
    pls3 = models.PLSR(num_comp=num_comp)
    pls3.set_max_iter(miter)
    pls3.set_tolerance(tol)

    pls3.fit(X, Y)
    Yhat3 = pls3.predict(X)
    Yhat3 = preprocY.revert(Yhat3)

    SSYdiff3 = np.sum((Yorig - Yhat3) ** 2.0)
    utils.debug("PLSR         : R2Yhat = %.4f" % (1.0 - (SSYdiff3 / SSY)))

    assert abs(SSYdiff1 - SSYdiff3) < 0.00005

    assert_array_almost_equal(Yhat1, Yhat3, decimal=5,
            err_msg="PLSR gives wrong prediction")

    # Predict using sklearn.PLSCanonical
    X = Xorig.copy()
    Y = Yorig.copy()
    pls2 = PLSCanonical(n_components=num_comp, scale=scale,
                        tol=tol, max_iter=miter, copy=True)
    pls2.fit(X, Y)
    Yhat2 = pls2.predict(X)

    SSYdiff2 = np.sum((Yorig - Yhat2) ** 2.0)
    utils.debug("PLSCanonical : R2Yhat = %.4f" % (1.0 - (SSYdiff2 / SSY)))

    # Predict using PLSC
    X = preprocX.process(Xorig)
    Y = preprocY.process(Yorig)
    pls4 = models.PLSC(num_comp=num_comp)
    pls4.set_max_iter(miter)
    pls4.set_tolerance(tol)
    pls4.fit(X, Y)

    Yhat4 = pls4.predict(X)
    Yhat4 = preprocY.revert(Yhat4)

    SSYdiff4 = np.sum((Yorig - Yhat4) ** 2.0)
    utils.debug("PLSC         : R2Yhat = %.4f" % (1.0 - (SSYdiff4 / SSY)))

    # Predict using TuckerFactorAnalysis
    X = preprocX.process(Xorig)
    Y = preprocY.process(Yorig)
    plsTFA = models.TuckerFactorAnalysis(num_comp=num_comp)
    plsTFA.set_max_iter(miter)
    plsTFA.set_tolerance(tol)
    plsTFA.fit(X, Y)

    YhatTFA = plsTFA.predict(X)
    YhatTFA = preprocY.revert(YhatTFA)

    SSYdiffTFA = np.sum((Yorig - YhatTFA) ** 2.0)
    utils.debug("TuckerIBFA   : R2Yhat = %.4f" % (1.0 - (SSYdiffTFA / SSY)))

    # Predict using O2PLS
    X = preprocX.process(Xorig)
    Y = preprocY.process(Yorig)
    pls5 = models.O2PLS(num_comp=[num_comp, 1, 0])
    pls5.set_max_iter(miter)
    pls5.set_tolerance(tol)
    pls5.fit(X, Y)

    Yhat5 = pls5.predict(X)
    Yhat5 = preprocY.revert(Yhat5)

    SSYdiff5 = np.sum((Yorig - Yhat5) ** 2.0)
    utils.debug("O2PLS X-Y    : R2Yhat = %.4f" % (1.0 - (SSYdiff5 / SSY)))

    assert SSYdiff2 > SSYdiff4
    assert SSYdiff2 > SSYdiff5

    # Make sure O2PLS is symmetric!
    X = preprocX.process(Xorig)
    Y = preprocY.process(Yorig)
    pls6 = models.O2PLS(num_comp=[num_comp, 0, 1])
    pls6.set_max_iter(miter)
    pls6.set_tolerance(tol)
    pls6.fit(Y, X)

    Yhat6 = pls6.predict(Y=X)
    Yhat6 = preprocY.revert(Yhat6)

    pls5.W, pls6.C = utils.direct(pls5.W, pls6.C, compare=True)
    assert_array_almost_equal(pls5.W, pls6.C, decimal=5,
            err_msg="O2PLS is not symmetic")
    pls5.T, pls6.U = utils.direct(pls5.T, pls6.U, compare=True)
    assert_array_almost_equal(pls5.T, pls6.U, decimal=5,
            err_msg="O2PLS is not symmetic")
    pls5.P, pls6.Q = utils.direct(pls5.P, pls6.Q, compare=True)
    assert_array_almost_equal(pls5.P, pls6.Q, decimal=5,
            err_msg="O2PLS is not symmetic")

    pls5.C, pls6.W = utils.direct(pls5.C, pls6.W, compare=True)
    assert_array_almost_equal(pls5.C, pls6.W, decimal=5,
            err_msg="O2PLS is not symmetic")
    pls5.U, pls6.T = utils.direct(pls5.U, pls6.T, compare=True)
    assert_array_almost_equal(pls5.U, pls6.T, decimal=5,
            err_msg="O2PLS is not symmetic")
    pls5.Q, pls6.P = utils.direct(pls5.Q, pls6.P, compare=True)
    assert_array_almost_equal(pls5.Q, pls6.P, decimal=5,
            err_msg="O2PLS is not symmetic")

    assert_array_almost_equal(pls5.Bx, pls6.By, decimal=5,
            err_msg="O2PLS is not symmetic")
    assert_array_almost_equal(pls5.By, pls6.Bx, decimal=5,
            err_msg="O2PLS is not symmetic")

    SSYdiff6 = np.sum((Yorig - Yhat6) ** 2.0)
    utils.debug("O2PLS Y-X    : R2Yhat = %.4f" % (1.0 - (SSYdiff6 / SSY)))

    assert abs(SSYdiff6 - SSYdiff5) < utils.TOLERANCE * 10.0


if __name__ == "__main__":

    test()
