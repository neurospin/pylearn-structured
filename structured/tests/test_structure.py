# -*- coding: utf-8 -*-
"""
Created on Thu May 23 09:37:48 2013

TODO: Fix the proximal operators. Normalisation?

@author: Tommy Löfstedt
@email: tommy.loefstedt@cea.fr
"""

import numpy as np
import structured.preprocess as preprocess
import structured.models as models

import structured.utils as utils
from structured.utils.testing import assert_array_almost_equal
from structured.utils.testing import orth_matrix, fleiss_kappa

import matplotlib.pyplot as plot
import matplotlib.cm as cm


def test():

    np.random.seed(42)

    x = np.arange(-8, 8, 1)
    y = np.arange(-8, 8, 1)
    nrows, ncols = len(x), len(y)
    px = ncols
    py = nrows
    pz = 1
    p = nrows * ncols
    n = 200
    mask = np.zeros((nrows, ncols))
    beta = np.zeros((nrows, ncols))
    for i in xrange(nrows):
        for j in xrange(ncols):
#            if (((x[i] - 3) ** 2 + (y[j] - 3) ** 2 > 8) &
#                ((x[i] - 3) ** 2 + (y[j] - 3) ** 2 < 25)):
#                mask[i, j] = 1

            if ((x[i] - 3) ** 2 + (y[j] - 3) ** 2 < 25):
                mask[i, j] = 1

            if (((x[i] + 1) ** 2 + (y[j] - 5) ** 2 > 5) &
                ((x[i] + 1) ** 2 + (y[j] - 5) ** 2 < 16)):
                mask[i, j] = 1

            if (y[j] > 1) & (x[i] > 3) & (y[j] + x[i] < 10):
                beta[i, j] = (x[i] - 3) ** 2 + (y[j] - 3) ** 2 + 25

#    beta = np.random.rand(nrows, ncols)
#    beta = np.sort(np.abs(beta), axis=0)
#    beta = np.sort(np.abs(beta), axis=1)

    beta1D = beta.reshape((p, 1))
    mask1D = mask.reshape((p, 1))

    r = 0.0
    u = r * np.random.randn(p, p)
    u += (1.0 - r) * np.eye(p, p)
    sigma = np.dot(u.T, u)
    mean = np.zeros(p)

    X = np.random.multivariate_normal(mean, sigma, n)
    y = np.dot(X, beta1D)
    betaOLS = np.dot(np.linalg.pinv(X), y)

    eps = 0.01
    maxit = 10000
    mu = 0.001

    # Linear regression
    pgm = models.LinearRegression()
    pgm.set_max_iter(maxit)
    pgm.set_tolerance(eps)
    pgm.fit(X, y)
#    f = pgm.get_algorithm().f

#    plot.subplot(4, 4, 1)
#    plot.plot(beta1D[:, 0], '-', pgm.beta[:, 0], '*')
#    plot.title("PGM Linear regression (it=%d, err=%f)" % (len(f), f[-1]))
#
#    plot.subplot(4, 4, 2)
#    plot.imshow(np.reshape(pgm.beta, (pz, py, px))[0, :, :],
#                interpolation='nearest', cmap=cm.gist_rainbow)

    egm = models.EGMRidgeRegression(utils.TOLERANCE)
    egm.set_max_iter(maxit)
    egm.set_tolerance(eps)
    egm.fit(X, y)
#    f = egm.get_algorithm().f

#    plot.subplot(4, 4, 3)
#    plot.plot(beta1D[:, 0], '-', egm.beta[:, 0], '*')
#    plot.title("EGM Linear regression (it=%d, err=%f)" % (len(f), f[-1]))
#
#    plot.subplot(4, 4, 4)
#    plot.imshow(np.reshape(egm.beta, (pz, py, px))[0, :, :],
#                interpolation='nearest', cmap=cm.gist_rainbow)

    lr_pgm_ss = np.sum(pgm.beta ** 2.0)
    utils.debug("SS beta: ", lr_pgm_ss)
    lr_egm_ss = np.sum(egm.beta ** 2.0)
    utils.debug("SS beta: ", lr_egm_ss)
    assert lr_egm_ss < lr_pgm_ss

    lr_true_error_pgm = np.sum((beta1D - pgm.beta) ** 2.0) \
                        / np.sum(beta1D ** 2.0)
    utils.debug("True difference PGM:", lr_true_error_pgm)
    lr_true_error_egm = np.sum((beta1D - egm.beta) ** 2.0) \
                        / np.sum(beta1D ** 2.0)
    utils.debug("True difference EGM:", lr_true_error_egm)
    assert lr_true_error_pgm < 0.25
    assert lr_true_error_egm < 0.25
    assert abs(lr_true_error_pgm - lr_true_error_egm) < 0.005

    lr_ols_diff_pgm = np.sum((betaOLS - pgm.beta) ** 2.0) \
                        / np.sum(betaOLS ** 2.0)
    utils.debug("Difference to OLS EGM:", lr_ols_diff_pgm)
    lr_ols_diff_egm = np.sum((betaOLS - egm.beta) ** 2.0) \
                        / np.sum(betaOLS ** 2.0)
    utils.debug("Difference to OLS PGM:", lr_ols_diff_egm)
    assert lr_ols_diff_pgm < 0.00005
    assert lr_ols_diff_egm < 0.00005

    lr_diff_egm_pgm = np.sum((pgm.beta - egm.beta) ** 2.0) \
                        / np.sum(pgm.beta ** 2.0)
    utils.debug("Difference EGM -- PGM:", lr_diff_egm_pgm)
    lr_diff_pgm_egm = np.sum((pgm.beta - egm.beta) ** 2.0) \
                        / np.sum(egm.beta ** 2.0)
    utils.debug("Difference PGM -- EGM:", lr_diff_pgm_egm)
    assert lr_diff_egm_pgm < 0.00005
    assert lr_diff_pgm_egm < 0.00005

    lr_r2_pgm = 1.0 - np.sum((y - np.dot(X, pgm.beta)) ** 2.0) \
                        / np.sum(y ** 2.0)
    utils.debug("R2 PGM:", lr_r2_pgm)
    lr_r2_egm = 1.0 - np.sum((y - np.dot(X, egm.beta)) ** 2.0) \
                        / np.sum(y ** 2.0)
    utils.debug("R2 EGM:", lr_r2_egm)
    assert abs(lr_r2_pgm - 1.0) < 0.00005
    assert abs(lr_r2_egm - 1.0) < 0.00005

    utils.debug("")

#    # LASSO (Linear regression + L1 penalty)
#    l = 1.0
#    pgm = models.Lasso(l)
#    pgm.set_max_iter(maxit)
#    pgm.set_tolerance(eps)
#    pgm.fit(X, y)
##    f = pgm.get_algorithm().f
#
##    plot.subplot(4, 4, 5)
##    plot.plot(beta1D[:, 0], '-', pgm.beta[:, 0], '*')
##    plot.title("PGM LASSO (%f, %f)" % (len(f), f[-1]))
##
##    plot.subplot(4, 4, 6)
##    plot.imshow(np.reshape(pgm.beta, (pz, py, px))[0, :, :],
##                interpolation='nearest', cmap=cm.gist_rainbow)
#
#    egm = models.EGMLinearRegressionL1L2(l, 0.001, p, mu=mu)
#    egm.set_max_iter(maxit)
#    egm.set_tolerance(eps)
#    egm.fit(X, y)
##    f = egm.get_algorithm().f
#
##    plot.subplot(4, 4, 7)
##    plot.plot(beta1D[:, 0], '-', egm.beta[:, 0], '*')
##    plot.title("EGM LASSO (%f, %f)" % (len(f), f[-1]))
##
##    plot.subplot(4, 4, 8)
##    plot.imshow(np.reshape(egm.beta, (pz, py, px))[0, :, :],
##                interpolation='nearest', cmap=cm.gist_rainbow)
#
#    lasso_pgm_ss = np.sum(pgm.beta ** 2.0)
#    utils.debug("SS beta: ", lasso_pgm_ss)
#    lasso_egm_ss = np.sum(egm.beta ** 2.0)
#    utils.debug("SS beta: ", lasso_egm_ss)
#    assert lasso_egm_ss < lasso_pgm_ss
#    assert lasso_egm_ss > lr_egm_ss
#    assert lasso_pgm_ss > lr_pgm_ss
#
#    lasso_true_error_pgm = np.sum((beta1D - pgm.beta) ** 2.0) \
#                        / np.sum(beta1D ** 2.0)
#    utils.debug("True difference PGM:", lasso_true_error_pgm)
#    lasso_true_error_egm = np.sum((beta1D - egm.beta) ** 2.0) \
#                        / np.sum(beta1D ** 2.0)
#    utils.debug("True difference EGM:", lasso_true_error_egm)
#    assert lasso_true_error_pgm < 0.05
#    assert lasso_true_error_egm < 0.05
#    assert abs(lasso_true_error_pgm - lasso_true_error_egm) < 0.005
#    assert lasso_true_error_pgm < lr_true_error_pgm
#    assert lasso_true_error_egm < lr_true_error_egm
#
#    lasso_ols_diff_pgm = np.sum((betaOLS - pgm.beta) ** 2.0) \
#                        / np.sum(betaOLS ** 2.0)
#    utils.debug("Difference to OLS EGM:", lasso_ols_diff_pgm)
#    lasso_ols_diff_egm = np.sum((betaOLS - egm.beta) ** 2.0) \
#                        / np.sum(betaOLS ** 2.0)
#    utils.debug("Difference to OLS PGM:", lasso_ols_diff_egm)
#    assert lasso_ols_diff_pgm < 0.31
#    assert lasso_ols_diff_egm < 0.30
#    assert lasso_ols_diff_pgm > lr_ols_diff_pgm
#    assert lasso_ols_diff_egm > lr_ols_diff_egm
#
#    lasso_diff_egm_pgm = np.sum((pgm.beta - egm.beta) ** 2.0) \
#                        / np.sum(pgm.beta ** 2.0)
#    utils.debug("Difference EGM -- PGM:", lasso_diff_egm_pgm)
#    lasso_diff_pgm_egm = np.sum((pgm.beta - egm.beta) ** 2.0) \
#                        / np.sum(egm.beta ** 2.0)
#    utils.debug("Difference PGM -- EGM:", lasso_diff_pgm_egm)
#    assert lasso_diff_egm_pgm < 0.005
#    assert lasso_diff_pgm_egm < 0.005
#    assert lasso_diff_egm_pgm > lr_diff_egm_pgm
#    assert lasso_diff_pgm_egm > lr_diff_pgm_egm
#
#    lasso_r2_pgm = 1.0 - np.sum((y - np.dot(X, pgm.beta)) ** 2.0) \
#                        / np.sum(y ** 2.0)
#    utils.debug("R2 PGM:", lasso_r2_pgm)
#    lasso_r2_egm = 1.0 - np.sum((y - np.dot(X, egm.beta)) ** 2.0) \
#                        / np.sum(y ** 2.0)
#    utils.debug("R2 EGM:", lasso_r2_egm)
#    assert abs(lasso_r2_pgm - 1.0) < 0.00005
#    assert abs(lasso_r2_egm - 1.0) < 0.00005
#    assert lasso_r2_pgm < lr_r2_pgm
#    assert lasso_r2_egm < lr_r2_egm
#
#    utils.debug("")

#    # Elastic Net (Linear regression + Elastic Net)
#    l = 0.8
#    pgm = models.ElasticNet(l)
#    pgm.set_max_iter(maxit)
#    pgm.set_tolerance(eps)
#    pgm.fit(X, y)
##    f = pgm.get_algorithm().f
#
##    plot.subplot(2, 2, 1)
##    plot.plot(beta1D[:, 0], '-', pgm.beta[:, 0], '*')
##    plot.title("Elastic Net (%d, %f)" % (len(f), f[-1]))
##
##    plot.subplot(2, 2, 2)
##    plot.imshow(np.reshape(pgm.beta, (pz, py, px))[0, :, :],
##                interpolation='nearest', cmap=cm.gist_rainbow)
#
#    egm = models.EGMElasticNet(l, p, mu=mu)
#    egm.set_max_iter(maxit)
#    egm.set_tolerance(eps)
#    egm.fit(X, y)
##    f = egm.get_algorithm().f
#
##    plot.subplot(2, 2, 3)
##    plot.plot(beta1D[:, 0], '-', egm.beta[:, 0], '*')
##    plot.title("Elastic Net (%d, %f)" % (len(f), f[-1]))
##
##    plot.subplot(2, 2, 4)
##    plot.imshow(np.reshape(egm.beta, (pz, py, px))[0, :, :],
##                interpolation='nearest', cmap=cm.gist_rainbow)
#
#    en_pgm_ss = np.sum(pgm.beta ** 2.0)
#    utils.debug("SS beta: ", en_pgm_ss)
#    en_egm_ss = np.sum(egm.beta ** 2.0)
#    utils.debug("SS beta: ", en_egm_ss)
#    assert en_pgm_ss < en_egm_ss
#    assert en_egm_ss > lr_egm_ss
#    assert en_pgm_ss > lr_pgm_ss
##    assert en_egm_ss < lasso_egm_ss
##    assert en_pgm_ss < lasso_pgm_ss
#
#    en_true_error_pgm = np.sum((beta1D - pgm.beta) ** 2.0) \
#                        / np.sum(beta1D ** 2.0)
#    utils.debug("True difference PGM:", en_true_error_pgm)
#    en_true_error_egm = np.sum((beta1D - egm.beta) ** 2.0) \
#                        / np.sum(beta1D ** 2.0)
#    utils.debug("True difference EGM:", en_true_error_egm)
#    assert en_true_error_pgm < 0.03
#    assert en_true_error_egm < 0.02
#    assert abs(en_true_error_pgm - en_true_error_egm) < 0.05
#    assert en_true_error_pgm < lr_true_error_pgm
#    assert en_true_error_egm < lr_true_error_egm
##    assert en_true_error_pgm > lasso_true_error_pgm
##    assert en_true_error_egm > lasso_true_error_egm
#
#    en_ols_diff_pgm = np.sum((betaOLS - pgm.beta) ** 2.0) \
#                        / np.sum(betaOLS ** 2.0)
#    utils.debug("Difference to OLS EGM:", en_ols_diff_pgm)
#    en_ols_diff_egm = np.sum((betaOLS - egm.beta) ** 2.0) \
#                        / np.sum(betaOLS ** 2.0)
#    utils.debug("Difference to OLS PGM:", en_ols_diff_egm)
#    assert en_ols_diff_pgm < 0.14
#    assert en_ols_diff_egm < 0.18
#    assert en_ols_diff_pgm > lr_ols_diff_pgm
#    assert en_ols_diff_egm > lr_ols_diff_pgm
##    assert en_ols_diff_pgm < lasso_ols_diff_pgm
##    assert en_ols_diff_egm < lasso_ols_diff_pgm
#
#    en_diff_egm_pgm = np.sum((pgm.beta - egm.beta) ** 2.0) \
#                        / np.sum(pgm.beta ** 2.0)
#    utils.debug("Difference EGM -- PGM:", en_diff_egm_pgm)
#    en_diff_pgm_egm = np.sum((pgm.beta - egm.beta) ** 2.0) \
#                        / np.sum(egm.beta ** 2.0)
#    utils.debug("Difference PGM -- EGM:", en_diff_pgm_egm)
#    assert en_diff_egm_pgm < 0.005
#    assert en_diff_pgm_egm < 0.005
#    assert en_diff_egm_pgm > lr_diff_egm_pgm
#    assert en_diff_pgm_egm > lr_diff_pgm_egm
##    assert en_diff_egm_pgm < lasso_diff_egm_pgm
##    assert en_diff_pgm_egm < lasso_diff_pgm_egm
#
#    en_r2_pgm = 1.0 - np.sum((y - np.dot(X, pgm.beta)) ** 2.0) \
#                        / np.sum(y ** 2.0)
#    utils.debug("R2 PGM:", en_r2_pgm)
#    en_r2_egm = 1.0 - np.sum((y - np.dot(X, egm.beta)) ** 2.0) \
#                        / np.sum(y ** 2.0)
#    utils.debug("R2 EGM:", en_r2_egm)
#    assert abs(en_r2_pgm - 1.0) < 0.00005
#    assert abs(en_r2_egm - 1.0) < 0.00005
#    assert en_r2_pgm < lr_r2_pgm
#    assert en_r2_egm < lr_r2_egm
##    assert en_r2_pgm < lasso_r2_pgm
##    assert en_r2_egm < lasso_r2_egm
#
#    utils.debug("")

#    # Linear regression + Total variation penalty
#    gamma = 1.0
#    pgm = models.LinearRegressionTV(gamma, (pz, py, px), mu=mu)
#    pgm.set_max_iter(maxit)
#    pgm.set_tolerance(eps)
#    pgm.fit(X, y)
#    f = pgm.get_algorithm().f
#
#    plot.subplot(4, 4, 9)
#    plot.plot(beta1D[:, 0], '-', pgm.beta[:, 0], '*')
#    plot.title("Linear regression + TV (%f, %f)" % (len(f), f[-1]))
#
#    plot.subplot(4, 4, 10)
#    plot.imshow(np.reshape(pgm.beta, (pz, py, px))[0, :, :],
#                interpolation='nearest', cmap=cm.gist_rainbow)
#
#    egm = models.EGMRidgeRegressionTV(0.00035, gamma, (pz, py, px))
#    egm.set_max_iter(maxit)
#    egm.set_tolerance(eps)
#    egm.fit(X, y)
#    f = egm.get_algorithm().f
#
#    plot.subplot(4, 4, 11)
#    plot.plot(beta1D[:, 0], '-', egm.beta[:, 0], '*')
#    plot.title("Linear regression + TV (%f, %f)" % (len(f), f[-1]))
#
#    plot.subplot(4, 4, 12)
#    plot.imshow(np.reshape(egm.beta, (pz, py, px))[0, :, :],
#                interpolation='nearest', cmap=cm.gist_rainbow)
#
#    tv_pgm_ss = np.sum(pgm.beta ** 2.0)
#    utils.debug("SS beta: ", tv_pgm_ss)
#    tv_egm_ss = np.sum(egm.beta ** 2.0)
#    utils.debug("SS beta: ", tv_egm_ss)
#    assert tv_pgm_ss < tv_egm_ss
#    assert tv_pgm_ss > lr_pgm_ss
#    assert tv_egm_ss > lr_egm_ss
#
#    tv_true_error_pgm = np.sum((beta1D - pgm.beta) ** 2.0) \
#                        / np.sum(beta1D ** 2.0)
#    utils.debug("True difference PGM:", tv_true_error_pgm)
#    tv_true_error_egm = np.sum((beta1D - egm.beta) ** 2.0) \
#                        / np.sum(beta1D ** 2.0)
#    utils.debug("True difference EGM:", tv_true_error_egm)
#    assert tv_true_error_pgm < 0.08
#    assert tv_true_error_egm < 0.07
#    assert abs(tv_true_error_pgm - tv_true_error_egm) < 0.05
#    assert tv_true_error_pgm < lr_true_error_pgm
#    assert tv_true_error_egm < lr_true_error_egm
##    assert tv_true_error_pgm > lasso_true_error_pgm
##    assert tv_true_error_egm > lasso_true_error_egm
##    assert tv_true_error_pgm > en_true_error_pgm
##    assert tv_true_error_egm > en_true_error_egm
#
#    tv_ols_diff_pgm = np.sum((betaOLS - pgm.beta) ** 2.0) \
#                        / np.sum(betaOLS ** 2.0)
#    utils.debug("Difference to OLS EGM:", tv_ols_diff_pgm)
#    tv_ols_diff_egm = np.sum((betaOLS - egm.beta) ** 2.0) \
#                        / np.sum(betaOLS ** 2.0)
#    utils.debug("Difference to OLS PGM:", tv_ols_diff_egm)
#    assert tv_ols_diff_pgm < 0.07
#    assert tv_ols_diff_egm < 0.33
#    assert tv_ols_diff_pgm > lr_ols_diff_pgm
#    assert tv_ols_diff_egm > lr_ols_diff_pgm
#
#    tv_diff_egm_pgm = np.sum((pgm.beta - egm.beta) ** 2.0) \
#                        / np.sum(pgm.beta ** 2.0)
#    utils.debug("Difference EGM -- PGM:", tv_diff_egm_pgm)
#    tv_diff_pgm_egm = np.sum((pgm.beta - egm.beta) ** 2.0) \
#                        / np.sum(egm.beta ** 2.0)
#    utils.debug("Difference PGM -- EGM:", tv_diff_pgm_egm)
#    assert tv_diff_egm_pgm < 0.14
#    assert tv_diff_pgm_egm < 0.11
#    assert tv_diff_egm_pgm > lr_diff_egm_pgm
#    assert tv_diff_pgm_egm > lr_diff_pgm_egm
##    assert tv_diff_egm_pgm > lasso_diff_egm_pgm
##    assert tv_diff_pgm_egm > lasso_diff_pgm_egm
##    assert tv_diff_egm_pgm > en_diff_egm_pgm
##    assert tv_diff_pgm_egm > en_diff_pgm_egm
#
#    tv_r2_pgm = 1.0 - np.sum((y - np.dot(X, pgm.beta)) ** 2.0) \
#                        / np.sum(y ** 2.0)
#    utils.debug("R2 PGM:", tv_r2_pgm)
#    tv_r2_egm = 1.0 - np.sum((y - np.dot(X, egm.beta)) ** 2.0) \
#                        / np.sum(y ** 2.0)
#    utils.debug("R2 EGM:", tv_r2_egm)
#    assert abs(tv_r2_pgm - 1.0) < 0.00005
#    assert abs(tv_r2_egm - 1.0) < 0.00005
#    assert tv_r2_pgm < lr_r2_pgm
#    assert tv_r2_egm < lr_r2_egm
##    assert tv_r2_pgm < lasso_r2_pgm
##    assert tv_r2_egm < lasso_r2_egm
#
#    utils.debug("")

    # Lasso + Total variation penalty (Linear regression + L1 + TV)
    l = 0.5
    gamma = 1.0
    pgm = models.LinearRegressionL1TV(l, gamma, (pz, py, px), mu=mu)
    pgm.set_max_iter(maxit)
    pgm.set_tolerance(eps)
    pgm.fit(X, y)
    f = pgm.get_algorithm().f

    plot.subplot(4, 4, 13)
    plot.plot(beta1D[:, 0], '-', pgm.beta[:, 0], '*')
    plot.title("Linear regression + L1 + TV (%f, %f)" % (len(f), f[-1]))

    plot.subplot(4, 4, 14)
    plot.imshow(np.reshape(pgm.beta, (pz, py, px))[0, :, :],
                interpolation='nearest', cmap=cm.gist_rainbow)

    egm = models.EGMLinearRegressionL1L2TV(l, 0.0008, gamma, (pz, py, px))
    egm.set_max_iter(maxit)
    egm.set_tolerance(eps)
    egm.fit(X, y)
    f = egm.get_algorithm().f

    plot.subplot(4, 4, 15)
    plot.plot(beta1D[:, 0], '-', egm.beta[:, 0], '*')
    plot.title("Linear regression + L1 + TV (%f, %f)" % (len(f), f[-1]))

    plot.subplot(4, 4, 16)
    plot.imshow(np.reshape(egm.beta, (pz, py, px))[0, :, :],
                interpolation='nearest', cmap=cm.gist_rainbow)

    tv_pgm_ss = np.sum(pgm.beta ** 2.0)
    utils.debug("SS beta: ", tv_pgm_ss)
    tv_egm_ss = np.sum(egm.beta ** 2.0)
    utils.debug("SS beta: ", tv_egm_ss)
#    assert tv_pgm_ss < tv_egm_ss
#    assert tv_pgm_ss > lr_pgm_ss
#    assert tv_egm_ss > lr_egm_ss

    tv_true_error_pgm = np.sum((beta1D - pgm.beta) ** 2.0) \
                        / np.sum(beta1D ** 2.0)
    utils.debug("True difference PGM:", tv_true_error_pgm)
    tv_true_error_egm = np.sum((beta1D - egm.beta) ** 2.0) \
                        / np.sum(beta1D ** 2.0)
    utils.debug("True difference EGM:", tv_true_error_egm)
#    assert tv_true_error_pgm < 0.08
#    assert tv_true_error_egm < 0.07
#    assert abs(tv_true_error_pgm - tv_true_error_egm) < 0.05
#    assert tv_true_error_pgm < lr_true_error_pgm
#    assert tv_true_error_egm < lr_true_error_egm
#    assert tv_true_error_pgm > lasso_true_error_pgm
#    assert tv_true_error_egm > lasso_true_error_egm
#    assert tv_true_error_pgm > en_true_error_pgm
#    assert tv_true_error_egm > en_true_error_egm

    tv_ols_diff_pgm = np.sum((betaOLS - pgm.beta) ** 2.0) \
                        / np.sum(betaOLS ** 2.0)
    utils.debug("Difference to OLS EGM:", tv_ols_diff_pgm)
    tv_ols_diff_egm = np.sum((betaOLS - egm.beta) ** 2.0) \
                        / np.sum(betaOLS ** 2.0)
    utils.debug("Difference to OLS PGM:", tv_ols_diff_egm)
#    assert tv_ols_diff_pgm < 0.07
#    assert tv_ols_diff_egm < 0.33
#    assert tv_ols_diff_pgm > lr_ols_diff_pgm
#    assert tv_ols_diff_egm > lr_ols_diff_pgm

    tv_diff_egm_pgm = np.sum((pgm.beta - egm.beta) ** 2.0) \
                        / np.sum(pgm.beta ** 2.0)
    utils.debug("Difference EGM -- PGM:", tv_diff_egm_pgm)
    tv_diff_pgm_egm = np.sum((pgm.beta - egm.beta) ** 2.0) \
                        / np.sum(egm.beta ** 2.0)
    utils.debug("Difference PGM -- EGM:", tv_diff_pgm_egm)
#    assert tv_diff_egm_pgm < 0.14
#    assert tv_diff_pgm_egm < 0.11
#    assert tv_diff_egm_pgm > lr_diff_egm_pgm
#    assert tv_diff_pgm_egm > lr_diff_pgm_egm
#    assert tv_diff_egm_pgm > lasso_diff_egm_pgm
#    assert tv_diff_pgm_egm > lasso_diff_pgm_egm
#    assert tv_diff_egm_pgm > en_diff_egm_pgm
#    assert tv_diff_pgm_egm > en_diff_pgm_egm

    tv_r2_pgm = 1.0 - np.sum((y - np.dot(X, pgm.beta)) ** 2.0) \
                        / np.sum(y ** 2.0)
    utils.debug("R2 PGM:", tv_r2_pgm)
    tv_r2_egm = 1.0 - np.sum((y - np.dot(X, egm.beta)) ** 2.0) \
                        / np.sum(y ** 2.0)
    utils.debug("R2 EGM:", tv_r2_egm)
#    assert abs(tv_r2_pgm - 1.0) < 0.00005
#    assert abs(tv_r2_egm - 1.0) < 0.00005
#    assert tv_r2_pgm < lr_r2_pgm
#    assert tv_r2_egm < lr_r2_egm
#    assert tv_r2_pgm < lasso_r2_pgm
#    assert tv_r2_egm < lasso_r2_egm

    utils.debug("")
#    # Elastic Net + Total variation penalty (Linear regression + EN + TV)
#    l = 0.8
#    gamma = 1.0
#    pgm = models.ElasticNetTV(l, gamma, (pz, py, px), mu=mus[0])
#    pgm.set_max_iter(maxit)
#    pgm.set_tolerance(eps)
#    pgm.fit(X, y)
#    f = pgm.get_algorithm().f
#    print "SS: ", np.sum(pgm.beta ** 2.0)
#
#    plot.subplot(2, 2, 1)
#    plot.plot(beta1D[:, 0], '-', pgm.beta[:, 0], '*')
#    plot.title("Elastic Net + TV (%d, %f)" % (len(f), f[-1]))
#
#    plot.subplot(2, 2, 2)
#    plot.imshow(np.reshape(pgm.beta, (pz, py, px))[0, :, :],
#                interpolation='nearest', cmap=cm.gist_rainbow)
#
#    egm = models.EGMElasticNetTV(l, gamma, (pz, py, px), mu=mus[0])
#    egm.set_max_iter(maxit)
#    egm.set_tolerance(eps)
#    egm.fit(X, y)
#    f = egm.get_algorithm().f
#    print "SS: ", np.sum(egm.beta ** 2.0)
#
#    plot.subplot(2, 2, 3)
#    plot.plot(beta1D[:, 0], '-', egm.beta[:, 0], '*')
#    plot.title("Elastic Net + TV (%d, %f)" % (len(f), f[-1]))
#
#    plot.subplot(2, 2, 4)
#    plot.imshow(np.reshape(egm.beta, (pz, py, px))[0, :, :],
#                interpolation='nearest', cmap=cm.gist_rainbow)
#
#    utils.debug("True difference PGM:",
#        np.sum((beta1D - pgm.beta) ** 2.0) / np.sum(beta1D ** 2.0))
#    utils.debug("True difference EGM:",
#        np.sum((beta1D - egm.beta) ** 2.0) / np.sum(beta1D ** 2.0))
#    utils.debug("Difference to OLS EGM:",
#        np.sum((betaOLS - pgm.beta) ** 2.0) / np.sum(betaOLS ** 2.0))
#    utils.debug("Difference to OLS PGM:",
#        np.sum((betaOLS - egm.beta) ** 2.0) / np.sum(betaOLS ** 2.0))
#    utils.debug("Difference EGM -- PGM:",
#        np.sum((pgm.beta - egm.beta) ** 2.0) / np.sum(pgm.beta ** 2.0))
#    utils.debug("Difference PGM -- EGM:",
#        np.sum((pgm.beta - egm.beta) ** 2.0) / np.sum(egm.beta ** 2.0))
#    utils.debug("R2 PGM:",
#        np.sum((y - np.dot(X, pgm.beta)) ** 2.0) / np.sum(y ** 2.0))
#    utils.debug("R2 EGM:",
#        np.sum((y - np.dot(X, egm.beta)) ** 2.0) / np.sum(y ** 2.0))

    plot.show()


if __name__ == "__main__":

    test()