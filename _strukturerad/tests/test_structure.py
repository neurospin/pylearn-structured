# -*- coding: utf-8 -*-
"""
Created on Thu May 23 09:37:48 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: TBD
"""

import numpy as np
import structured.models as models
import structured.utils as utils

from time import time


def test():

    pass
#    np.random.seed(42)
#
#    w = 6
#    x = np.arange(-w, w, 1)
#    y = np.arange(-w, w, 1)
#    nrows, ncols = len(x), len(y)
#    px = ncols
#    py = nrows
#    pz = 1
#    p = nrows * ncols
#    n = 100
#    mask = np.zeros((nrows, ncols))
#    beta = np.zeros((nrows, ncols))
#    for i in xrange(nrows):
#        for j in xrange(ncols):
##            if (((x[i] - 3) ** 2 + (y[j] - 3) ** 2 > 8) &
##                ((x[i] - 3) ** 2 + (y[j] - 3) ** 2 < 25)):
##                mask[i, j] = 1
#
##            if ((x[i] - 3) ** 2 + (y[j] - 3) ** 2 < 25):
##                mask[i, j] = 1
##
##            if (((x[i] + 1) ** 2 + (y[j] - 5) ** 2 > 5) &
##                ((x[i] + 1) ** 2 + (y[j] - 5) ** 2 < 16)):
##                mask[i, j] = 1
#
#            if (y[j] >= 0) and (x[i] >= 0) and (y[j] + x[i] < w):
#                beta[i, j] = (x[i] - 3) ** 2 + (y[j] - 3) ** 2 + 20
#
##    beta = np.random.rand(nrows, ncols)
##    beta = np.sort(np.abs(beta), axis=0)
##    beta = np.sort(np.abs(beta), axis=1)
#
#    beta1D = beta.reshape((p, 1))
#    mask1D = mask.reshape((p, 1))
#
#    r = 0.0
#    u = r * np.random.randn(p, p)
#    u += (1.0 - r) * np.eye(p, p)
#    sigma = np.dot(u.T, u)
#    mean = np.zeros(p)
#
#    X = np.random.multivariate_normal(mean, sigma, n)
#    y = np.dot(X, beta1D)
#    betaOLS = np.dot(np.linalg.pinv(X), y)
#
#    eps = 0.01
#    maxit = 2000
#    mu = 0.001
#
#    # Linear regression
#    utils.debug("Linear regression")
#    pgm = models.LinearRegression()
#    pgm.set_max_iter(maxit)
#    pgm.set_tolerance(eps)
#    pgm.fit(X, y)
##    f = pgm.get_algorithm().f
##
##    plot.subplot(2, 2, 1)
##    plot.plot(beta1D[:, 0], '-', pgm._beta[:, 0], '*')
##    plot.title("PGM Linear regression (it=%d, err=%f)" % (len(f), f[-1]))
##
##    plot.subplot(2, 2, 2)
##    plot.imshow(np.reshape(pgm._beta, (pz, py, px))[0, :, :],
##                interpolation='nearest', cmap=cm.gist_rainbow)
#
#    egm = models.EGMRidgeRegression(0.0001)
#    egm.set_max_iter(maxit)
#    egm.set_tolerance(eps)
#    egm.fit(X, y)
##    f = egm.get_algorithm().f
##
##    plot.subplot(2, 2, 3)
##    plot.plot(beta1D[:, 0], '-', egm._beta[:, 0], '*')
##    plot.title("EGM Linear regression (it=%d, err=%f)" % (len(f), f[-1]))
##
##    plot.subplot(2, 2, 4)
##    plot.imshow(np.reshape(egm._beta, (pz, py, px))[0, :, :],
##                interpolation='nearest', cmap=cm.gist_rainbow)
#
#    lr_pgm_ss = np.sum(pgm._beta ** 2.0)
#    utils.debug("SS beta: ", lr_pgm_ss)
#    lr_egm_ss = np.sum(egm._beta ** 2.0)
#    utils.debug("SS beta: ", lr_egm_ss)
#    assert lr_egm_ss < lr_pgm_ss
#
#    lr_true_error_pgm = np.sum((beta1D - pgm._beta) ** 2.0) \
#                        / np.sum(beta1D ** 2.0)
#    utils.debug("True difference PGM:", lr_true_error_pgm)
#    lr_true_error_egm = np.sum((beta1D - egm._beta) ** 2.0) \
#                        / np.sum(beta1D ** 2.0)
#    utils.debug("True difference EGM:", lr_true_error_egm)
#    assert lr_true_error_pgm < 0.31
#    assert lr_true_error_egm < 0.31
#    assert abs(lr_true_error_pgm - lr_true_error_egm) < 0.005
#
#    lr_ols_diff_pgm = np.sum((betaOLS - pgm._beta) ** 2.0) \
#                        / np.sum(betaOLS ** 2.0)
#    utils.debug("Difference to OLS EGM:", lr_ols_diff_pgm)
#    lr_ols_diff_egm = np.sum((betaOLS - egm._beta) ** 2.0) \
#                        / np.sum(betaOLS ** 2.0)
#    utils.debug("Difference to OLS PGM:", lr_ols_diff_egm)
#    assert lr_ols_diff_pgm < 0.00005
#    assert lr_ols_diff_egm < 0.00005
#
#    lr_diff_egm_pgm = np.sum((pgm._beta - egm._beta) ** 2.0) \
#                        / np.sum(pgm._beta ** 2.0)
#    utils.debug("Difference EGM -- PGM:", lr_diff_egm_pgm)
#    lr_diff_pgm_egm = np.sum((pgm._beta - egm._beta) ** 2.0) \
#                        / np.sum(egm._beta ** 2.0)
#    utils.debug("Difference PGM -- EGM:", lr_diff_pgm_egm)
#    assert lr_diff_egm_pgm < 0.00005
#    assert lr_diff_pgm_egm < 0.00005
#
#    lr_r2_pgm = 1.0 - np.sum((y - np.dot(X, pgm._beta)) ** 2.0) \
#                        / np.sum(y ** 2.0)
#    utils.debug("R2 PGM:", lr_r2_pgm)
#    lr_r2_egm = 1.0 - np.sum((y - np.dot(X, egm._beta)) ** 2.0) \
#                        / np.sum(y ** 2.0)
#    utils.debug("R2 EGM:", lr_r2_egm)
#    assert abs(lr_r2_pgm - 1.0) < 0.0000005
#    assert abs(lr_r2_egm - 1.0) < 0.0000005
#
#    utils.debug("")
#
#    # LASSO (Linear regression + L1 penalty)
#    utils.debug("LASSO")
#    l = 1.0
#    pgm = models.Lasso(l)
#    pgm.set_max_iter(maxit)
#    pgm.set_tolerance(eps)
#    pgm.fit(X, y)
##    f = pgm.get_algorithm().f
##
##    plot.subplot(2, 2, 1)
##    plot.plot(beta1D[:, 0], '-', pgm._beta[:, 0], '*')
##    plot.title("PGM LASSO (%d, %f)" % (len(f), f[-1]))
##
##    plot.subplot(2, 2, 2)
##    plot.imshow(np.reshape(pgm._beta, (pz, py, px))[0, :, :],
##                interpolation='nearest', cmap=cm.gist_rainbow)
#
#    egm = models.EGMLinearRegressionL1L2(l, 0.001, p)
#    egm.set_max_iter(maxit)
#    egm.set_tolerance(eps)
#    egm.fit(X, y)
##    f = egm.get_algorithm().f
##
##    plot.subplot(2, 2, 3)
##    plot.plot(beta1D[:, 0], '-', egm._beta[:, 0], '*')
##    plot.title("EGM LASSO (%d, %f)" % (len(f), f[-1]))
##
##    plot.subplot(2, 2, 4)
##    plot.imshow(np.reshape(egm._beta, (pz, py, px))[0, :, :],
##                interpolation='nearest', cmap=cm.gist_rainbow)
#
#    lasso_pgm_ss = np.sum(pgm._beta ** 2.0)
#    utils.debug("SS beta: ", lasso_pgm_ss)
#    lasso_egm_ss = np.sum(egm._beta ** 2.0)
#    utils.debug("SS beta: ", lasso_egm_ss)
#    assert lasso_egm_ss < lasso_pgm_ss
#    assert lasso_egm_ss > lr_egm_ss
#    assert lasso_pgm_ss > lr_pgm_ss
#
#    lasso_true_error_pgm = np.sum((beta1D - pgm._beta) ** 2.0) \
#                        / np.sum(beta1D ** 2.0)
#    utils.debug("True difference PGM:", lasso_true_error_pgm)
#    lasso_true_error_egm = np.sum((beta1D - egm._beta) ** 2.0) \
#                        / np.sum(beta1D ** 2.0)
#    utils.debug("True difference EGM:", lasso_true_error_egm)
#    assert lasso_true_error_pgm < 0.01
#    assert lasso_true_error_egm < 0.01
#    assert abs(lasso_true_error_pgm - lasso_true_error_egm) < 0.01
#    assert lasso_true_error_pgm < lr_true_error_pgm
#    assert lasso_true_error_egm < lr_true_error_egm
#
#    lasso_ols_diff_pgm = np.sum((betaOLS - pgm._beta) ** 2.0) \
#                        / np.sum(betaOLS ** 2.0)
#    utils.debug("Difference to OLS EGM:", lasso_ols_diff_pgm)
#    lasso_ols_diff_egm = np.sum((betaOLS - egm._beta) ** 2.0) \
#                        / np.sum(betaOLS ** 2.0)
#    utils.debug("Difference to OLS PGM:", lasso_ols_diff_egm)
#    assert lasso_ols_diff_pgm < 0.45
#    assert lasso_ols_diff_egm < 0.40
#    assert lasso_ols_diff_pgm > lr_ols_diff_pgm
#    assert lasso_ols_diff_egm > lr_ols_diff_egm
#
#    lasso_diff_egm_pgm = np.sum((pgm._beta - egm._beta) ** 2.0) \
#                        / np.sum(pgm._beta ** 2.0)
#    utils.debug("Difference EGM -- PGM:", lasso_diff_egm_pgm)
#    lasso_diff_pgm_egm = np.sum((pgm._beta - egm._beta) ** 2.0) \
#                        / np.sum(egm._beta ** 2.0)
#    utils.debug("Difference PGM -- EGM:", lasso_diff_pgm_egm)
#    assert lasso_diff_egm_pgm < 0.01
#    assert lasso_diff_pgm_egm < 0.01
#    assert lasso_diff_egm_pgm > lr_diff_egm_pgm
#    assert lasso_diff_pgm_egm > lr_diff_pgm_egm
#
#    lasso_r2_pgm = 1.0 - np.sum((y - np.dot(X, pgm._beta)) ** 2.0) \
#                        / np.sum(y ** 2.0)
#    utils.debug("R2 PGM:", lasso_r2_pgm)
#    lasso_r2_egm = 1.0 - np.sum((y - np.dot(X, egm._beta)) ** 2.0) \
#                        / np.sum(y ** 2.0)
#    utils.debug("R2 EGM:", lasso_r2_egm)
#    assert abs(lasso_r2_pgm - 1.0) < 0.0000005
#    assert abs(lasso_r2_egm - 1.0) < 0.0000005
#    assert lasso_r2_pgm < lr_r2_pgm
#    assert lasso_r2_egm < lr_r2_egm
#
#    utils.debug("")
#
#    # Elastic Net (Linear regression + Elastic Net)
#    utils.debug("Elastic net")
#    l = 0.95
#    pgm = models.ElasticNet(l)
#    pgm.set_max_iter(maxit)
#    pgm.set_tolerance(eps)
#    pgm.fit(X, y)
##    f = pgm.get_algorithm().f
##
##    plot.subplot(2, 2, 1)
##    plot.plot(beta1D[:, 0], '-', pgm._beta[:, 0], '*')
##    plot.title("Elastic Net (%d, %f)" % (len(f), f[-1]))
##
##    plot.subplot(2, 2, 2)
##    plot.imshow(np.reshape(pgm._beta, (pz, py, px))[0, :, :],
##                interpolation='nearest', cmap=cm.gist_rainbow)
#
#    egm = models.EGMElasticNet(l, p)
#    egm.set_max_iter(maxit)
#    egm.set_tolerance(eps)
#    egm.fit(X, y)
##    f = egm.get_algorithm().f
##
##    plot.subplot(2, 2, 3)
##    plot.plot(beta1D[:, 0], '-', egm._beta[:, 0], '*')
##    plot.title("Elastic Net (%d, %f)" % (len(f), f[-1]))
##
##    plot.subplot(2, 2, 4)
##    plot.imshow(np.reshape(egm._beta, (pz, py, px))[0, :, :],
##                interpolation='nearest', cmap=cm.gist_rainbow)
#
#    en_pgm_ss = np.sum(pgm._beta ** 2.0)
#    utils.debug("SS beta: ", en_pgm_ss)
#    en_egm_ss = np.sum(egm._beta ** 2.0)
#    utils.debug("SS beta: ", en_egm_ss)
#    assert en_egm_ss < en_pgm_ss
#    assert en_egm_ss > lr_egm_ss
#    assert en_pgm_ss > lr_pgm_ss
#    assert en_egm_ss < lasso_egm_ss
#    assert en_pgm_ss < lasso_pgm_ss
#
#    en_true_error_pgm = np.sum((beta1D - pgm._beta) ** 2.0) \
#                        / np.sum(beta1D ** 2.0)
#    utils.debug("True difference PGM:", en_true_error_pgm)
#    en_true_error_egm = np.sum((beta1D - egm._beta) ** 2.0) \
#                        / np.sum(beta1D ** 2.0)
#    utils.debug("True difference EGM:", en_true_error_egm)
#    assert en_true_error_pgm < 0.002
#    assert en_true_error_egm < 0.002
#    assert abs(en_true_error_pgm - en_true_error_egm) < 0.05
#    assert en_true_error_pgm < lr_true_error_pgm
#    assert en_true_error_egm < lr_true_error_egm
##    assert en_true_error_pgm > lasso_true_error_pgm
##    assert en_true_error_egm < lasso_true_error_egm
#
#    en_ols_diff_pgm = np.sum((betaOLS - pgm._beta) ** 2.0) \
#                        / np.sum(betaOLS ** 2.0)
#    utils.debug("Difference to OLS EGM:", en_ols_diff_pgm)
#    en_ols_diff_egm = np.sum((betaOLS - egm._beta) ** 2.0) \
#                        / np.sum(betaOLS ** 2.0)
#    utils.debug("Difference to OLS PGM:", en_ols_diff_egm)
#    assert en_ols_diff_pgm < 0.40
#    assert en_ols_diff_egm < 0.40
#    assert en_ols_diff_pgm > lr_ols_diff_pgm
#    assert en_ols_diff_egm > lr_ols_diff_pgm
#    assert en_ols_diff_pgm < lasso_ols_diff_pgm
#    assert en_ols_diff_egm < lasso_ols_diff_pgm
#
#    en_diff_egm_pgm = np.sum((pgm._beta - egm._beta) ** 2.0) \
#                        / np.sum(pgm._beta ** 2.0)
#    utils.debug("Difference EGM -- PGM:", en_diff_egm_pgm)
#    en_diff_pgm_egm = np.sum((pgm._beta - egm._beta) ** 2.0) \
#                        / np.sum(egm._beta ** 2.0)
#    utils.debug("Difference PGM -- EGM:", en_diff_pgm_egm)
#    assert en_diff_egm_pgm < 0.0005
#    assert en_diff_pgm_egm < 0.0005
#    assert en_diff_egm_pgm < lr_diff_egm_pgm
#    assert en_diff_pgm_egm < lr_diff_pgm_egm
#    assert en_diff_egm_pgm < lasso_diff_egm_pgm
#    assert en_diff_pgm_egm < lasso_diff_pgm_egm
#
#    en_r2_pgm = 1.0 - np.sum((y - np.dot(X, pgm._beta)) ** 2.0) \
#                        / np.sum(y ** 2.0)
#    utils.debug("R2 PGM:", en_r2_pgm)
#    en_r2_egm = 1.0 - np.sum((y - np.dot(X, egm._beta)) ** 2.0) \
#                        / np.sum(y ** 2.0)
#    utils.debug("R2 EGM:", en_r2_egm)
#    assert abs(en_r2_pgm - 1.0) < 0.00005
#    assert abs(en_r2_egm - 1.0) < 0.00005
#    assert en_r2_pgm < lr_r2_pgm
#    assert en_r2_egm < lr_r2_egm
#    assert en_r2_pgm < lasso_r2_pgm
#    assert en_r2_egm < lasso_r2_egm
#
#    utils.debug("")
#
#    # Linear regression + Total variation penalty
#    utils.debug("Linear regression + Total variation")
#    gamma = 0.01
#    pgm = models.LinearRegressionTV(gamma, shape=(pz, py, px), mu=mu)
#    pgm_lrtv = pgm  # For testing compute_mu and compute_eps
#    pgm.set_max_iter(maxit)
#    pgm.set_tolerance(eps)
#    pgm.fit(X, y)
##    f = pgm.get_algorithm().f
##
##    plot.subplot(2, 2, 1)
##    plot.plot(beta1D[:, 0], '-', pgm._beta[:, 0], '*')
##    plot.title("Linear regression + TV (%d, %f)" % (len(f), f[-1]))
##
##    plot.subplot(2, 2, 2)
##    plot.imshow(np.reshape(pgm._beta, (pz, py, px))[0, :, :],
##                interpolation='nearest', cmap=cm.gist_rainbow)
#
#    egm = models.EGMRidgeRegressionTV(0.0005, gamma, (pz, py, px))
#    egm.set_max_iter(maxit)
#    egm.set_tolerance(eps)
#    egm.fit(X, y)
##    f = egm.get_algorithm().f
##
##    plot.subplot(2, 2, 3)
##    plot.plot(beta1D[:, 0], '-', egm._beta[:, 0], '*')
##    plot.title("Linear regression + TV (%d, %f)" % (len(f), f[-1]))
##
##    plot.subplot(2, 2, 4)
##    plot.imshow(np.reshape(egm._beta, (pz, py, px))[0, :, :],
##                interpolation='nearest', cmap=cm.gist_rainbow)
#
#    tv_pgm_ss = np.sum(pgm._beta ** 2.0)
#    utils.debug("SS beta: ", tv_pgm_ss)
#    tv_egm_ss = np.sum(egm._beta ** 2.0)
#    utils.debug("SS beta: ", tv_egm_ss)
#    assert tv_pgm_ss < tv_egm_ss
#    assert tv_pgm_ss > lr_pgm_ss
#    assert tv_egm_ss > lr_egm_ss
#    assert tv_pgm_ss > en_pgm_ss
#    assert tv_egm_ss > en_egm_ss
#
#    tv_true_error_pgm = np.sum((beta1D - pgm._beta) ** 2.0) \
#                        / np.sum(beta1D ** 2.0)
#    utils.debug("True difference PGM:", tv_true_error_pgm)
#    tv_true_error_egm = np.sum((beta1D - egm._beta) ** 2.0) \
#                        / np.sum(beta1D ** 2.0)
#    utils.debug("True difference EGM:", tv_true_error_egm)
#    assert tv_true_error_pgm < 0.001
#    assert tv_true_error_egm < 0.0001
#    assert abs(tv_true_error_pgm - tv_true_error_egm) < 0.005
#    assert tv_true_error_pgm < lr_true_error_pgm
#    assert tv_true_error_egm < lr_true_error_egm
##    assert tv_true_error_pgm > lasso_true_error_pgm
##    assert tv_true_error_egm > lasso_true_error_egm
#    assert tv_true_error_pgm < en_true_error_pgm
#    assert tv_true_error_egm < en_true_error_egm
#
#    tv_ols_diff_pgm = np.sum((betaOLS - pgm._beta) ** 2.0) \
#                        / np.sum(betaOLS ** 2.0)
#    utils.debug("Difference to OLS EGM:", tv_ols_diff_pgm)
#    tv_ols_diff_egm = np.sum((betaOLS - egm._beta) ** 2.0) \
#                        / np.sum(betaOLS ** 2.0)
#    utils.debug("Difference to OLS PGM:", tv_ols_diff_egm)
#    assert tv_ols_diff_pgm < 0.43
#    assert tv_ols_diff_egm < 0.45
#    assert tv_ols_diff_pgm > lr_ols_diff_pgm
#    assert tv_ols_diff_egm > lr_ols_diff_pgm
#    assert tv_ols_diff_pgm > en_ols_diff_pgm
#    assert tv_ols_diff_egm > en_ols_diff_pgm
#
#    tv_diff_egm_pgm = np.sum((pgm._beta - egm._beta) ** 2.0) \
#                        / np.sum(pgm._beta ** 2.0)
#    utils.debug("Difference EGM -- PGM:", tv_diff_egm_pgm)
#    tv_diff_pgm_egm = np.sum((pgm._beta - egm._beta) ** 2.0) \
#                        / np.sum(egm._beta ** 2.0)
#    utils.debug("Difference PGM -- EGM:", tv_diff_pgm_egm)
#    assert tv_diff_egm_pgm < 0.001
#    assert tv_diff_pgm_egm < 0.001
#    assert tv_diff_egm_pgm > lr_diff_egm_pgm
#    assert tv_diff_pgm_egm > lr_diff_pgm_egm
#    assert tv_diff_egm_pgm < lasso_diff_egm_pgm
#    assert tv_diff_pgm_egm < lasso_diff_pgm_egm
#    assert tv_diff_egm_pgm > en_diff_egm_pgm
#    assert tv_diff_pgm_egm > en_diff_pgm_egm
#
#    tv_r2_pgm = 1.0 - np.sum((y - np.dot(X, pgm._beta)) ** 2.0) \
#                        / np.sum(y ** 2.0)
#    utils.debug("R2 PGM:", tv_r2_pgm)
#    tv_r2_egm = 1.0 - np.sum((y - np.dot(X, egm._beta)) ** 2.0) \
#                        / np.sum(y ** 2.0)
#    utils.debug("R2 EGM:", tv_r2_egm)
#    assert abs(tv_r2_pgm - 1.0) < 0.00005
#    assert abs(tv_r2_egm - 1.0) < 0.00005
#    assert tv_r2_pgm < lr_r2_pgm
#    assert tv_r2_egm < lr_r2_egm
#    assert tv_r2_pgm > lasso_r2_pgm
#    assert tv_r2_egm > lasso_r2_egm
#    assert tv_r2_pgm > en_r2_pgm
#    assert tv_r2_egm > en_r2_egm
#
#    utils.debug("")
#
#    # Lasso + Total variation penalty (Linear regression + L1 + TV)
#    utils.debug("LASSO + Total variation")
#    l = 0.01
#    gamma = 0.005
#    pgm = models.LinearRegressionL1TV(l, gamma, (pz, py, px), mu=mu)
##    pgm = models.ContinuationRun(pgm, gaps=[100 * eps, 10 * eps, eps])
##    pgm.set_max_iter(maxit / 3.0)
#    pgm.set_max_iter(maxit)
#    pgm.set_tolerance(eps)
#    pgm.fit(X, y)
##    f = pgm.get_algorithm().f
##
##    plot.subplot(2, 2, 1)
##    plot.plot(beta1D[:, 0], '-', pgm._beta[:, 0], '*')
##    plot.title("Linear regression + L1 + TV (%d, %f)" % (len(f), f[-1]))
##
##    plot.subplot(2, 2, 2)
##    plot.imshow(np.reshape(pgm._beta, (pz, py, px))[0, :, :],
##                interpolation='nearest', cmap=cm.gist_rainbow)
#
#    egm = models.EGMRidgeRegressionL1TV(l, 0.0005, gamma, (pz, py, px))
#    egm.set_max_iter(maxit)
#    egm.set_tolerance(eps)
#    egm.fit(X, y)
##    f = egm.get_algorithm().f
##
##    plot.subplot(2, 2, 3)
##    plot.plot(beta1D[:, 0], '-', egm._beta[:, 0], '*')
##    plot.title("Linear regression + L1 + TV (%d, %f)" % (len(f), f[-1]))
##
##    plot.subplot(2, 2, 4)
##    plot.imshow(np.reshape(egm._beta, (pz, py, px))[0, :, :],
##                interpolation='nearest', cmap=cm.gist_rainbow)
#
#    lassotv_pgm_ss = np.sum(pgm._beta ** 2.0)
#    utils.debug("SS beta: ", lassotv_pgm_ss)
#    lassotv_egm_ss = np.sum(egm._beta ** 2.0)
#    utils.debug("SS beta: ", lassotv_egm_ss)
#    assert lassotv_pgm_ss < lassotv_egm_ss
#    assert lassotv_pgm_ss > lr_pgm_ss
#    assert lassotv_egm_ss > lr_egm_ss
#    assert lassotv_pgm_ss > en_pgm_ss
#    assert lassotv_egm_ss > en_egm_ss
#
#    lassotv_true_error_pgm = np.sum((beta1D - pgm._beta) ** 2.0) \
#                        / np.sum(beta1D ** 2.0)
#    utils.debug("True difference PGM:", lassotv_true_error_pgm)
#    lassotv_true_error_egm = np.sum((beta1D - egm._beta) ** 2.0) \
#                        / np.sum(beta1D ** 2.0)
#    utils.debug("True difference EGM:", lassotv_true_error_egm)
#    assert lassotv_true_error_pgm < 0.0005
#    assert lassotv_true_error_egm < 0.0005
#    assert abs(lassotv_true_error_pgm - lassotv_true_error_egm) < 0.005
#    assert lassotv_true_error_pgm < lr_true_error_pgm
#    assert lassotv_true_error_egm < lr_true_error_egm
##    assert lassotv_true_error_pgm < lasso_true_error_pgm
##    assert lassotv_true_error_egm < lasso_true_error_egm
#    assert lassotv_true_error_pgm < en_true_error_pgm
#    assert lassotv_true_error_egm < en_true_error_egm
#    assert lassotv_true_error_pgm < tv_true_error_pgm
#    assert lassotv_true_error_egm < tv_true_error_egm
#
#    lassotv_ols_diff_pgm = np.sum((betaOLS - pgm._beta) ** 2.0) \
#                        / np.sum(betaOLS ** 2.0)
#    utils.debug("Difference to OLS EGM:", lassotv_ols_diff_pgm)
#    lassotv_ols_diff_egm = np.sum((betaOLS - egm._beta) ** 2.0) \
#                        / np.sum(betaOLS ** 2.0)
#    utils.debug("Difference to OLS PGM:", lassotv_ols_diff_egm)
#    assert lassotv_ols_diff_pgm < 0.44
#    assert lassotv_ols_diff_egm < 0.45
#    assert lassotv_ols_diff_pgm > lr_ols_diff_pgm
#    assert lassotv_ols_diff_egm > lr_ols_diff_pgm
#    assert lassotv_ols_diff_pgm > en_ols_diff_pgm
#    assert lassotv_ols_diff_egm > en_ols_diff_pgm
#
#    lassotv_diff_egm_pgm = np.sum((pgm._beta - egm._beta) ** 2.0) \
#                        / np.sum(pgm._beta ** 2.0)
#    utils.debug("Difference EGM -- PGM:", lassotv_diff_egm_pgm)
#    lassotv_diff_pgm_egm = np.sum((pgm._beta - egm._beta) ** 2.0) \
#                        / np.sum(egm._beta ** 2.0)
#    utils.debug("Difference PGM -- EGM:", lassotv_diff_pgm_egm)
#    assert lassotv_diff_egm_pgm < 0.0005
#    assert lassotv_diff_pgm_egm < 0.0005
#    assert lassotv_diff_egm_pgm > lr_diff_egm_pgm
#    assert lassotv_diff_pgm_egm > lr_diff_pgm_egm
#    assert lassotv_diff_egm_pgm < lasso_diff_egm_pgm
#    assert lassotv_diff_pgm_egm < lasso_diff_pgm_egm
#    assert lassotv_diff_egm_pgm > en_diff_egm_pgm
#    assert lassotv_diff_pgm_egm > en_diff_pgm_egm
#    assert lassotv_diff_egm_pgm < tv_diff_egm_pgm
#    assert lassotv_diff_pgm_egm < tv_diff_pgm_egm
#
#    lassotv_r2_pgm = 1.0 - np.sum((y - np.dot(X, pgm._beta)) ** 2.0) \
#                        / np.sum(y ** 2.0)
#    utils.debug("R2 PGM:", lassotv_r2_pgm)
#    lassotv_r2_egm = 1.0 - np.sum((y - np.dot(X, egm._beta)) ** 2.0) \
#                        / np.sum(y ** 2.0)
#    utils.debug("R2 EGM:", lassotv_r2_egm)
#    assert abs(lassotv_r2_pgm - 1.0) < 0.00005
#    assert abs(lassotv_r2_egm - 1.0) < 0.00005
#    assert lassotv_r2_pgm < lr_r2_pgm
#    assert lassotv_r2_egm < lr_r2_egm
#    assert lassotv_r2_pgm > lasso_r2_pgm
#    assert lassotv_r2_egm > lasso_r2_egm
#    assert lassotv_r2_pgm > en_r2_pgm
#    assert lassotv_r2_egm > en_r2_egm
##    assert lassotv_r2_pgm < tv_r2_pgm
##    assert lassotv_r2_egm < tv_r2_egm
#
#    utils.debug("")
#
#    # Elastic Net + Total variation penalty (Linear regression + EN + TV)
#    utils.debug("Elastic net + total variation")
#    l = 0.8
#    gamma = 1.0
#    pgm = models.ElasticNetTV(l, gamma, (pz, py, px))  # , mu=mu)
#    pgm = models.ContinuationRun(pgm, gaps=[10000 * eps, 100 * eps, eps])
#    pgm.set_max_iter(maxit / 3.0)
##    pgm.set_tolerance(eps)
#    pgm.fit(X, y)
##    f = pgm.get_algorithm().f
##
##    plot.subplot(2, 2, 1)
##    plot.plot(beta1D[:, 0], '-', pgm._beta[:, 0], '*')
##    plot.title("Elastic Net + TV (%d, %f)" % (len(f), f[-1]))
##
##    plot.subplot(2, 2, 2)
##    plot.imshow(np.reshape(pgm._beta, (pz, py, px))[0, :, :],
##                interpolation='nearest', cmap=cm.gist_rainbow)
#
#    egm = models.EGMElasticNetTV(l, gamma, (pz, py, px))
#    egm.set_max_iter(maxit)
#    egm.set_tolerance(eps)
#    egm.fit(X, y)
##    f = egm.get_algorithm().f
##
##    plot.subplot(2, 2, 3)
##    plot.plot(beta1D[:, 0], '-', egm._beta[:, 0], '*')
##    plot.title("Elastic Net + TV (%d, %f)" % (len(f), f[-1]))
##
##    plot.subplot(2, 2, 4)
##    plot.imshow(np.reshape(egm._beta, (pz, py, px))[0, :, :],
##                interpolation='nearest', cmap=cm.gist_rainbow)
#
#    entv_pgm_ss = np.sum(pgm._beta ** 2.0)
#    utils.debug("SS beta: ", entv_pgm_ss)
#    entv_egm_ss = np.sum(egm._beta ** 2.0)
#    utils.debug("SS beta: ", entv_egm_ss)
#    assert entv_pgm_ss > entv_egm_ss
#    assert entv_pgm_ss > lr_pgm_ss
#    assert entv_egm_ss > lr_egm_ss
#    assert entv_pgm_ss < lasso_pgm_ss
#    assert entv_egm_ss < lasso_egm_ss
#    assert entv_pgm_ss < en_pgm_ss
#    assert entv_egm_ss < en_egm_ss
#    assert entv_pgm_ss < tv_pgm_ss
#    assert entv_egm_ss < tv_egm_ss
#    assert entv_pgm_ss < lassotv_pgm_ss
#    assert entv_egm_ss < lassotv_egm_ss
#
#    entv_true_error_pgm = np.sum((beta1D - pgm._beta) ** 2.0) \
#                        / np.sum(beta1D ** 2.0)
#    utils.debug("True difference PGM:", entv_true_error_pgm)
#    entv_true_error_egm = np.sum((beta1D - egm._beta) ** 2.0) \
#                        / np.sum(beta1D ** 2.0)
#    utils.debug("True difference EGM:", entv_true_error_egm)
#    assert entv_true_error_pgm < 0.005
#    assert entv_true_error_egm < 0.005
#    assert abs(entv_true_error_pgm - entv_true_error_egm) < 0.005
#    assert entv_true_error_pgm < lr_true_error_pgm
#    assert entv_true_error_egm < lr_true_error_egm
##    assert entv_true_error_pgm > lasso_true_error_pgm
##    assert entv_true_error_egm > lasso_true_error_egm
#    assert entv_true_error_pgm > en_true_error_pgm
#    assert entv_true_error_egm > en_true_error_egm
#    assert entv_true_error_pgm > tv_true_error_pgm
#    assert entv_true_error_egm > tv_true_error_egm
#    assert entv_true_error_pgm > lassotv_true_error_pgm
#    assert entv_true_error_egm > lassotv_true_error_egm
#
#    entv_ols_diff_pgm = np.sum((betaOLS - pgm._beta) ** 2.0) \
#                        / np.sum(betaOLS ** 2.0)
#    utils.debug("Difference to OLS EGM:", entv_ols_diff_pgm)
#    entv_ols_diff_egm = np.sum((betaOLS - egm._beta) ** 2.0) \
#                        / np.sum(betaOLS ** 2.0)
#    utils.debug("Difference to OLS PGM:", entv_ols_diff_egm)
#    assert entv_ols_diff_pgm < 0.37
#    assert entv_ols_diff_egm < 0.37
#    assert entv_ols_diff_pgm > lr_ols_diff_pgm
#    assert entv_ols_diff_egm > lr_ols_diff_egm
#    assert entv_ols_diff_pgm < lasso_ols_diff_pgm
#    assert entv_ols_diff_egm < lasso_ols_diff_egm
#    assert entv_ols_diff_pgm < en_ols_diff_pgm
#    assert entv_ols_diff_egm < en_ols_diff_egm
#    assert entv_ols_diff_pgm < tv_ols_diff_pgm
#    assert entv_ols_diff_egm < tv_ols_diff_egm
#    assert entv_ols_diff_pgm < lassotv_ols_diff_pgm
#    assert entv_ols_diff_egm < lassotv_ols_diff_egm
#
#    entv_diff_egm_pgm = np.sum((pgm._beta - egm._beta) ** 2.0) \
#                        / np.sum(pgm._beta ** 2.0)
#    utils.debug("Difference EGM -- PGM:", entv_diff_egm_pgm)
#    entv_diff_pgm_egm = np.sum((pgm._beta - egm._beta) ** 2.0) \
#                        / np.sum(egm._beta ** 2.0)
#    utils.debug("Difference PGM -- EGM:", entv_diff_pgm_egm)
#    assert entv_diff_egm_pgm < 0.00001
#    assert entv_diff_pgm_egm < 0.00001
#    assert entv_diff_egm_pgm < lr_diff_egm_pgm
#    assert entv_diff_pgm_egm < lr_diff_pgm_egm
#    assert entv_diff_egm_pgm < lasso_diff_egm_pgm
#    assert entv_diff_pgm_egm < lasso_diff_pgm_egm
#    assert entv_diff_egm_pgm < en_diff_egm_pgm
#    assert entv_diff_pgm_egm < en_diff_pgm_egm
#    assert entv_diff_egm_pgm < tv_diff_egm_pgm
#    assert entv_diff_pgm_egm < tv_diff_pgm_egm
#    assert entv_diff_egm_pgm < lassotv_diff_egm_pgm
#    assert entv_diff_pgm_egm < lassotv_diff_pgm_egm
#
#    entv_r2_pgm = 1.0 - np.sum((y - np.dot(X, pgm._beta)) ** 2.0) \
#                        / np.sum(y ** 2.0)
#    utils.debug("R2 PGM:", entv_r2_pgm)
#    entv_r2_egm = 1.0 - np.sum((y - np.dot(X, egm._beta)) ** 2.0) \
#                        / np.sum(y ** 2.0)
#    utils.debug("R2 EGM:", entv_r2_egm)
#    assert abs(entv_r2_pgm - 1.0) < 0.00005
#    assert abs(entv_r2_egm - 1.0) < 0.00005
#    assert entv_r2_pgm < lr_r2_pgm
#    assert entv_r2_egm < lr_r2_egm
#    assert entv_r2_pgm < lasso_r2_pgm
#    assert entv_r2_egm < lasso_r2_egm
#    assert entv_r2_pgm < en_r2_pgm
#    assert entv_r2_egm < en_r2_egm
#    assert entv_r2_pgm < tv_r2_pgm
#    assert entv_r2_egm < tv_r2_egm
#    assert entv_r2_pgm < lassotv_r2_pgm
#    assert entv_r2_egm < lassotv_r2_egm
#
#    utils.debug("")
#
##    plot.show()
#
#    # Test the functions compute_mu and compute_gap
#    utils.debug("Testing compute_mu and compute_gap:")
#    g = pgm_lrtv.get_g()
#    lr = g.a
#    tv = g.b
#
#    D = tv.num_compacts() / 2.0
##    print "D:", D
#    A = tv.Lipschitz(1.0)
#    l = lr.Lipschitz()
#
##    print "A:", A
##    print "l:", l
#
#    def mu_plus(eps):
#        return (-2.0 * D * A + np.sqrt((2.0 * D * A) ** 2.0 \
#                + 4.0 * D * l * eps * A)) / (2.0 * D * l)
#
#    def eps_plus(mu):
#        return ((2.0 * mu * D * l + 2.0 * D * A) ** 2.0 \
#                - (2.0 * D * A) ** 2.0) / (4.0 * D * l * A)
#
#    utils.debug("Testing eps:")
#    for eps in [1000000.0, 100000.0, 10000.0, 1000.0, 100.0, 10.0, \
#                1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]:
#        mu1 = mu_plus(eps)
#        eps1 = eps_plus(mu_plus(eps))
#        err1 = abs(eps - eps1) / eps
#        mu2 = pgm_lrtv.compute_mu(eps)
#        eps2 = pgm_lrtv.compute_gap(pgm_lrtv.compute_mu(eps))
#        err2 = abs(eps - eps2) / eps
#
#        utils.debug("eps: %.8f -> mu: %.8f -> eps: %.8f (err: %.8f)" \
#                % (eps, mu1, eps1, err1))
#        utils.debug("eps: %.8f -> mu: %.8f -> eps: %.8f (err: %.8f)" \
#                % (eps, mu2, eps2, err2))
#
#        if eps < 0.0001:
#            assert err1 < 1.0
#            assert err2 < 1.0
#        elif eps < 0.00001:
#            assert err1 < 0.005
#            assert err2 < 0.005
#        else:
#            assert err1 < 0.0005
#            assert err2 < 0.0005
#
#    utils.debug("")
#    utils.debug("Testing mu:")
#    for mu in [1000000.0, 100000.0, 10000.0, 1000.0, 100.0, 10.0,
#               1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]:
#        eps1 = eps_plus(mu)
#        mu1 = mu_plus(eps_plus(mu))
#        err1 = abs(mu - mu1) / mu
#        eps2 = pgm_lrtv.compute_gap(mu)
#        mu2 = pgm_lrtv.compute_mu(pgm_lrtv.compute_gap(mu))
#        err2 = abs(mu - mu2) / mu
#
#        utils.debug("mu: %.8f -> eps: %.8f -> mu: %.8f (err: %.8f)" \
#                % (mu, eps1, mu1, err1))
#        utils.debug("mu: %.8f -> eps: %.8f -> mu: %.8f (err: %.8f)" \
#                % (mu, eps2, mu2, err2))
#
#        if mu > 100000.0:
#            assert err1 < 0.01
#            assert err2 < 0.01
#        else:
#            assert err1 < 0.0005
#            assert err2 < 0.0005


if __name__ == "__main__":

    t = time()
    test()
    utils.debug("test_structure took %.2f seconds" % (time() - t))