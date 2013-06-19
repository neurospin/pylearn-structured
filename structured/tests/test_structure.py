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

    num_mus = 1
    mus = [0] * num_mus
    mus[0] = 1.0
#    mus[1] = 0.01
#    mus[2] = 0.0001
#    mus[3] = 0.000001
#    mus[4] = 0.00000001

    # Linear regression
    pgm = models.LinearRegression()
    pgm.set_max_iter(maxit)
    pgm.set_tolerance(eps)
    pgm.fit(X, y)
#    f = pgm.get_algorithm().f
    print "SS: ", np.sum(pgm.beta ** 2.0)

#    plot.subplot(4, 4, 1)
#    plot.plot(beta1D[:, 0], '-', m.beta[:, 0], '*')
#    plot.title("PGM Linear regression (%f, %f)" % (len(f), f[-1]))
#
#    plot.subplot(4, 4, 2)
#    plot.imshow(np.reshape(m.beta, (pz, py, px))[0, :, :],
#                interpolation='nearest', cmap=cm.gist_rainbow)

    egm = models.EGMRidgeRegression(utils.TOLERANCE)
    egm.set_max_iter(maxit)
    egm.set_tolerance(eps)
    egm.fit(X, y)
#    f = egm.get_algorithm().f
    print "SS: ", np.sum(egm.beta ** 2.0)

    utils.debug("True difference PGM:",
        1.0 - np.sum((beta1D - pgm.beta) ** 2.0) / np.sum(beta1D ** 2.0))
    utils.debug("True difference EGM:",
        1.0 - np.sum((beta1D - egm.beta) ** 2.0) / np.sum(beta1D ** 2.0))
    utils.debug("Difference to OLS EGM:",
        1.0 - np.sum((betaOLS - pgm.beta) ** 2.0) / np.sum(betaOLS ** 2.0))
    utils.debug("Difference to OLS PGM:",
        1.0 - np.sum((betaOLS - egm.beta) ** 2.0) / np.sum(betaOLS ** 2.0))
    utils.debug("Difference EGM -- PGM:",
        1.0 - np.sum((pgm.beta - egm.beta) ** 2.0) / np.sum(pgm.beta ** 2.0))
    utils.debug("Difference PGM -- EGM:",
        1.0 - np.sum((pgm.beta - egm.beta) ** 2.0) / np.sum(egm.beta ** 2.0))
    utils.debug("R2 PGM:",
        1.0 - np.sum((y - np.dot(X, pgm.beta)) ** 2.0) / np.sum(y ** 2.0))
    utils.debug("R2 EGM:",
        1.0 - np.sum((y - np.dot(X, egm.beta)) ** 2.0) / np.sum(y ** 2.0))

#    plot.subplot(4, 4, 3)
#    plot.plot(beta1D[:, 0], '-', m.beta[:, 0], '*')
#    plot.title("EGM Linear regression (%f, %f)" % (len(f), f[-1]))
#
#    plot.subplot(4, 4, 4)
#    plot.imshow(np.reshape(m.beta, (pz, py, px))[0, :, :],
#                interpolation='nearest', cmap=cm.gist_rainbow)

#    # LASSO (Linear regression + L1 penalty)
#    l = 1.0
#    pgm = models.Lasso(l)
#    pgm.set_max_iter(maxit)
#    pgm.set_tolerance(eps)
#    pgm.fit(X, y)
#    f = pgm.get_algorithm().f
#    print "SS: ", np.sum(pgm.beta ** 2.0)
#
##    plot.subplot(4, 4, 5)
##    plot.plot(beta1D[:, 0], '-', pgm.beta[:, 0], '*')
##    plot.title("PGM LASSO (%f, %f)" % (len(f), f[-1]))
##
##    plot.subplot(4, 4, 6)
##    plot.imshow(np.reshape(pgm.beta, (pz, py, px))[0, :, :],
##                interpolation='nearest', cmap=cm.gist_rainbow)
#
#    egm = models.EGMLinearRegressionL1L2(l, 0.00002, p, mu=mus[0])
#    egm.set_max_iter(maxit)
#    egm.set_tolerance(eps)
#    egm.fit(X, y)
#    f = egm.get_algorithm().f
#    print "SS: ", np.sum(egm.beta ** 2.0)
#
##    plot.subplot(4, 4, 7)
##    plot.plot(beta1D[:, 0], '-', egm.beta[:, 0], '*')
##    plot.title("EGM LASSO (%f, %f)" % (len(f), f[-1]))
##
##    plot.subplot(4, 4, 8)
##    plot.imshow(np.reshape(egm.beta, (pz, py, px))[0, :, :],
##                interpolation='nearest', cmap=cm.gist_rainbow)
#
#    print 1.0 - np.sum((beta1D - pgm.beta) ** 2.0) / np.sum(beta1D ** 2.0)
#    print 1.0 - np.sum((beta1D - egm.beta) ** 2.0) / np.sum(beta1D ** 2.0)
#    print 1.0 - np.sum((betaOLS - pgm.beta) ** 2.0) / np.sum(betaOLS ** 2.0)
#    print 1.0 - np.sum((betaOLS - egm.beta) ** 2.0) / np.sum(betaOLS ** 2.0)
#    print 1.0 - np.sum((pgm.beta - egm.beta) ** 2.0) / np.sum(pgm.beta ** 2.0)
#    print 1.0 - np.sum((pgm.beta - egm.beta) ** 2.0) / np.sum(egm.beta ** 2.0)
#    print 1.0 - np.sum((y - np.dot(X, pgm.beta)) ** 2.0) / np.sum(y ** 2.0)
#    print 1.0 - np.sum((y - np.dot(X, egm.beta)) ** 2.0) / np.sum(y ** 2.0)

#    # Elastic Net (Linear regression + EN)
#    l = 0.8
#    pgm = models.ElasticNet(l)
#    pgm.set_max_iter(maxit)
#    pgm.set_tolerance(eps)
#    pgm.fit(X, y)
#    f = pgm.get_algorithm().f
#    print "SS: ", np.sum(pgm.beta ** 2.0)
#
#    plot.subplot(2, 2, 1)
#    plot.plot(beta1D[:, 0], '-', pgm.beta[:, 0], '*')
#    plot.title("Elastic Net (%d, %f)" % (len(f), f[-1]))
#
#    plot.subplot(2, 2, 2)
#    plot.imshow(np.reshape(pgm.beta, (pz, py, px))[0, :, :],
#                interpolation='nearest', cmap=cm.gist_rainbow)
#
#    egm = models.EGMElasticNet(l, p, mu=mus[0])
#    egm.set_max_iter(maxit)
#    egm.set_tolerance(eps)
#    egm.fit(X, y)
#    f = egm.get_algorithm().f
#    print "SS: ", np.sum(egm.beta ** 2.0)
#
#    plot.subplot(2, 2, 3)
#    plot.plot(beta1D[:, 0], '-', egm.beta[:, 0], '*')
#    plot.title("Elastic Net (%d, %f)" % (len(f), f[-1]))
#
#    plot.subplot(2, 2, 4)
#    plot.imshow(np.reshape(egm.beta, (pz, py, px))[0, :, :],
#                interpolation='nearest', cmap=cm.gist_rainbow)
#
#    utils.debug("True difference PGM:",
#        1.0 - np.sum((beta1D - pgm.beta) ** 2.0) / np.sum(beta1D ** 2.0))
#    utils.debug("True difference EGM:",
#        1.0 - np.sum((beta1D - egm.beta) ** 2.0) / np.sum(beta1D ** 2.0))
#    utils.debug("Difference to OLS EGM:",
#        1.0 - np.sum((betaOLS - pgm.beta) ** 2.0) / np.sum(betaOLS ** 2.0))
#    utils.debug("Difference to OLS PGM:",
#        1.0 - np.sum((betaOLS - egm.beta) ** 2.0) / np.sum(betaOLS ** 2.0))
#    utils.debug("Difference EGM -- PGM:",
#        1.0 - np.sum((pgm.beta - egm.beta) ** 2.0) / np.sum(pgm.beta ** 2.0))
#    utils.debug("Difference PGM -- EGM:",
#        1.0 - np.sum((pgm.beta - egm.beta) ** 2.0) / np.sum(egm.beta ** 2.0))
#    utils.debug("R2 PGM:",
#        1.0 - np.sum((y - np.dot(X, pgm.beta)) ** 2.0) / np.sum(y ** 2.0))
#    utils.debug("R2 EGM:",
#        1.0 - np.sum((y - np.dot(X, egm.beta)) ** 2.0) / np.sum(y ** 2.0))

#    # Linear regression + Total variation penalty
#    gamma = 1.0
#    pgm = models.LinearRegressionTV(gamma, (pz, py, px), mu=mus[0])
#    pgm.set_max_iter(maxit)
#    pgm.set_tolerance(eps)
#    pgm.fit(X, y)
#    f = pgm.get_algorithm().f
#    print "SS: ", np.sum(pgm.beta ** 2.0)
#
##    plot.subplot(4, 4, 9)
##    plot.plot(beta1D[:, 0], '-', pgm.beta[:, 0], '*')
##    plot.title("Linear regression + TV (%f, %f)" % (len(f), f[-1]))
##
##    plot.subplot(4, 4, 10)
##    plot.imshow(np.reshape(pgm.beta, (pz, py, px))[0, :, :],
##                interpolation='nearest', cmap=cm.gist_rainbow)
#
#    egm = models.EGMRidgeRegressionTV(0.0002, gamma, (pz, py, px), mu=mus[0])
#    egm.set_max_iter(maxit)
#    egm.set_tolerance(eps)
#    egm.fit(X, y)
#    f = egm.get_algorithm().f
#    print "SS: ", np.sum(egm.beta ** 2.0)
#
##    plot.subplot(4, 4, 11)
##    plot.plot(beta1D[:, 0], '-', egm.beta[:, 0], '*')
##    plot.title("Linear regression + TV (%f, %f)" % (len(f), f[-1]))
##
##    plot.subplot(4, 4, 12)
##    plot.imshow(np.reshape(egm.beta, (pz, py, px))[0, :, :],
##                interpolation='nearest', cmap=cm.gist_rainbow)
#
#    utils.debug("True difference PGM:",
#        1.0 - np.sum((beta1D - pgm.beta) ** 2.0) / np.sum(beta1D ** 2.0))
#    utils.debug("True difference EGM:",
#        1.0 - np.sum((beta1D - egm.beta) ** 2.0) / np.sum(beta1D ** 2.0))
#    utils.debug("Difference to OLS EGM:",
#        1.0 - np.sum((betaOLS - pgm.beta) ** 2.0) / np.sum(betaOLS ** 2.0))
#    utils.debug("Difference to OLS PGM:",
#        1.0 - np.sum((betaOLS - egm.beta) ** 2.0) / np.sum(betaOLS ** 2.0))
#    utils.debug("Difference EGM -- PGM:",
#        1.0 - np.sum((pgm.beta - egm.beta) ** 2.0) / np.sum(pgm.beta ** 2.0))
#    utils.debug("Difference PGM -- EGM:",
#        1.0 - np.sum((pgm.beta - egm.beta) ** 2.0) / np.sum(egm.beta ** 2.0))
#    utils.debug("R2 PGM:",
#        1.0 - np.sum((y - np.dot(X, pgm.beta)) ** 2.0) / np.sum(y ** 2.0))
#    utils.debug("R2 EGM:",
#        1.0 - np.sum((y - np.dot(X, egm.beta)) ** 2.0) / np.sum(y ** 2.0))

#    # Lasso + Total variation penalty (Linear regression + L1 + TV)
#    l = 0.5
#    gamma = 1.0
#    pgm = models.LinearRegressionL1TV(l, gamma, (pz, py, px), mu=mus[0])
#    pgm.set_max_iter(maxit)
#    pgm.set_tolerance(eps)
#    pgm.fit(X, y)
#    f = pgm.get_algorithm().f
#    print "SS: ", np.sum(pgm.beta ** 2.0)
#
#    plot.subplot(4, 4, 13)
#    plot.plot(beta1D[:, 0], '-', pgm.beta[:, 0], '*')
#    plot.title("Linear regression + L1 + TV (%f, %f)" % (len(f), f[-1]))
#
#    plot.subplot(4, 4, 14)
#    plot.imshow(np.reshape(pgm.beta, (pz, py, px))[0, :, :],
#                interpolation='nearest', cmap=cm.gist_rainbow)
#
#    egm = models.EGMLinearRegressionL1L2TV(l, 0.0002, gamma, (pz, py, px),
#                                           mu=mus[0])
#    egm.set_max_iter(maxit)
#    egm.set_tolerance(eps)
#    egm.fit(X, y)
#    f = egm.get_algorithm().f
#    print "SS: ", np.sum(egm.beta ** 2.0)
#
#    plot.subplot(4, 4, 15)
#    plot.plot(beta1D[:, 0], '-', egm.beta[:, 0], '*')
#    plot.title("Linear regression + L1 + TV (%f, %f)" % (len(f), f[-1]))
#
#    plot.subplot(4, 4, 16)
#    plot.imshow(np.reshape(egm.beta, (pz, py, px))[0, :, :],
#                interpolation='nearest', cmap=cm.gist_rainbow)
#
#    utils.debug("True difference PGM:",
#        1.0 - np.sum((beta1D - pgm.beta) ** 2.0) / np.sum(beta1D ** 2.0))
#    utils.debug("True difference EGM:",
#        1.0 - np.sum((beta1D - egm.beta) ** 2.0) / np.sum(beta1D ** 2.0))
#    utils.debug("Difference to OLS EGM:",
#        1.0 - np.sum((betaOLS - pgm.beta) ** 2.0) / np.sum(betaOLS ** 2.0))
#    utils.debug("Difference to OLS PGM:",
#        1.0 - np.sum((betaOLS - egm.beta) ** 2.0) / np.sum(betaOLS ** 2.0))
#    utils.debug("Difference EGM -- PGM:",
#        1.0 - np.sum((pgm.beta - egm.beta) ** 2.0) / np.sum(pgm.beta ** 2.0))
#    utils.debug("Difference PGM -- EGM:",
#        1.0 - np.sum((pgm.beta - egm.beta) ** 2.0) / np.sum(egm.beta ** 2.0))
#    utils.debug("R2 PGM:",
#        1.0 - np.sum((y - np.dot(X, pgm.beta)) ** 2.0) / np.sum(y ** 2.0))
#    utils.debug("R2 EGM:",
#        1.0 - np.sum((y - np.dot(X, egm.beta)) ** 2.0) / np.sum(y ** 2.0))

    # Elastic Net + Total variation penalty (Linear regression + EN + TV)
    l = 0.8
    gamma = 1.0
    pgm = models.ElasticNetTV(l, gamma, (pz, py, px), mu=mus[0])
    pgm.set_max_iter(maxit)
    pgm.set_tolerance(eps)
    pgm.fit(X, y)
    f = pgm.get_algorithm().f
    print "SS: ", np.sum(pgm.beta ** 2.0)

    plot.subplot(2, 2, 1)
    plot.plot(beta1D[:, 0], '-', pgm.beta[:, 0], '*')
    plot.title("Elastic Net + TV (%d, %f)" % (len(f), f[-1]))

    plot.subplot(2, 2, 2)
    plot.imshow(np.reshape(pgm.beta, (pz, py, px))[0, :, :],
                interpolation='nearest', cmap=cm.gist_rainbow)

    egm = models.EGMElasticNetTV(l, gamma, (pz, py, px), mu=mus[0])
    egm.set_max_iter(maxit)
    egm.set_tolerance(eps)
    egm.fit(X, y)
    f = egm.get_algorithm().f
    print "SS: ", np.sum(egm.beta ** 2.0)

    plot.subplot(2, 2, 3)
    plot.plot(beta1D[:, 0], '-', egm.beta[:, 0], '*')
    plot.title("Elastic Net + TV (%d, %f)" % (len(f), f[-1]))

    plot.subplot(2, 2, 4)
    plot.imshow(np.reshape(egm.beta, (pz, py, px))[0, :, :],
                interpolation='nearest', cmap=cm.gist_rainbow)

    utils.debug("True difference PGM:",
        1.0 - np.sum((beta1D - pgm.beta) ** 2.0) / np.sum(beta1D ** 2.0))
    utils.debug("True difference EGM:",
        1.0 - np.sum((beta1D - egm.beta) ** 2.0) / np.sum(beta1D ** 2.0))
    utils.debug("Difference to OLS EGM:",
        1.0 - np.sum((betaOLS - pgm.beta) ** 2.0) / np.sum(betaOLS ** 2.0))
    utils.debug("Difference to OLS PGM:",
        1.0 - np.sum((betaOLS - egm.beta) ** 2.0) / np.sum(betaOLS ** 2.0))
    utils.debug("Difference EGM -- PGM:",
        1.0 - np.sum((pgm.beta - egm.beta) ** 2.0) / np.sum(pgm.beta ** 2.0))
    utils.debug("Difference PGM -- EGM:",
        1.0 - np.sum((pgm.beta - egm.beta) ** 2.0) / np.sum(egm.beta ** 2.0))
    utils.debug("R2 PGM:",
        1.0 - np.sum((y - np.dot(X, pgm.beta)) ** 2.0) / np.sum(y ** 2.0))
    utils.debug("R2 EGM:",
        1.0 - np.sum((y - np.dot(X, egm.beta)) ** 2.0) / np.sum(y ** 2.0))

    plot.show()


if __name__ == "__main__":

    test()