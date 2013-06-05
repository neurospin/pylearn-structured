# -*- coding: utf-8 -*-
"""
Created on Thu May 16 09:41:13 2013

@author: Tommy Löfstedt
'"""

import numpy as np
import multiblock.utils as utils
from multiblock import *
import multiblock.methods as methods
#import multiblock.start_vectors as start_vectors
#import multiblock.prox_ops as prox_ops
#import multiblock.schemes as schemes
#from sklearn.datasets import load_linnerud
from time import time

import matplotlib.pyplot as plot
import matplotlib.cm as cm
#import pylab


def test_lasso():

    np.random.seed(42)

    eps = 0.01
    maxit = 10000

    px = 100
    py = 1
    pz = 1
    p = px * py * pz  # Must be even!
    n = 60
    X = np.random.randn(n, p)
    betastar = np.concatenate((np.zeros((p / 2, 1)),
                               np.random.randn(p / 2, 1)))
    betastar = np.sort(np.abs(betastar), axis=0)
    y = np.dot(X, betastar)


    print "LinearRegression"
    lr = methods.LinearRegression()
    lr.set_max_iter(maxit)
    lr.set_tolerance(eps)
    lr.fit(X, y)
    computed_beta = lr.beta

    plot.subplot(3, 1, 1)
    plot.plot(betastar[:, 0], '-', computed_beta[:, 0], '*')
    plot.title("Linear regression")



    print "LASSO"
    l = 20.0
    lasso = methods.LASSO(l)
    lasso.set_max_iter(maxit)
    lasso.set_tolerance(eps)

    lasso.fit(X, y)
    computed_beta = lasso.beta

    plot.subplot(3, 1, 2)
    plot.plot(betastar[:, 0], '-', computed_beta[:, 0], '*')
    plot.title("LASSO")



    print "LinearRegressionTV"
    gamma = 0.01
    lrtv = methods.LinearRegressionTV(gamma, (pz, py, px), mu=0.01)
    lrtv.set_max_iter(maxit)
    lrtv.set_tolerance(eps)
    cr = methods.ContinuationRun(lrtv, [1.0, 0.1, 0.01, 0.001, 0.0001])

    cr.fit(X, y)
    computed_beta = cr.beta

    plot.subplot(3, 1, 3)
    plot.plot(betastar[:, 0], '-', computed_beta[:, 0], '*')
    plot.title("Linear regression + TV")



#    print "LinearRegressionL1TV"
#    gamma = 0.01
#    l = 1.0
#    lrl1tv = methods.LinearRegressionL1TV(l, gamma, (pz, py, px), mu=0.01)
#    lrl1tv.set_max_iter(maxit)
#    lrl1tv.set_tolerance(eps)
#    cr = methods.ContinuationRun(lrtv, [1.0, 0.1, 0.01, 0.001, 0.0001])
#
#    cr.fit(X, y)
#    computed_beta = cr.beta
#
#    plot.subplot(4, 1, 4)
#    plot.plot(betastar[:, 0], '-', computed_beta[:, 0], '*')
#    plot.title("Linear regression + TV + L1")

    plot.show()




def test_lasso_tv():
    np.random.seed(42)

    x = np.arange(-10, 10, 1)
    y = np.arange(-10, 10, 1)
    nrows, ncols = len(x), len(y)
    px = ncols
    py = nrows
    pz = 1
    p = nrows * ncols
    n = 70
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

#    u = np.random.randn(p, p)
    u = np.eye(p, p)
    sigma = np.dot(u.T, u)
    mean = np.zeros(p)

    X = np.random.multivariate_normal(mean, sigma, n)
    y = np.dot(X, beta1D)

    eps = 0.01
    maxit = 50000

    num_mus = 1
    mus = [0] * num_mus
    mus[0] = 10.0
#    mus[1] = 0.01
#    mus[2] = 0.0001
#    mus[3] = 0.000001
#    mus[4] = 0.00000001

    lr = methods.LinearRegression()
    lr.set_max_iter(maxit)
    lr.set_tolerance(eps)
    lr.fit(X, y)
    computed_beta = lr.beta

    plot.subplot(3, 3, 1)
    plot.plot(beta1D[:, 0], '-', computed_beta[:, 0], '*')
    plot.title("Linear regression")

    plot.subplot(3, 3, 2)
    plot.imshow(beta, interpolation='nearest', cmap=cm.gist_rainbow)

    plot.subplot(3, 3, 3)
    plot.imshow(np.reshape(computed_beta, (pz, py, px))[0, :, :],
                interpolation='nearest', cmap=cm.gist_rainbow)



    l = 1.0
    l1 = methods.LASSO(l)
    l1.set_max_iter(maxit)
    l1.set_tolerance(eps)
    l1.fit(X, y)
    computed_beta = l1.beta

    plot.subplot(3, 3, 4)
    plot.plot(beta1D[:, 0], '-', computed_beta[:, 0], '*')
    plot.title("LASSO")

    plot.subplot(3, 3, 5)
    plot.imshow(beta, interpolation='nearest', cmap=cm.gist_rainbow)

    plot.subplot(3, 3, 6)
    plot.imshow(np.reshape(computed_beta, (pz, py, px))[0, :, :],
                interpolation='nearest', cmap=cm.gist_rainbow)


    gamma = 10.0
    lrtv = methods.LinearRegressionTV(gamma, (pz, py, px), mu=mus[0])
    lrtv.set_max_iter(maxit)
    lrtv.set_tolerance(eps)
    lrtv.fit(X, y)
    computed_beta = lrtv.beta

    plot.subplot(3, 3, 7)
    plot.plot(beta1D[:, 0], '-', computed_beta[:, 0], '*')
    plot.title("Linear regression + TV")

    plot.subplot(3, 3, 8)
    plot.imshow(beta, interpolation='nearest', cmap=cm.gist_rainbow)

    plot.subplot(3, 3, 9)
    plot.imshow(np.reshape(computed_beta, (pz, py, px))[0, :, :],
                interpolation='nearest', cmap=cm.gist_rainbow)


    plot.show()


def test_tv():
    np.random.seed(42)

    x = np.arange(-10, 10, 1)
    y = np.arange(-10, 10, 1)
    nrows, ncols = len(x), len(y)
    px = ncols
    py = nrows
    pz = 1
    p = nrows * ncols
    n = 70
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

    u = np.random.randn(p, p)
#    u = np.eye(p, p)
    sigma = np.dot(u.T, u)
    mean = np.zeros(p)

#    pylab.imshow(beta, extent=(x.min(), x.max(), y.max(), y.min()),
#               interpolation='nearest', cmap=cm.gist_rainbow)
#    pylab.show()
#
#    pylab.imshow(mask, extent=(x.min(), x.max(), y.max(), y.min()),
#               interpolation='nearest', cmap=cm.gist_rainbow)
#    pylab.show()

    X = np.random.multivariate_normal(mean, sigma, n)
    y = np.dot(X, beta1D)

#    px = 1
#    py = 300
#    pz = 1
#    p = px * py * pz  # Must be even!
#    n = 50
#    X = np.random.randn(n, p)
#    betastar = np.concatenate((np.zeros((p / 2, 1)),
#                               np.random.randn(p / 2, 1)))
#    beta1D = np.sort(np.abs(betastar), axis=0)
#    y = np.dot(X, beta1D)

    eps = 0.1
    maxit = 10000

    gamma = 10.0
    l = 0.1
    en_lambda = 0.95

    num_mus = 1
    mus = [0] * num_mus
    mus[0] = 10.0
#    mus[1] = 0.01
#    mus[2] = 0.0001
#    mus[3] = 0.000001
#    mus[4] = 0.00000001
#    for k in xrange(0, num_mus - 1):
#        tau = 2.0 / (float(k) + 3.0)
#        mus[k + 1] = (1.0 - tau) * mus[k]

#    r = 0
#    for i in xrange(X.shape[1]):
#        r = max(r, abs(utils.cov(X[:, [i]], y)))
#    mus = [r * 0.5 ** i for i in xrange(num_mus)]

    total_start = time()
    init_start = time()

#    lrtv = methods.LinearRegressionTV((pz, py, px), gamma, mu, mask1D)
    mask1D = mask1D.flatten().astype(int).tolist()
    preprocess_mask = preprocess.Mask(mask1D)
#    X = preprocess_mask.process(X)

    lrtv = methods.LinearRegressionTV(gamma, (pz, py, px), mu=mus[0])#,
                                      #mask=mask1D)
    lrtv.set_max_iter(maxit)
    lrtv.set_tolerance(eps)
    method = lrtv
#    cr = methods.ContinuationRun(lrtv, mus)
#    method = cr

#    rrtv = RidgeRegressionTV(1.0 - en_lambda, gamma, (pz, py, px), mask=mask1D)
#    rrtv.set_max_iter(maxit)
#    rrtv.set_tolerance(eps)
#    method = rrtv

    print "Init time:", (time() - init_start)

#    pz = 1
#    py = 5
#    px = 5
#    X = np.ones((5, pz * py * px))
#    lrtv = methods.LinearRegressionTV(10.0, (pz, py, px), mu=10.0)
#    beta = lrtv.get_start_vector().get_vector(X)
#    print lrtv._tv.grad(beta).T
#    import scipy.sparse as sparse
#    Ax, Ay, Az = lrtv._tv.A()
#    print Ax.todense()
#    return

    method.fit(X, y)
    computed_beta = method.beta
#    computed_beta = preprocess_mask.revert(method.beta.T).T

    alg = method.get_algorithm()
    print "Algorithm:", alg
    print "Total time:", (time() - total_start)
    print "Total iterations:", alg.iterations, "(%d)" % len(alg.f)
    print "Error:", alg.f[-1]

    plot.subplot(2, 2, 1)
    plot.plot(beta1D[:, 0], '-', computed_beta[:, 0], '*')
    plot.title("Iterations: " + str(alg.iterations))

    plot.subplot(2, 2, 3)
    plot.plot(alg.f, '-b')
    plot.title("f: " + str(alg.f[-1]))

    plot.subplot(2, 2, 2)
    plot.imshow(beta,  # , extent=(x.min(), x.max(), y.max(), y.min()),
                interpolation='nearest', cmap=cm.gist_rainbow)

    plot.subplot(2, 2, 4)
    plot.imshow(np.reshape(computed_beta, (pz, py, px))[0, :, :] + mask,
                # extent=(x.min(), x.max(), y.max(), y.min()),
                interpolation='nearest', cmap=cm.gist_rainbow)
    plot.show()


if __name__ == "__main__":
    test_tv()
#    test_lasso()
#    test_lasso_tv()

#    pz = 1
#    py = 2
#    px = 3
#    p = px * py * pz
##    beta = np.ones((pz * py * px, 1))
##    X = np.reshape(xrange(p), (pz, py, px))
#    X = np.ones((5, pz * py * px))
#    print X
#    lrtv = methods.LinearRegressionTV(10.0, (pz, py, px), mu=10.0)
#    beta = 10.0 * lrtv.get_start_vector().get_vector(X)
#    print lrtv._tv.grad(beta).T
