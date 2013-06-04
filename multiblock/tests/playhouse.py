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

    eps = 0.0001
    maxit = 500000

    px = 300
    py = 1
    pz = 1
    p = px * py * pz  # Must be even!
    n = 50
    X = np.random.randn(n, p)
    betastar = np.concatenate((np.zeros((p / 2, 1)),
                               np.random.randn(p / 2, 1)))
    betastar = np.sort(np.abs(betastar), axis=0)
    y = np.dot(X, betastar)


    gamma = 0.0
    l = 10.0
    lrtv = methods.LinearRegressionL1TV(l, gamma, (pz, py, px), mu=0.01)
    lrtv.set_max_iter(maxit)
    lrtv.set_tolerance(eps)
    method = lrtv

#    cr = methods.ContinuationRun(lrtv, mus)
#    method = cr

#    rrtv = RidgeRegressionTV(1.0 - en_lambda, gamma, (pz, py, px), mask=mask1D)
#    rrtv.set_max_iter(maxit)
#    rrtv.set_tolerance(eps)
#    method = rrtv

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
#    print "Total time:", (time() - total_start)
    print "Total iterations:", alg.iterations, "(%d)" % len(alg.f)
    print "Error:", alg.f[-1]

    plot.subplot(2, 1, 1)
    plot.plot(betastar[:, 0], '-', computed_beta[:, 0], '*')
    plot.title("Iterations: " + str(alg.iterations))

    plot.subplot(2, 1, 2)
    plot.plot(alg.f, '-b')
    plot.title("f: " + str(alg.f[-1]))

#    plot.subplot(2, 2, 2)
#    plot.imshow(beta,  # , extent=(x.min(), x.max(), y.max(), y.min()),
#                interpolation='nearest', cmap=cm.gist_rainbow)
#
#    plot.subplot(2, 2, 4)
#    plot.imshow(np.reshape(computed_beta, (pz, py, px))[0, :, :] + mask,
#                # extent=(x.min(), x.max(), y.max(), y.min()),
#                interpolation='nearest', cmap=cm.gist_rainbow)
    plot.show()




def test_tv():
#    np.random.seed(42)

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

    eps = 0.01
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
#    test_tv()
    test_lasso()