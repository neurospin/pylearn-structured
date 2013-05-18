# -*- coding: utf-8 -*-
"""
Created on Thu May 16 09:41:13 2013

@author: Tommy Löfstedt
"""

import numpy as np
import multiblock.utils as utils
from multiblock import *
import multiblock.methods as methods
#import multiblock.start_vectors as start_vectors
#import multiblock.prox_ops as prox_ops
#import multiblock.schemes as schemes
#import multiblock.error_functions as error_functions
#from sklearn.datasets import load_linnerud
from time import time

import matplotlib.pyplot as plot
import matplotlib.cm as cm
#import pylab


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

#    u = np.random.randn(p, p)
    u = np.eye(p,p)
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

    eps = 0.0000001
    maxit = 1000000

    gamma = 1.0
    l = 0.1

#    r = 0
#    for i in xrange(X.shape[1]):
#        r = max(r, abs(utils.cov(X[:, [i]], y)))
#    mus = [r * 0.5 ** i for i in xrange(5)]
    mu = 1.0

    total_start = time()
    init_start = time()

#    lrtv = methods.LinearRegressionTV((pz, py, px), gamma, mu, mask1D)
    mask1D = mask1D.flatten().astype(int).tolist()
    lrtv = methods.LinearRegressionTV((pz, py, px), gamma, mu, mask=mask1D,
                                      algorithm=algorithms.ISTARegression())
    lrtv.set_max_iter(maxit)
    lrtv.set_tolerance(eps)

    print "Init time:", (time() - init_start)

    lrtv.fit(X, y)
    alg = lrtv.get_algorithm()
    print "alg:", alg

    print "Total time:", (time() - total_start)
    print "Total iterations:", alg.iterations
    print "Total iterations:", len(alg.f)
    print "Error:", alg.f[-1]

    plot.subplot(2, 2, 1)
    plot.plot(beta1D[:, 0], '-', lrtv.beta[:, 0], '*')
    plot.title("Iterations: " + str(alg.iterations))

    plot.subplot(2, 2, 3)
    plot.plot(alg.f, '-b')
    plot.title("f: " + str(alg.f[-1]))

    plot.subplot(2, 2, 2)
    plot.imshow(beta,  # , extent=(x.min(), x.max(), y.max(), y.min()),
                interpolation='nearest', cmap=cm.gist_rainbow)

    plot.subplot(2, 2, 4)
    plot.imshow(np.reshape(lrtv.beta, (pz, py, px))[0, :, :] + mask,
                # extent=(x.min(), x.max(), y.max(), y.min()),
                interpolation='nearest', cmap=cm.gist_rainbow)
    plot.show()


if __name__ == "__main__":
    test_tv()