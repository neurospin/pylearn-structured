# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 13:30:56 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: TBD
"""

import numpy as np
import strukturerad.utils as utils
import strukturerad.datasets.simulated.l1_l2 as l1_l2

from strukturerad.utils import math

from time import time
import matplotlib.pyplot as plot
import matplotlib.cm as cm


np.random.seed(42)

eps = 1e-5
maxit = 5000

#px = 100
#py = 1
#pz = 1
#p = px * py * pz  # Must be even!
p = 20
n = 30
#X = np.random.randn(n, p)
#betastar = np.concatenate((np.zeros((p / 2, 1)),
#                           np.random.randn(p / 2, 1)))
#betastar = np.sort(np.abs(betastar), axis=0)
#y = np.dot(X, betastar)

e = np.random.randn(n, 1) * 0.5
S = 0.5 * np.ones((p, p)) + 0.5 * np.eye(p, p)
M = np.random.multivariate_normal(np.zeros(p), S, n)
density = 0.5
l = 3.0
k = 0.0
snr = 286.7992
X, y, betastar = l1_l2.load(l, k, density, snr, M, e)
beta0 = np.random.randn(*betastar.shape)

#gamma = 0.0
#mu = 0.01


# Function value of Ridge regression and L1
def f(X, y, beta, l, k):
    return 0.5 * np.sum((np.dot(X, beta) - y) ** 2.0) \
                + l * np.sum(np.abs(beta)) \
                + 0.5 * k * np.sum(beta ** 2.0)


# Proximal operator of the L1 norm
def prox(x, l):
    return (np.abs(x) > l) * (x - l * np.sign(x - l))


# Gradient of Ridge regression
def grad(X, y, beta, k):
    return np.dot((np.dot(X, beta) - y).T, X).T + k * beta


# The fast iterative shrinkage threshold algorithm
def FISTA(X, y, l, k, beta, const=None, epsilon=eps, maxit=maxit):
    if const == None:
        _, s, _ = np.linalg.svd(X, full_matrices=False)
        const = np.max(s) ** 2.0 + k
#        l_min = np.min(s) ** 2.0 + k
        const = 1.0 / const

    unew = uold = betanew = betaold = beta

    crit = [f(X, y, beta, l, k)]
    for i in xrange(1, maxit):
        uold = unew
        ii = float(i)
        unew = betanew + ((ii - 2.0) / (ii + 1.0)) * (betanew - betaold)
        betaold = betanew
        betanew = prox(unew - const * grad(X, y, unew, k), const * l)
        crit.append(f(X, y, betanew, l, k))
        if math.norm1(betanew - unew) < utils.TOLERANCE * const:
            break
        if i == maxit - 1:
            print "k=maxit"

    return (betanew, crit)


def phi(X, y, beta, alpha, l, k):
    return 0.5 * np.sum((np.dot(X, beta) - y) ** 2.0) \
                + 0.5 * k * np.sum(beta ** 2.0) \
                + l * np.dot(alpha.T, beta)[0, 0]


def gap_function(X, y, beta, l, k):
    gradbetak = (-1.0 / l) * (np.dot(X.T, np.dot(X, beta) - y) + k * beta)
    alphak = min(1.0, l / np.max(np.abs(gradbetak))) * gradbetak
    XXkI = np.dot(X.T, X) + k * np.eye(X.shape[1])
    betahatk = np.dot(np.linalg.inv(XXkI), np.dot(X.T, y) - l * alphak)

    return phi(X, y, beta, alphak, l, k) - phi(X, y, betahatk, alphak, l, k)

#t = time()
#beta, crit = FISTA(X, y, l, k, beta0, epsilon=eps)
#print(time() - t)
#print beta - betastar
#fstar = f(X, y, betastar, l, k)
#print f(X, y, beta, l, k) - fstar
#
##plot.subplot(1, 1, 2)
#plot.loglog(range(1, len(crit) + 1), crit, '-b')
#plot.loglog(np.asarray([1, len(crit) - 1]), np.asarray([crit[-1], crit[-1]]), '--g')
#plot.title("Function value")
#plot.show()

print "Gap at beta* = ", gap_function(X, y, betastar, l, k)
print "... and at a random point = ", gap_function(X, y, np.random.randn(*betastar.shape), l, k)
print "... and at a less random point = ", gap_function(X, y, betastar + 0.005 * np.random.randn(*betastar.shape), l, k)