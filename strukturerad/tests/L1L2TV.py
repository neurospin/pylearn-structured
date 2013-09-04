# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 12:07:50 2013

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
#beta0 = np.ones(betastar.shape)


class RidgeRegression(object):

    def __init__(self):
        pass

    """ Function value of Ridge regression.
    """
    def f(self, X, y, beta, k):
        return 0.5 * np.sum((np.dot(X, beta) - y) ** 2.0) \
                    + 0.5 * k * np.sum(beta ** 2.0)

    """ Gradient of Ridge regression
    """
    def grad(self, X, y, beta, k):
        return np.dot((np.dot(X, beta) - y).T, X).T + k * beta


class L1(object):

    def __init__(self):
        pass

    """ Function value of L1.
    """
    def f(self, beta, l):
        return l * np.sum(np.abs(beta))

    """ Proximal operator of the L1 norm
    """
    def prox(self, x, l):
        return (np.abs(x) > l) * (x - l * np.sign(x - l))


rr = RidgeRegression()
l1 = L1()


# The fast iterative shrinkage threshold algorithm
def FISTA(X, y, beta, l, k, const=None, epsilon=eps, maxit=maxit):
    if const == None:
        _, s, _ = np.linalg.svd(X, full_matrices=False)
        const = np.max(s) ** 2.0 + k
#        l_min = np.min(s) ** 2.0 + k
        const = 1.0 / const

    unew = uold = betanew = betaold = beta

    crit = [f(X, y, beta, l, k)]
    for i in xrange(1, maxit + 1):
        uold = unew
        ii = float(i)
        unew = betanew + ((ii - 2.0) / (ii + 1.0)) * (betanew - betaold)
        betaold = betanew
        betanew = prox(unew - const * grad(X, y, unew, k), const * l)
        crit.append(f(X, y, betanew, l, k))
        if math.norm1(betanew - unew) < epsilon * const:
            break
        if i == maxit:
            print "k=maxit"

    return (betanew, crit)


def phi(X, y, beta, alpha, l, k):
    return 0.5 * np.sum((np.dot(X, beta) - y) ** 2.0) \
                + 0.5 * k * np.sum(beta ** 2.0) \
                + l * np.dot(alpha.T, beta)[0, 0]


def gap_function(X, y, beta, l, k):
#    gradbetak = (-1.0 / l) * (np.dot(X.T, np.dot(X, beta) - y) + k * beta)
#    alphak = min(1.0, l / np.max(np.abs(gradbetak))) * gradbetak
    gradbetak = np.dot(X.T, np.dot(X, beta) - y) + k * beta
    alphak = -min(1.0, 1.0 / np.max(np.abs(gradbetak))) * gradbetak

    XXkI = np.dot(X.T, X) + k * np.eye(X.shape[1])
    betahatk = np.dot(np.linalg.pinv(XXkI), np.dot(X.T, y) - l * alphak)

    return phi(X, y, beta, alphak, l, k) - phi(X, y, betahatk, alphak, l, k)


def sinf(u):
    unorm = np.abs(u)
    i = unorm > 1.0
    unorm_i = unorm[i]
    u[i] = np.divide(u[i], unorm_i)

    return u


# Function value of Ridge regression and smoothed L1
def fmu(X, y, beta, l, k, mu):
    alphastar = sinf(beta / mu)
    return 0.5 * np.sum((np.dot(X, beta) - y) ** 2.0) \
                + 0.5 * k * np.sum(beta ** 2.0) \
                + l * (np.dot(beta.T, alphastar)[0, 0] \
                        - 0.5 * mu * np.sum(alphastar ** 2.0))


# Gradient of Ridge regression and smoothed L1
def gradmu(X, y, beta, l, k, mu):
    return np.dot(X.T, np.dot(X, beta) - y) + k * beta + l * sinf(beta / mu)


# The fast iterative shrinkage threshold algorithm
def FISTAmu(X, y, l, k, beta, const=None, epsilon=eps, maxit=maxit, mu=1e-1):
    if const == None:
        _, s, _ = np.linalg.svd(X, full_matrices=False)
        const = np.max(s) ** 2.0 + k + l / mu
        const = 1.0 / const

    unew = uold = betanew = betaold = beta

    crit = [f(X, y, beta, l, k)]
    critmu = [fmu(X, y, beta, l, k, mu)]
    for i in xrange(1, maxit + 1):
        uold = unew
        ii = float(i)
        unew = betanew + ((ii - 2.0) / (ii + 1.0)) * (betanew - betaold)
        betaold = betanew
        betanew = unew - const * gradmu(X, y, unew, l, k, mu)
        crit.append(f(X, y, betanew, l, k))
        critmu.append(fmu(X, y, betanew, l, k, mu))
        if math.norm1(betanew - unew) < epsilon * const:
            break
        if i == maxit:
            print "k=maxit"

    return (betanew, crit, critmu)


def gap_mu_function(X, y, beta, l, k, mu):
    alphak = sinf(beta / mu)
    gradbetak = (-1.0 / l) * (np.dot(X.T, np.dot(X, beta) - y) + k * beta)

    i = np.abs(beta) < utils.TOLERANCE
    alphak[i] = gradbetak[i]

    XXkI = np.dot(X.T, X) + k * np.eye(X.shape[1])
    betahatk = np.dot(np.linalg.pinv(XXkI), np.dot(X.T, y) - l * alphak)

    return phi(X, y, beta, alphak, l, k) - phi(X, y, betahatk, alphak, l, k)


def mu_plus(l, p, lmax, epsilon):
    return (-p * l ** 2.0 + np.sqrt((p * l ** 2.0) ** 2.0 \
            + 2.0 * p * epsilon * lmax * l ** 2.0)) / (p * lmax * l)


def conesmo(X, y, l, k, beta, eps, maxit=10*100):
    print "eps:", eps, ", maxit:", maxit
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    lambdamax = np.max(s) ** 2.0 + k

    mu = np.max(np.abs(math.corr(X, y)))
    print "start mu:", mu

    gap = gap_function(X, y, beta, l, k)
    gapvec = [gap]
    gapmu = gap_mu_function(X, y, beta, l, k, mu)
    gapmuvec = [gapmu]
    crit = []
    critmu = []
    it = 0
    while it < maxit:
        (betanew, crit_, critmu_) = FISTAmu(X, y, l, k, beta, epsilon=0, maxit=100, mu=mu)
        crit += crit_
        critmu += critmu_
        it += 100
        beta = betanew
        gap = gap_function(X, y, beta, l, k)
        gapmu = gap_mu_function(X, y, beta, l, k, mu)
        gapvec.append(gap)
        gapmuvec.append(gapmu)
        if gap < eps:
            print "gap:", gap, ", eps:", eps
            print "Gap < epsilon!"
            break
        if gap < gapmu / 2.0:
            mu = mu / 2.0
        else:
            mu = min(mu, mu_plus(l, X.shape[1], lambdamax, gap))

    if it >= maxit:
        print "it = maxit!"
    return (beta, gapvec, gapmuvec, mu, crit, critmu)

it = 10*100

t = time()
beta, crit = FISTA(X, y, l, k, beta0, epsilon=eps, maxit=it)
print "Time:", (time() - t)
print "beta - betastar:", np.sum((beta - betastar) ** 2.0)
print "f(betastar) = ", f(X, y, betastar, l, k)
print "f(beta) = ", f(X, y, beta, l, k)
fstar = f(X, y, betastar, l, k)
print "err:", f(X, y, beta, l, k) - fstar

print "Gap at beta* = ", gap_function(X, y, betastar, l, k)
print "... and at a random point = ", gap_function(X, y, np.random.randn(*betastar.shape), l, k)
print "... and at a less random point = ", gap_function(X, y, betastar + 0.005 * np.random.randn(*betastar.shape), l, k)

plot.subplot(2, 2, 1)
plot.loglog(range(1, len(crit) + 1), crit, '-b')
plot.loglog(np.asarray([1, len(crit) - 1]), np.asarray([fstar, fstar]), '--g')
plot.title("Function value")
#plot.show()

t = time()
beta, gapvec, gapmuvec, mu, crit, critmu = conesmo(X, y, l, k, beta0, eps, maxit=it)
#mu = 5e-2
#beta, crit, critmu = FISTAmu(X, y, l, k, beta0, epsilon=eps, maxit=it, mu=mu)
print "Time:", (time() - t)
print "last mu: ", mu
print "beta - betastar:", np.sum((beta - betastar) ** 2.0)
print "f(betastar) = ", f(X, y, betastar, l, k)
print "f(beta) = ", f(X, y, beta, l, k)
print "fmu(betastar) = ", fmu(X, y, betastar, l, k, mu)
print "fmu(beta) = ", fmu(X, y, beta, l, k, mu)
fstar = f(X, y, betastar, l, k)
print "err: ", f(X, y, beta, l, k) - fstar
print "gap:", gap_function(X, y, beta, l, k)
print "gapmu:", gap_mu_function(X, y, beta, l, k, mu)

print "Gap at beta* = ", gap_function(X, y, betastar, l, k)
print "... and at a random point = ", gap_function(X, y, np.random.randn(*betastar.shape), l, k)
print "... and at a less random point = ", gap_function(X, y, betastar + 0.005 * np.random.randn(*betastar.shape), l, k)
print "Gapmu at beta* = ", gap_mu_function(X, y, betastar, l, k, mu)
print "... and at a random point = ", gap_mu_function(X, y, np.random.randn(*betastar.shape), l, k, mu)
print "... and at a less random point = ", gap_mu_function(X, y, betastar + 0.005 * np.random.randn(*betastar.shape), l, k, mu)

plot.subplot(2, 2, 2)
plot.loglog(range(1, len(crit) + 1), crit, '-g')
plot.loglog(range(1, len(critmu) + 1), critmu, '-b')
plot.loglog(np.asarray([1, len(crit) - 1]), np.asarray([fstar, fstar]), '--r')
plot.title("Function value")

plot.subplot(2, 2, 3)
plot.plot(betastar, 'g')
plot.plot(beta, 'b')

#plot.subplot(2, 2, 4)
#plot.plot(gapvec, 'g')
#plot.plot(gapmuvec, 'b')

plot.show()

#plot.subplot(3,1,2)
#crit = sum(crit, [])
#critmu = sum(critmu, [])
##print crit
#print critmu
#plot.semilogy(range(1, len(crit) + 1), crit, '-b')
#f_ = f(X, y, betastar, l, k)
#plot.semilogy(np.asarray([1, len(crit) - 1]), np.asarray([f_, f_]), '--g')
#plot.semilogy(range(1, len(critmu) + 1), critmu, '-b')
#f_ = fmu(X, y, betastar, l, k, mu)
#plot.semilogy(np.asarray([1, len(critmu) - 1]), np.asarray([f_, f_]), '--g')
#plot.title("Function value")
#
#plot.subplot(3,1,3)
#plot.semilogy(gapvec, 'g')
#plot.semilogy(gapmuvec, 'b')
#
#
#plot.show()