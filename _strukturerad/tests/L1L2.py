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
#beta0 = np.ones(betastar.shape)


class RidgeRegression(object):

    def __init__(self):
        pass

    """ Function value of Ridge regression.
    """
    def f(self, X, y, k, beta):
        return 0.5 * np.sum((np.dot(X, beta) - y) ** 2.0) \
                    + 0.5 * k * np.sum(beta ** 2.0)

    """ Gradient of Ridge regression
    """
    def grad(self, X, y, k, beta):
        return np.dot(X.T, np.dot(X, beta) - y) + k * beta

    ### Methods for the dual formulation ###

    def phi(self, X, y, k, beta, mu=0.0):
        return self.f(X, y, k, beta)


class L1(object):

    def __init__(self):
        pass

    """ Function value of L1.
    """
    def f(self, l, beta):
        return l * np.sum(np.abs(beta))

    """ Proximal operator of the L1 norm
    """
    def prox(self, l, x):
        return (np.abs(x) > l) * (x - l * np.sign(x - l))

    ### Methods for the dual formulation ###

#    def fmu(self, beta, alpha, l, mu):
#        return l * (np.dot(alpha.T, beta)[0, 0] \
#                - 0.5 * mu * np.sum(alpha ** 2.0))

    def phi(self, l, beta, alpha, mu=0.0):
#        return l * np.dot(alpha.T, beta)[0, 0]
        return l * (np.dot(alpha.T, beta)[0, 0] \
                - 0.5 * mu * np.sum(alpha ** 2.0))

    def project(self, a):
        anorm = np.abs(a)
        i = anorm > 1.0
        anorm_i = anorm[i]
        a[i] = np.divide(a[i], anorm_i)

        return a


rr = RidgeRegression()
l1 = L1()


# Function value of Ridge regression and L1
def f(X, y, l, k, beta):
#    return 0.5 * np.sum((np.dot(X, beta) - y) ** 2.0) \
#                + l * np.sum(np.abs(beta)) \
#                + 0.5 * k * np.sum(beta ** 2.0)
    return rr.f(X, y, k, beta) + l1.f(l, beta)


# Dual function value of Ridge regression and smoothed L1
def phi(X, y, l, k, beta, alpha=None, mu=0.0):
#    return 0.5 * np.sum((np.dot(X, beta) - y) ** 2.0) \
#                + 0.5 * k * np.sum(beta ** 2.0) \
#                + l * np.dot(alpha.T, beta)[0, 0]
#    return rr.phi(X, y, k, beta) + l1.phi(l, beta, alpha)

#def fmu(X, y, l, k, beta, alpha=None, mu=0.0):
    if alpha == None and mu > 0.0:
        alpha = l1.project(beta / mu)
#    return 0.5 * np.sum((np.dot(X, beta) - y) ** 2.0) \
#                + 0.5 * k * np.sum(beta ** 2.0) \
#                + l * (np.dot(beta.T, alphastar)[0, 0] \
#                        - 0.5 * mu * np.sum(alphastar ** 2.0))
#    return rr.f(X, y, beta, k) + l1.fmu(beta, alphastar, l, mu)
    return rr.phi(X, y, k, beta) + l1.phi(l, beta, alpha, mu)


# Gradient of Ridge regression
def grad(X, y, k, beta, mu=0.0):
#    return np.dot((np.dot(X, beta) - y).T, X).T + k * beta
    return rr.grad(X, y, k, beta)


# Gradient of Ridge regression and smoothed L1
def gradmu(X, y, l, k, beta, mu):
    return rr.grad(X, y, k, beta) + l * l1.project(beta / mu)


# Proximal operator of the L1 norm
def prox(l, x):
#    return (np.abs(x) > l) * (x - l * np.sign(x - l))
    return l1.prox(l, x)


def betahat(X, y, l, k, beta, alpha):
    XXkI = np.dot(X.T, X) + k * np.eye(X.shape[1])
    betahatk = np.dot(np.linalg.pinv(XXkI), np.dot(X.T, y) - l * alpha)

    return betahatk


def gap_function(X, y, l, k, beta):
#    gradbetak = (-1.0 / l) * (np.dot(X.T, np.dot(X, beta) - y) + k * beta)
#    alphak = min(1.0, l / np.max(np.abs(gradbetak))) * gradbetak
    gradbetak = rr.grad(X, y, k, beta)
    alphak = -min(1.0, 1.0 / np.max(np.abs(gradbetak))) * gradbetak
#    alphak = -sinf(gradbetak)

#    XXkI = np.dot(X.T, X) + k * np.eye(X.shape[1])
#    betahatk = np.dot(np.linalg.pinv(XXkI), np.dot(X.T, y) - l * alphak)
    betahatk = betahat(X, y, l, k, beta, alphak)

    return phi(X, y, l, k, beta, alphak) - phi(X, y, l, k, betahatk, alphak)


def gap_mu_function(X, y, l, k, beta, mu):
    alphak = l1.project(beta / mu)
    gradbetak = rr.grad(X, y, k, beta)
#    gradbetak = (-1.0 / l) * (np.dot(X.T, np.dot(X, beta) - y) + k * beta)

    i = np.abs(beta) < utils.TOLERANCE
    alphak[i] = (-1.0 / l) * gradbetak[i]

#    XXkI = np.dot(X.T, X) + k * np.eye(X.shape[1])
#    betahatk = np.dot(np.linalg.pinv(XXkI), np.dot(X.T, y) - l * alphak)
    betahatk = betahat(X, y, l, k, beta, alphak)

    return phi(X, y, l, k, beta, alphak) - phi(X, y, l, k, betahatk, alphak)


def Lipschitz(X, l, k, mu=0.0):
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    if mu > 0.0:
        return np.max(s) ** 2.0 + k + l / mu
    else:
        return np.max(s) ** 2.0 + k


def mu_plus(l, p, lmax, epsilon):
    return (-p * l ** 2.0 + np.sqrt((p * l ** 2.0) ** 2.0 \
            + 2.0 * p * epsilon * lmax * l ** 2.0)) / (p * lmax * l)


#def sinf(u):
#    unorm = np.abs(u)
#    i = unorm > 1.0
#    unorm_i = unorm[i]
#    u[i] = np.divide(u[i], unorm_i)
#    return u


# The fast iterative shrinkage threshold algorithm
def FISTA(X, y, l, k, beta, step, epsilon=eps, maxit=maxit):

    unew = uold = betanew = betaold = beta

    crit = [f(X, y, l, k, beta)]
    for i in xrange(1, maxit + 1):
        uold = unew
        ii = float(i)
        unew = betanew + ((ii - 2.0) / (ii + 1.0)) * (betanew - betaold)
        betaold = betanew
        betanew = prox(step * l, unew - step * grad(X, y, k, unew))
        crit.append(f(X, y, l, k, betanew))
        if math.norm1(betanew - unew) < epsilon * step:
            break
#        if i == maxit:
#            print "k=maxit"

    return (betanew, crit)


# The fast iterative shrinkage threshold algorithm
def FISTAmu(X, y, l, k, beta, step, epsilon=eps, maxit=maxit, mu=1e-2):

    unew = uold = betanew = betaold = beta

    crit = [f(X, y, l, k, beta)]
#    critmu = [fmu(X, y, l, k, beta, mu)]
    critmu = [phi(X, y, l, k, beta, mu=mu)]
    for i in xrange(1, maxit + 1):
        uold = unew
        ii = float(i)
        unew = betanew + ((ii - 2.0) / (ii + 1.0)) * (betanew - betaold)
        betaold = betanew
        betanew = unew - step * gradmu(X, y, l, k, unew, mu)
        crit.append(f(X, y, l, k, betanew))
#        critmu.append(fmu(X, y, l, k, betanew, mu))
        critmu.append(phi(X, y, l, k, betanew, mu=mu))
        if math.norm1(betanew - unew) < epsilon * step:
            break
#        if i == maxit:
#            print "k=maxit"

    return (betanew, crit, critmu)


def conesmo(X, y, l, k, beta, eps, conts=10, maxit=100):

    lambdamax = Lipschitz(X, l, k)
    mu = np.max(np.abs(math.corr(X, y)))
    print "start mu:", mu, ", eps:", eps, ", conts:", conts, "maxit:", maxit

    gap = gap_function(X, y, l, k, beta)
    gapvec = [gap]
    gapmu = gap_mu_function(X, y, l, k, beta, mu)
    gapmuvec = [gapmu]
    crit = []
    critmu = []
    it = 0
    while it < conts * maxit:
        step = 1.0 / Lipschitz(X, l, k, mu)
        (betanew, crit_, critmu_) = FISTAmu(X, y, l, k, beta, step, epsilon=0, maxit=maxit, mu=mu)
        crit += crit_
        critmu += critmu_
        beta = betanew
        gap = gap_function(X, y, l, k, beta)
        gapmu = gap_mu_function(X, y, l, k, beta, mu)
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

        it += maxit

    if it >= conts * maxit:
        print "it = maxit!"

    return (beta, gapvec, gapmuvec, mu, crit, critmu)

conts = 20
maxit = 50
mu = 0.860328859167

t = time()
step = 1.0 / Lipschitz(X, l, k, mu)
beta, crit, critmu = FISTAmu(X, y, l, k, beta0, step, epsilon=eps, maxit=conts * maxit, mu=mu)
print "Time:", (time() - t)
print "beta - betastar:", np.sum((beta - betastar) ** 2.0)
print "f(betastar) = ", f(X, y, l, k, betastar)
print "f(beta) = ", f(X, y, l, k, beta)
fstar = f(X, y, l, k, betastar)
print "err:", f(X, y, l, k, beta) - fstar

print "Gap at beta* = ", gap_function(X, y, l, k, betastar)
print "... and at a random point = ", gap_function(X, y, l, k, np.random.randn(*betastar.shape))
print "... and at a less random point = ", gap_function(X, y, l, k, betastar + 0.005 * np.random.randn(*betastar.shape))

plot.subplot(2, 2, 1)
plot.loglog(range(1, len(crit) + 1), crit, '-b')
plot.loglog(np.asarray([1, len(crit) - 1]), np.asarray([fstar, fstar]), '--g')
plot.title("Function value")
#plot.show()

t = time()
beta, gapvec, gapmuvec, mu, crit, critmu = conesmo(X, y, l, k, beta0, eps, conts=conts, maxit=maxit)
#mu = 5e-2
#beta, crit, critmu = FISTAmu(X, y, l, k, beta0, epsilon=eps, maxit=it, mu=mu)
print "Time:", (time() - t)
print "last mu: ", mu
print "beta - betastar:", np.sum((beta - betastar) ** 2.0)
print "f(betastar) = ", f(X, y, l, k, betastar)
print "f(beta) = ", f(X, y, l, k, beta)
#print "fmu(betastar) = ", fmu(X, y, l, k, betastar, mu)
print "phi(betastar) = ", phi(X, y, l, k, betastar, mu=mu)
#print "fmu(beta) = ", fmu(X, y, l, k, beta, mu)
print "phi(beta) = ", phi(X, y, l, k, beta, mu=mu)
fstar = f(X, y, l, k, betastar)
print "err: ", f(X, y, l, k, beta) - fstar
print "gap:", gap_function(X, y, l, k, beta)
print "gapmu:", gap_mu_function(X, y, l, k, beta, mu)

print "Gap at beta* = ", gap_function(X, y, l, k, betastar)
print "... and at a random point = ", gap_function(X, y, l, k, np.random.randn(*betastar.shape))
print "... and at a less random point = ", gap_function(X, y, l, k, betastar + 0.005 * np.random.randn(*betastar.shape))
print "Gapmu at beta* = ", gap_mu_function(X, y, l, k, betastar, mu)
print "... and at a random point = ", gap_mu_function(X, y, l, k, np.random.randn(*betastar.shape), mu)
print "... and at a less random point = ", gap_mu_function(X, y, l, k, betastar + 0.005 * np.random.randn(*betastar.shape), mu)

plot.subplot(2, 2, 2)
plot.loglog(range(1, len(crit) + 1), crit, '-g')
plot.loglog(range(1, len(critmu) + 1), critmu, '-b')
plot.loglog(np.asarray([1, len(crit) - 1]), np.asarray([fstar, fstar]), '--r')
plot.title("Function value")

plot.subplot(2, 2, 3)
plot.plot(betastar, 'g')
plot.plot(beta, 'b')

plot.subplot(2, 2, 4)
plot.plot(gapvec, 'g')
plot.plot(gapmuvec, 'b')

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