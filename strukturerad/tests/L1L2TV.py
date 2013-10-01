# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 12:07:50 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: TBD
"""

import numpy as np
import scipy.sparse as sparse
import strukturerad.algorithms as algorithms
import strukturerad.utils as utils

import strukturerad.datasets.simulated.l1_l2_tv as l1_l2_tv
import strukturerad.datasets.simulated.l1_l2_tvmu as l1_l2_tvmu
import strukturerad.datasets.simulated.l1mu_l2_tvmu as l1mu_l2_tvmu

import strukturerad.datasets.simulated.beta as generate_beta

from strukturerad.utils import math

from time import time
import matplotlib.pyplot as plot
import matplotlib.cm as cm


np.random.seed(42)

mu_zero = 5e-8


class RidgeRegression(object):

    def __init__(self, k):

        self.k = float(k)

    """ Function value of Ridge regression.
    """
    def f(self, X, y, beta):

        return 0.5 * np.sum((np.dot(X, beta) - y) ** 2.0) \
                    + 0.5 * self.k * np.sum(beta ** 2.0)

    """ Gradient of Ridge regression
    """
    def grad(self, X, y, beta):

        return np.dot(X.T, np.dot(X, beta) - y) + self.k * beta

    ### Methods for the dual formulation ###

    def phi(self, X, y, beta):

        return self.f(X, y, self.k, beta)

    def Lipschitz(self, X):

        _, s, _ = np.linalg.svd(X, full_matrices=False)
        return np.max(s) ** 2.0 + self.k


class L1(object):

    def __init__(self, l):

        self._A = None
        self.l = float(l)

    """ Function value of L1.
    """
    def f(self, beta):

        return self.l * np.sum(np.abs(beta))

    """ Proximal operator of the L1 norm
    """
    def prox(self, x, factor=1.0):

        l = self.l * factor

        return (np.abs(x) > l) * (x - l * np.sign(x - l))

    ### Methods for the dual formulation ###

    def phi(self, beta, alpha, mu):

        return self.l * (np.dot(alpha[0].T, beta)[0, 0] \
                            - (mu / 2.0) * np.sum(alpha[0] ** 2.0))

    def grad(self, beta, mu):

        alpha = self.alpha(beta, mu)

        A = self.A(beta.shape[0])
        grad = A[0].T.dot(alpha[0])
        for i in xrange(1, len(A)):
            grad += A[i].T.dot(alpha[i])

        return self.l * grad

    def A(self, p=None):

        if self._A == None:
            self._A = sparse.eye(p, p)
        return [self._A]

    def Aa(self, alpha):

        return alpha[0]

    def alpha(self, beta, mu):

        # Compute a*
        alpha = self.project([beta / mu])

        return alpha

    def project(self, a):

        a = a[0]
        anorm = np.abs(a)
        i = anorm > 1.0
        anorm_i = anorm[i]
        a[i] = np.divide(a[i], anorm_i)

        return [a]

    def Lipschitz(self, mu):

        return self.l / mu


class TotalVariation(object):

    def __init__(self, g, shape):

        self.g = float(g)
        self._A = self.precompute(shape, mask=None, compress=False)
        self._lmaxA = None

    """ Function value of Ridge regression.
    """
    def f(self, X, y, beta, mu):

        if self.g < utils.TOLERANCE:
            return 0.0

        alpha = self.alpha(beta, mu)
        Aa = self.Aa(alpha)

        alpha_sqsum = 0.0
        for a in alpha:
            alpha_sqsum += np.sum(a ** 2.0)

        return self.g * (np.dot(Aa.T, beta)[0, 0] - (mu / 2.0) * alpha_sqsum)

    def phi(self, X, y, beta, alpha, mu):

        Aa = self.Aa(alpha)

        alpha_sqsum = 0.0
        for a in alpha:
            alpha_sqsum += np.sum(a ** 2.0)

        return self.g * (np.dot(Aa.T, beta)[0, 0] - (mu / 2.0) * alpha_sqsum)

    """ Gradient of Total variation
    """
    def grad(self, beta, mu):

        alpha = self.alpha(beta, mu)

        A = self.A()
        grad = A[0].T.dot(alpha[0])
        for i in xrange(1, len(A)):
            grad += A[i].T.dot(alpha[i])

        return self.g * grad

    def Lipschitz(self, mu):

        if self.g < utils.TOLERANCE:
            return 0.0

        # Note that we can do this here since lmaxA does not change.
        if self._lmaxA == None:
            A = sparse.vstack(self.A())
            v = algorithms.SparseSVD(max_iter=100).run(A)
            us = A.dot(v)
            self._lmaxA = np.sum(us ** 2.0)

        return self.g * self._lmaxA / mu

    def A(self):

        return self._A

    def Aa(self, alpha):

        A = self.A()
        Aa = A[0].T.dot(alpha[0])
        for i in xrange(1, len(A)):
            Aa += A[i].T.dot(alpha[i])

        return Aa

    def alpha(self, beta, mu):

        # Compute a*
        A = self.A()
        alpha = [0] * len(A)
        for i in xrange(len(A)):
            alpha[i] = A[i].dot(beta) / mu

        # Apply projection
        alpha = self.project(alpha)

        return alpha

    def project(self, a):

        ax = a[0]
        ay = a[1]
        az = a[2]
        anorm = ax ** 2.0 + ay ** 2.0 + az ** 2.0
        i = anorm > 1.0

        anorm_i = anorm[i] ** 0.5  # Square root is taken here. Faster.
        ax[i] = np.divide(ax[i], anorm_i)
        ay[i] = np.divide(ay[i], anorm_i)
        az[i] = np.divide(az[i], anorm_i)

        return [ax, ay, az]

    def D(self):
        return self._A[0].shape[0] / 2.0

    def mu(self, beta):

        SS = 0
        A = self.A()
        for i in xrange(len(A)):
            SS += A[i].dot(beta) ** 2.0

        anorm = np.sqrt(SS)

        return np.max(anorm)

    @staticmethod
    def precompute(shape, mask=None, compress=True):

        def _find_mask_ind(mask, ind):

            xshift = np.concatenate((mask[:, :, 1:], -np.ones((mask.shape[0],
                                                              mask.shape[1],
                                                              1))),
                                    axis=2)
            yshift = np.concatenate((mask[:, 1:, :], -np.ones((mask.shape[0],
                                                              1,
                                                              mask.shape[2]))),
                                    axis=1)
            zshift = np.concatenate((mask[1:, :, :], -np.ones((1,
                                                              mask.shape[1],
                                                              mask.shape[2]))),
                                    axis=0)

            xind = ind[(mask - xshift) > 0]
            yind = ind[(mask - yshift) > 0]
            zind = ind[(mask - zshift) > 0]

            return xind.flatten().tolist(), \
                   yind.flatten().tolist(), \
                   zind.flatten().tolist()

        Z = shape[0]
        Y = shape[1]
        X = shape[2]
        p = X * Y * Z

        smtype = 'csr'
        Ax = sparse.eye(p, p, 1, format=smtype) - sparse.eye(p, p)
        Ay = sparse.eye(p, p, X, format=smtype) - sparse.eye(p, p)
        Az = sparse.eye(p, p, X * Y, format=smtype) - sparse.eye(p, p)

        ind = np.reshape(xrange(p), (Z, Y, X))
        if mask != None:
            _mask = np.reshape(mask, (Z, Y, X))
            xind, yind, zind = _find_mask_ind(_mask, ind)
        else:
            xind = ind[:, :, -1].flatten().tolist()
            yind = ind[:, -1, :].flatten().tolist()
            zind = ind[-1, :, :].flatten().tolist()

        for i in xrange(len(xind)):
            Ax.data[Ax.indptr[xind[i]]: \
                    Ax.indptr[xind[i] + 1]] = 0
        Ax.eliminate_zeros()

        for i in xrange(len(yind)):
            Ay.data[Ay.indptr[yind[i]]: \
                    Ay.indptr[yind[i] + 1]] = 0
        Ay.eliminate_zeros()

#        for i in xrange(len(zind)):
#            Az.data[Az.indptr[zind[i]]: \
#                    Az.indptr[zind[i] + 1]] = 0
        Az.data[Az.indptr[zind[0]]: \
                Az.indptr[zind[-1] + 1]] = 0
        Az.eliminate_zeros()

        # Remove rows corresponding to indices excluded in all dimensions
        if compress:
            toremove = list(set(xind).intersection(yind).intersection(zind))
            toremove.sort()
            # Remove from the end so that indices are not changed
            toremove.reverse()
            for i in toremove:
                utils.delete_sparse_csr_row(Ax, i)
                utils.delete_sparse_csr_row(Ay, i)
                utils.delete_sparse_csr_row(Az, i)

        # Remove columns of A corresponding to masked-out variables
        if mask != None:
            Ax = Ax.T.tocsr()
            Ay = Ay.T.tocsr()
            Az = Az.T.tocsr()
            for i in reversed(xrange(p)):
                # TODO: Mask should be boolean!
                if mask[i] == 0:
                    utils.delete_sparse_csr_row(Ax, i)
                    utils.delete_sparse_csr_row(Ay, i)
                    utils.delete_sparse_csr_row(Az, i)

            Ax = Ax.T
            Ay = Ay.T
            Az = Az.T

        return [Ax, Ay, Az]


class RRL1TV(object):

    def __init__(self, l, k, g, shape):

        self.rr = RidgeRegression(k)
        self.l1 = L1(l)
        self.tv = TotalVariation(g, shape=shape)

    """ Function value of Ridge regression, L1 and TV.
    """
    def f(self, X, y, beta, mu):

        return self.rr.f(X, y, beta) \
             + self.l1.f(beta) \
             + self.tv.f(X, y, beta, mu)

    """ Gradient of the differentiable part with Ridge regression + TV.
    """
    def grad(self, X, y, beta, mu):

        return self.rr.grad(X, y, beta) \
             + self.tv.grad(beta, mu)

    def Lipschitz(self, X, mu):

        return self.rr.Lipschitz(X) + self.tv.Lipschitz(mu)

    """ Proximal operator of the L1 norm.
    """
    def prox(self, x, factor=1.0):

        return self.l1.prox(x, factor)

    def mu_opt(self, eps, lmaxX):

        # g = self.tv.g
        g = 1.0  # <-- g is in lmaxA!!
        D = self.tv.D()
        lmaxA = self.tv.Lipschitz(1.0)

        return (-D * g * lmaxA + np.sqrt((D * g * lmaxA) ** 2.0 \
                    + D * lmaxX * eps * g * lmaxA)) / (D * lmaxX)

    def eps_opt(self, mu, lmaxX):

        # g = self.tv.g
        g = 1.0  # <-- g is in lmaxA!!
        D = self.tv.D()
        lmaxA = self.tv.Lipschitz(1.0)

        return (2.0 * D * g * lmaxA * mu + D * lmaxX * mu ** 2.0) / (g * lmaxA)

    """ Returns the beta that minimises the dual function.
    """
    def betahat(self, X, y, gAalpha):

        XXkI = np.dot(X.T, X) + self.rr.k * np.eye(X.shape[1])
        beta = np.dot(np.linalg.inv(XXkI), np.dot(X.T, y) - gAalpha)

        return beta

    def gap(self, X, y, beta, mu, smooth_l1=False, maxit=100):

        alpha_tv = self.tv.alpha(beta, mu)

        if smooth_l1:
            alpha_l1 = self.l1.alpha(beta, mu)

            P = self.rr.f(X, y, beta) \
              + self.l1.phi(beta, alpha_l1, mu) \
              + self.tv.phi(X, y, beta, alpha_tv, mu)

            Aa_l1 = self.l1.Aa(alpha_l1)
            Aa_tv = self.tv.Aa(alpha_tv)

            lAa_l1 = self.l1.l * Aa_l1
            gAa_tv = self.tv.g * Aa_tv
            gAa = lAa_l1 + gAa_tv
            beta_hat = self.betahat(X, y, gAa)

            D = self.rr.f(X, y, beta_hat) \
              + self.l1.phi(beta_hat, alpha_l1, mu) \
              + self.tv.phi(X, y, beta_hat, alpha_tv, mu)

        else:
            gAa = self.tv.g * self.tv.Aa(alpha_tv)

            P = self.rr.f(X, y, beta) \
              + self.l1.f(beta) \
              + self.tv.phi(X, y, beta, alpha_tv, mu)

            t = 1.0 / rrl1tv.Lipschitz(X, mu)
            beta_old = beta_new = beta
            # TODO: Use function FISTA instead!!
            for i in xrange(1, maxit):
                z = beta_new + ((i - 2.0) / (i + 1.0)) * (beta_new - beta_old)
                beta_old = beta_new

                beta_new = self.prox(z - t * (self.rr.grad(X, y, z) \
                                            + self.tv.grad(z, mu)), t)

                D = self.rr.f(X, y, beta_new) \
                  + self.l1.f(beta_new) \
                  + self.tv.phi(X, y, beta_new, alpha_tv, mu)

#                if i % (maxit / 100) == 0:
#                    print "P:", P
#                    print "D:", D
#                    print "P - D: ", P - D
#                    print "grad:", np.linalg.norm(np.dot(X.T, np.dot(X, beta_new) - y) + k * beta_new + gAa)
#            print beta_new

        return P - D

# Function value of Ridge regression, L1 and TV
#def f(X, y, l, k, g, beta, mu):
#    return rr.f(X, y, k, beta) + l1.f(l, beta) + tv.f(X, y, g, beta, mu)


## Gradient of Ridge regression + TV
#def grad(X, y, k, g, beta, mu):
#    return rr.grad(X, y, k, beta) + tv.grad(g, beta, mu)


## Proximal operator of the L1 norm
#def prox(l, x):
#    return l1.prox(l, x)


#def betahat(X, y, k, gAalpha):
#
#    XXkI = np.dot(X.T, X) + k * np.eye(X.shape[1])
#    beta = np.dot(np.linalg.inv(XXkI), np.dot(X.T, y) - gAalpha)
#
#    return beta


#def gap(X, y, l, k, g, beta, mu, smooth_l1=False, maxit=100):
#
#    alpha_tv = tv.alpha(beta, mu)
#
#    if smooth_l1:
#        alpha_l1 = l1.alpha(beta, mu)
#
#        P = rr.f(X, y, k, beta) \
#          + l1.phi(l, beta, alpha_l1, mu) \
#          + tv.phi(X, y, g, beta, alpha_tv, mu)
#
#        Aa_l1 = l1.Aa(alpha_l1)
#        Aa_tv = tv.Aa(alpha_tv)
#
#        lAa_l1 = l * Aa_l1
#        gAa_tv = g * Aa_tv
#        gAa = lAa_l1 + gAa_tv
#        beta_hat = self.betahat(X, y, k, gAa)
#
#        D = rr.f(X, y, k, beta_hat) \
#          + l1.phi(l, beta_hat, alpha_l1, mu) \
#          + tv.phi(X, y, g, beta_hat, alpha_tv, mu)
#
#    else:
#        gAa = g * tv.Aa(alpha_tv)
#
#        P = rr.f(X, y, k, beta) \
#          + l1.f(l, beta) \
#          + tv.phi(X, y, g, beta, alpha_tv, mu)
#
#        t = 1.0 / Lipschitz(X, k, g, mu)
#        beta_old = beta_new = beta
#        # TODO: Use function FISTA instead!!
#        for i in range(1, maxit):
#            z = beta_new + ((i - 2.0) / (i + 1.0)) * (beta_new - beta_old)
#            beta_old = beta_new
#
#            beta_new = rrl1tv.l1.prox(l * t, z - t * (rr.grad(X, y, k, z) + \
#                                                      tv.grad(g, z, mu)))
#
#            D = rr.f(X, y, k, beta_new) \
#              + l1.f(l, beta_new) \
#              + tv.phi(X, y, g, beta_new, alpha_tv, mu)
#
##            if i % (maxit / 100) == 0:
##                print "P:", P
##                print "D:", D
##                print "P - D: ", P - D
##                print "grad:", np.linalg.norm(np.dot(X.T, np.dot(X, beta_new) - y) + k * beta_new + gAa)
##        print beta_new
#
#    return P - D


#def Lipschitz(X, k, g, mu):
#
#    return rr.Lipschitz(X, k) + tv.Lipschitz(g, mu)


#def mu_opt(eps, g, D, lmaxX, lmaxA):
#    return (-D * g * lmaxA + np.sqrt((D * g * lmaxA) ** 2.0 \
#                + D * lmaxX * eps * g * lmaxA)) / (D * lmaxX)
#
#
#def eps_opt(mu, g, D, lmaxX, lmaxA):
#    return (2.0 * D * g * lmaxA * mu + D * lmaxX * mu ** 2.0) / (g * lmaxA)


# The fast iterative shrinkage threshold algorithm
def FISTA(X, y, beta, step, mu,
          eps=utils.TOLERANCE, maxit=utils.MAX_ITER):

    z = betanew = betaold = beta

    crit = []
    for i in xrange(1, maxit + 1):
        z = betanew + ((i - 2.0) / (i + 1.0)) * (betanew - betaold)
        betaold = betanew
        betanew = rrl1tv.prox(z - step * rrl1tv.grad(X, y, z, mu), step)

        crit.append(rrl1tv.f(X, y, betanew, mu=mu))

        if math.norm(betanew - z) < eps * step:
            break

    return (betanew, crit)


def CONESTA(X, y, beta, mumin=utils.TOLERANCE, sigma=1.1,
            tau=0.5, eps=utils.TOLERANCE, conts=50, maxit=1000,
            dynamic=True):

    lmaxX = rrl1tv.rr.Lipschitz(X)
#    lmaxA = rrl1tv.tv.Lipschitz(1.0)

#    mu = [mu0]
#    print "mu:", mu[0]
    mu = [0.9 * rrl1tv.tv.mu(beta0)]
    print "mu:", mu[0]
    eps0 = rrl1tv.eps_opt(mu[0], lmaxX)
    print "eps:", eps0

    t = []
    tmin = 1.0 / rrl1tv.Lipschitz(X, mumin)

    fval = []
    Gval = []

    i = 0
    while True:
        t.append(1.0 / rrl1tv.Lipschitz(X, mu[i]))
#        lmaxA = tv.Lipschitz(g, mu[i])
        eps_plus = rrl1tv.eps_opt(mu[i], lmaxX)
        print "eps_plus: ", eps_plus
        (beta, crit) = FISTA(X, y, beta, t[i], mu[i],
                             eps=eps_plus, maxit=maxit)
        print "crit: ", crit[-1]
        print "it: ", len(crit)
        fval.append(crit)

        mumin = min(mumin, mu[i])
        tmin = min(tmin, t[i])
        beta_tilde = rrl1tv.prox(beta - tmin * rrl1tv.grad(X, y,
                                                           beta, mumin), tmin)

        if math.norm(beta - beta_tilde) < tmin * eps:
            break

        if i >= conts:
            break

        if dynamic:
            G = rrl1tv.gap(X, y, beta, mu[i])
            print "Gap:", G
            G = abs(G) / sigma

            if G <= utils.TOLERANCE and mu[i] <= utils.TOLERANCE:
                break

            Gval.append(G)
#            lmaxA = tv.Lipschitz(g, mu[i])
            mu_new = min(mu[i], rrl1tv.mu_opt(G, lmaxX))
            mu.append(max(mumin, mu_new))
        else:
#            lmaxA = tv.Lipschitz(g, mu[i])
            mu_new = rrl1tv.mu_opt(eps0 * tau ** (i + 1), lmaxX)
            mu.append(max(mumin, mu_new))

        print "mu:", mu[i + 1]

        i = i + 1

    return (beta, fval, Gval)


def U(a, b):
    t = max(a, b)
    a = float(min(a, b))
    b = float(t)
    return (np.random.rand() * (b - a)) + a


np.random.seed(42)

l = 0.5  # 0.61803
k = 0.5  # 0.271828
g = 1.0  # 3.14159

px = 6
py = 1
pz = 1
shape = (pz, py, px)
p = np.prod(shape)
n = 5

rrl1tv = RRL1TV(l, k, g, shape=shape)

a = 1.0
Sigma = a * np.eye(p) + (1.0 - a) * np.ones((p, p))
Mu = np.zeros(p)
M = np.random.multivariate_normal(Mu, Sigma, n)
e = np.random.randn(n, 1)
density = 0.5

eps = 1e-4
mu = 0.9 * (2.0 * eps / p)
print "mu:", mu
conts = 20
maxit = 1000

snr = 100.0

betastar = generate_beta.rand(shape, density=density, sort=True)
print betastar

X, y, betastar = l1_l2_tv.load(l, k, g, betastar, M, e, snr, shape)

print "snr = %.5f = %.5f = |X.b| / |e| = %.5f / %.5f" \
       % (snr, np.linalg.norm(np.dot(X, betastar)) / np.linalg.norm(e),
          np.linalg.norm(np.dot(X, betastar)), np.linalg.norm(e))


#X, y, betastar = l1_l2_tvmu.load(l, k, g, betastar, M, e, mu, snr, shape)
#Aa_l1 = l1.Aa(l1.alpha(betastar, mu_zero))
#Aa_tv = tv.Aa(tv.alpha(betastar, mu_zero))
#Aa_ = grad_TVmu(betastar, shape, mu)
#X, y = l1mu_l2_tvmu.load(l, k, g, betastar, M, e, Aa_l1, Aa_tv)

#Aa_l1 = l1.Aa(l1.alpha(betastar, mu_zero))
#Aa_ = grad_L1mu(np.reshape(betastar, (p, 1)), mu)

#print "err:", np.sum((np.dot(X, betastar) - y) ** 2.0)
#print "e  :", np.sum(e ** 2.0)
#print "Xb :", np.sum(np.dot(X, betastar) ** 2.0)
#print "y  :", np.sum(y ** 2.0)
#mus = [1e+3, 1e+2, 1e+1, 1e+0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, mu_zero]
#for mu_ in mus:
#    G = gap(X, y, l, k, g, betastar, mu_, maxit=100)
#    print "mu:", mu_, ", gap: ", G

beta0 = np.random.rand(*betastar.shape)

v = []
x = []
fval = []
beta_opt = 0

num_lin = 7
val = g
vals = np.maximum(0.0, np.linspace(val - val * 0.1, val + val * 0.1, num_lin))
beta = beta0
step = 1.0 / rrl1tv.Lipschitz(X, mu)
beta, _ = FISTA(X, y, beta, step, mu=100.0 * mu, eps=eps, maxit=10000)

t__ = time()
for i in range(len(vals)):
    val = vals[i]
    l_ = l
    k_ = k
    g_ = val

#    beta = beta0    
#    beta, crit = FISTA(X, y, l_, k_, g_, beta, step, mu=mu*100000.0, eps=eps, maxit=10000)
#    print "f:", f(X, y, l_, k_, g_, beta, mu=mu)
#    beta, crit = FISTA(X, y, l_, k_, g_, beta, step, mu=mu*10000.0, eps=eps, maxit=10000)
#    print "f:", f(X, y, l_, k_, g_, beta, mu=mu)
#    beta, crit = FISTA(X, y, l_, k_, g_, beta, step, mu=mu*1000.0, eps=eps, maxit=1000)
#    print "f:", f(X, y, l_, k_, g_, beta, mu=mu)
#    beta, crit = FISTA(X, y, l_, k_, g_, beta, step, mu=mu*100.0, eps=eps, maxit=1000)
#    print "f:", f(X, y, l_, k_, g_, beta, mu=mu)
#    beta, crit = FISTA(X, y, l_, k_, g_, beta, step, mu=mu*10.0, eps=eps, maxit=1000)
#    print "f:", f(X, y, l_, k_, g_, beta, mu=mu)
#    step = 1.0 / Lipschitz(X, k_, g_, mu)
#    beta, crit = FISTA(X, y, l_, k_, g_, beta, step, mu=mu, eps=eps, maxit=conts * maxit)
#    print "f:", rrl1tv.f(X, y, l_, k_, g_, beta, mu=mu)
#    beta0 = beta

    rrl1tv.rr.k = k_
    rrl1tv.l1.l = l_
    rrl1tv.tv.g = g_
    (beta, fval_, Gval_) = CONESTA(X, y, beta,
                                   mumin=mu_zero, sigma=1.01, tau=0.5,
                                   eps=utils.TOLERANCE,
                                   conts=conts, maxit=maxit, dynamic=True)

#    print "FISTA:  ", f(X, y, l, k, g, beta, mu=mu)
#    print "CONESTA:", f(X, y, l, k, g, beta_test, mu=mu)

    curr_val = np.sum((beta - betastar) ** 2.0)
    rrl1tv.rr.k = k
    rrl1tv.l1.l = l
    rrl1tv.tv.g = g
    f_ = rrl1tv.f(X, y, beta, mu=mu)

    rrl1tv.rr.k = k_
    rrl1tv.l1.l = l_
    rrl1tv.tv.g = g_
    print "rr:", rrl1tv.rr.f(X, y, beta)
    print "l1:", rrl1tv.l1.f(beta)
    print "tv:", rrl1tv.tv.f(X, y, beta, mu)

    v.append(curr_val)
    x.append(val)
    fval.append(f_)

    if curr_val <= min(v):
        beta_opt = beta
        fbest = fval_
        Gbest = Gval_
#        beta_test_opt = beta_test

    print "true = %.5f => %.7f" % (val, curr_val)

print "time:", (time() - t__)

rrl1tv.rr.k = k
rrl1tv.l1.l = l
rrl1tv.tv.g = g
print "best  f:", rrl1tv.f(X, y, betastar, mu=mu)
print "found f:", rrl1tv.f(X, y, beta_opt, mu=mu)
print "least f:", min(fval)

plot.subplot(3, 2, 1)
plot.plot(x, v, '-b')
plot.title("true: %.5f, min: %.5f" % (g, x[np.argmin(v)]))

plot.subplot(3, 2, 3)
plot.plot(x, fval, '-b')
plot.title("true: %.5f, min: %.5f" % (g, x[np.argmin(fval)]))

plot.subplot(3, 2, 5)
plot.plot(betastar, '-g', beta_opt, '-r')
#plot.plot(betastar, '-g', beta_opt, '-r', beta_test_opt, '--b')

plot.subplot(3, 2, 2)
print fbest
fbest = sum(fbest, [])
print fbest
plot.plot(fbest, '-b')

plot.subplot(3, 2, 4)
print "Gbest:", Gbest
#Gbest = sum(Gbest, [])
plot.plot(Gbest, '-b')

plot.show()










#conts = 100
#maxit = 100
#mu = 1e-0
#eps = utils.TOLERANCE
#
#t = time()
#step = 1.0 / Lipschitz(X, k, g, mu)
#beta, crit, critmu = FISTA(X, y, l, k, g, beta0, step, mu=mu, eps=eps, maxit=conts * maxit)
#print "Time:", (time() - t)
#print "beta - betastar:", np.sum((beta - betastar) ** 2.0)
#print "f(betastar) = ", f(X, y, l, k, g, betastar, mu=mu_zero)
#print "f(beta) = ", f(X, y, l, k, g, beta, mu=mu_zero)
#print "f(betastar, mu) = ", f(X, y, l, k, g, betastar, mu=mu)
#print "f(beta, mu) = ", f(X, y, l, k, g, beta, mu=mu)
#fstar = f(X, y, l, k, g, betastar, mu=mu_zero)
#print "err:", f(X, y, l, k, g, beta, mu=mu_zero) - fstar
#
#rand_point = np.random.randn(*betastar.shape)
#rand_point_less = betastar + 0.005 * np.random.randn(*betastar.shape)
#print "Gap at beta* = ", gap_function(X, y, k, g, betastar, mu=mu_zero)
#print "... and at a random point = ", gap_function(X, y, k, g, rand_point, mu=mu_zero)
#print "... and at a less random point = ", gap_function(X, y, k, g, rand_point_less, mu=mu_zero)
#
#print "Gapmu at beta* = ", gap_function(X, y, k, g, betastar, mu=mu)
#print "... and at a random point = ", gap_function(X, y, k, g, rand_point, mu=mu)
#print "... and at a less random point = ", gap_function(X, y, k, g, rand_point_less, mu=mu)
#
#plot.subplot(2, 1, 1)
#plot.loglog(range(1, len(crit) + 1), crit, '-b')
#plot.loglog(np.asarray([1, len(crit) - 1]), np.asarray([fstar, fstar]), '--g')
#plot.title("Function value")
#
#plot.subplot(2, 1, 2)
#plot.plot(betastar, 'g')
#plot.plot(beta, 'b')
#plot.show()

#t = time()
#beta, gapvec, gapmuvec, mu, crit, critmu = conesmo(X, y, l, k, beta0, eps, conts=conts, maxit=maxit)
##mu = 5e-2
##beta, crit, critmu = FISTAmu(X, y, l, k, beta0, epsilon=eps, maxit=it, mu=mu)
#print "Time:", (time() - t)
#print "last mu: ", mu
#print "beta - betastar:", np.sum((beta - betastar) ** 2.0)
#print "f(betastar) = ", f(X, y, l, k, g, betastar)
#print "f(beta) = ", f(X, y, l, k, g, beta)
##print "fmu(betastar) = ", fmu(X, y, l, k, betastar, mu)
#print "phi(betastar) = ", phi(X, y, l, k, g, betastar, mu=mu)
##print "fmu(beta) = ", fmu(X, y, l, k, beta, mu)
#print "phi(beta) = ", phi(X, y, l, k, g, beta, mu=mu)
#fstar = f(X, y, l, k, g, betastar)
#print "err: ", f(X, y, l, k, g, beta) - fstar
#print "gap:", gap_function(X, y, l, k, beta)
#print "gapmu:", gap_mu_function(X, y, l, k, beta, mu)
#
#print "Gap at beta* = ", gap_function(X, y, l, k, betastar)
#print "... and at a random point = ", gap_function(X, y, l, k, np.random.randn(*betastar.shape))
#print "... and at a less random point = ", gap_function(X, y, l, k, betastar + 0.005 * np.random.randn(*betastar.shape))
#print "Gapmu at beta* = ", gap_mu_function(X, y, l, k, betastar, mu)
#print "... and at a random point = ", gap_mu_function(X, y, l, k, np.random.randn(*betastar.shape), mu)
#print "... and at a less random point = ", gap_mu_function(X, y, l, k, betastar + 0.005 * np.random.randn(*betastar.shape), mu)
#
#plot.subplot(2, 2, 2)
#plot.loglog(range(1, len(crit) + 1), crit, '-g')
#plot.loglog(range(1, len(critmu) + 1), critmu, '-b')
#plot.loglog(np.asarray([1, len(crit) - 1]), np.asarray([fstar, fstar]), '--r')
#plot.title("Function value")
#
#plot.subplot(2, 2, 3)
#plot.plot(betastar, 'g')
#plot.plot(beta, 'b')
#
#plot.subplot(2, 2, 4)
#plot.plot(gapvec, 'g')
#plot.plot(gapmuvec, 'b')
#
#plot.show()
#
##plot.subplot(3,1,2)
##crit = sum(crit, [])
##critmu = sum(critmu, [])
###print crit
##print critmu
##plot.semilogy(range(1, len(crit) + 1), crit, '-b')
##f_ = f(X, y, betastar, l, k)
##plot.semilogy(np.asarray([1, len(crit) - 1]), np.asarray([f_, f_]), '--g')
##plot.semilogy(range(1, len(critmu) + 1), critmu, '-b')
##f_ = fmu(X, y, betastar, l, k, mu)
##plot.semilogy(np.asarray([1, len(critmu) - 1]), np.asarray([f_, f_]), '--g')
##plot.title("Function value")
##
##plot.subplot(3,1,3)
##plot.semilogy(gapvec, 'g')
##plot.semilogy(gapmuvec, 'b')
##
##
##plot.show()