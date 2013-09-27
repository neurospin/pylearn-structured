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
#import strukturerad.datasets.simulated.l1_l2 as l1_l2
import strukturerad.datasets.simulated.l1_l2_tv as l1_l2_tv
import strukturerad.datasets.simulated.l1_l2_tvmu as l1_l2_tvmu
import strukturerad.datasets.simulated.l1_l2_tv_2D as l1_l2_tv_2D

from strukturerad.utils import math

from time import time
import matplotlib.pyplot as plot
import matplotlib.cm as cm


np.random.seed(42)

#eps = 1e-5
#maxit = 5000

#px = 20
#py = 1
#pz = 1
#p = px * py * pz  # Must be even!
#n = 30
##X = np.random.randn(n, p)
##betastar = np.concatenate((np.zeros((p / 2, 1)),
##                           np.random.randn(p / 2, 1)))
##betastar = np.sort(np.abs(betastar), axis=0)
##y = np.dot(X, betastar)
#
#e = np.random.randn(n, 1) * 0.5
#S = 0.5 * np.ones((p, p)) + 0.5 * np.eye(p, p)
#M = np.random.multivariate_normal(np.zeros(p), S, n)
#density = 0.5
#l = 3.0
#k = 0.0
#g = 1.0
#snr = 286.7992
#X, y, betastar = l1_l2.load(l, k, density, snr, M, e)
#beta0 = np.random.randn(*betastar.shape)
#beta0 = np.ones(betastar.shape)
mu_zero = 5e-8


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

    def phi(self, X, y, k, beta):
        return self.f(X, y, k, beta)

    def Lipschitz(self, X, k):
        _, s, _ = np.linalg.svd(X, full_matrices=False)
        return np.max(s) ** 2.0 + k


class TotalVariation(object):

    def __init__(self, shape):

        self._A = self.precompute(shape, mask=None, compress=False)

    """ Function value of Ridge regression.
    """
    def f(self, X, y, g, beta, mu):

        if g < utils.TOLERANCE:
            return 0.0

#        if mu > 0.0:
        alpha = self.alpha(beta, mu)
        Aa = self.Aa(alpha)

        alpha_sqsum = 0.0
        for a in alpha:
            alpha_sqsum += np.sum(a ** 2.0)

        return g * (np.dot(Aa.T, beta)[0, 0] - (mu / 2.0) * alpha_sqsum)

#        else:
#            A = self.A()
#            sqsum = np.sum(np.sqrt(A[0].dot(beta) ** 2.0 + \
#                                   A[1].dot(beta) ** 2.0 + \
#                                   A[2].dot(beta) ** 2.0))
#
#            return g * sqsum

    def phi(self, X, y, g, beta, alpha, mu):

        Aa = self.Aa(alpha)

        alpha_sqsum = 0.0
        for a in alpha:
            alpha_sqsum += np.sum(a ** 2.0)

        return g * (np.dot(Aa.T, beta)[0, 0] - (mu / 2.0) * alpha_sqsum)

    """ Gradient of Total variation
    """
    def grad(self, g, beta, mu):

        alpha = self.alpha(beta, mu)

        A = self.A()
        grad = A[0].T.dot(alpha[0])
        for i in xrange(1, len(A)):
            grad += A[i].T.dot(alpha[i])

        return g * grad

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

    def Lipschitz(self, g, mu):

        if g < utils.TOLERANCE:
            return 0.0

        A = sparse.vstack(self.A())
        v = algorithms.SparseSVD(max_iter=100).run(A)
        us = A.dot(v)
        lmax = np.sum(us ** 2.0)

        return g * lmax / mu

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


class L1(object):

    def __init__(self):

        self._A = None

    """ Function value of L1.
    """
    def f(self, l, beta, mu=0.0):

        if mu > 0.0:
            alpha = self.alpha(beta, mu)
            return l * (np.dot(alpha[0].T, beta)[0, 0] \
                    - (mu / 2.0) * np.sum(alpha[0] ** 2.0))
        else:
            return l * np.sum(np.abs(beta))

    def phi(self, l, beta, alpha, mu):

        return l * (np.dot(alpha[0].T, beta)[0, 0] \
                - (mu / 2.0) * np.sum(alpha[0] ** 2.0))

    """ Proximal operator of the L1 norm
    """
    def prox(self, l, x):

        return (np.abs(x) > l) * (x - l * np.sign(x - l))

    ### Methods for the dual formulation ###

    def grad(self, l, beta, mu):

        alpha = self.alpha(beta, mu)

        A = self.A(beta.shape[0])
        grad = A[0].T.dot(alpha[0])
        for i in xrange(1, len(A)):
            grad += A[i].T.dot(alpha[i])

        return l * grad

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

    def Lipschitz(self, l, mu):

        return float(l) / mu


# Function value of Ridge regression, L1 and TV
def f(X, y, l, k, g, beta, mu):
    return rr.f(X, y, k, beta) + l1.f(l, beta) + tv.f(X, y, g, beta, mu)


# Gradient of Ridge regression + TV
def grad(X, y, k, g, beta, mu):
    return rr.grad(X, y, k, beta) + tv.grad(g, beta, mu)


# Proximal operator of the L1 norm
def prox(l, x):
    return l1.prox(l, x)


def betahat(X, y, k, gAalpha):

    XXkI = np.dot(X.T, X) + k * np.eye(X.shape[1])
    beta = np.dot(np.linalg.inv(XXkI), np.dot(X.T, y) - gAalpha)

    return beta


def gap(X, y, l, k, g, beta, mu):

    alpha_l1 = l1.alpha(beta, mu_zero)
    alpha_tv = tv.alpha(beta, mu)

    P = rr.f(X, y, k, beta) \
      + l1.phi(l, beta, alpha_l1, mu_zero) \
      + tv.phi(X, y, g, beta, alpha_tv, mu)

    Aa_l1 = l1.Aa(alpha_l1)
    Aa_tv = tv.Aa(alpha_tv)

    lAa_l1 = l * Aa_l1
    gAa_tv = g * Aa_tv
    gAa = lAa_l1 + gAa_tv
    beta_hat = betahat(X, y, k, gAa)
    print "beta_hat:", beta_hat

    D = rr.f(X, y, k, beta_hat) \
      + l1.phi(l, beta_hat, alpha_l1, mu_zero) \
      + tv.phi(X, y, g, beta_hat, alpha_tv, mu)

    return P - D


def Lipschitz(X, k, g, mu):
    if mu > 0.0:
        return rr.Lipschitz(X, k) + tv.Lipschitz(g, mu)
    else:
        return rr.Lipschitz(X, k)


def mu_opt(eps, g, D, lmaxX, lmaxA):
    return (-D * g * lmaxA + np.sqrt((D * g * lmaxA) ** 2.0 \
                + D * lmaxX * eps * g * lmaxA)) / (D * lmaxX)


def eps_opt(mu, g, D, lmaxX, lmaxA):
    return (2.0 * D * g * lmaxA * mu + D * lmaxX * mu ** 2.0) / (g * lmaxA)


# The fast iterative shrinkage threshold algorithm
def FISTA(X, y, l, k, g, beta, step, mu,
          eps=utils.TOLERANCE, maxit=utils.MAX_ITER):

    z = betanew = betaold = beta

    crit = []
    for i in xrange(1, maxit + 1):
        z = betanew + ((i - 2.0) / (i + 1.0)) * (betanew - betaold)
        betaold = betanew
        betanew = prox(step * l, z - step * grad(X, y, k, g, z, mu))

        crit.append(f(X, y, l, k, g, betanew, mu=mu))

        if math.norm(betanew - z) < eps * step:
            break

    return (betanew, crit)


def CONESTA(X, y, l, k, g, beta, mumin=utils.TOLERANCE, sigma=1.1,
            tau=0.5, eps=utils.TOLERANCE, conts=50, maxit=1000,
            dynamic=True):

    lmaxX = rr.Lipschitz(X, k)
    lmaxA = tv.Lipschitz(g, 1.0)

#    mu = [mu0]
#    print "mu:", mu[0]
    mu = [0.9 * tv.mu(beta0)]
    print "mu:", mu[0]
    eps0 = eps_opt(mu[0], g, tv.D(), lmaxX, lmaxA)
    print "eps:", eps0

    t = []
    tmin = 1.0 / Lipschitz(X, k, g, mumin)

    fval = []
    Gval = []

    i = 0
    while True:
        t.append(1.0 / Lipschitz(X, k, g, mu[i]))
#        lmaxA = tv.Lipschitz(g, mu[i])
        eps_plus = eps_opt(mu[i], g, tv.D(), lmaxX, lmaxA)
        print "eps_plus: ", eps_plus
        print "lmaxX: ", lmaxX
        print "lmaxA: ", lmaxA
        (beta, crit) = FISTA(X, y, l, k, g, beta, t[i], mu[i],
                             eps=eps_plus, maxit=maxit)
        print "crit: ", crit[-1]
        print "it: ", len(crit)
        fval.append(crit)

        mumin = min(mumin, mu[i])
        tmin = min(tmin, t[i])
        beta_tilde = prox(tmin * l, beta - tmin * grad(X, y, k, g,
                                                       beta, mumin))

        if math.norm(beta - beta_tilde) < tmin * eps:
            break

        if i >= conts:
            break

        if dynamic:
            G = gap(X, y, l, k, g, beta, mu[i])
            print "Gap:", G
            G = abs(G) / sigma

            if G <= utils.TOLERANCE and mu[i] <= utils.TOLERANCE:
                break

            Gval.append(G)
#            lmaxA = tv.Lipschitz(g, mu[i])
            mu_new = min(mu[i], mu_opt(G, g, tv.D(), lmaxX, lmaxA))
            mu.append(max(mumin, mu_new))
        else:
#            lmaxA = tv.Lipschitz(g, mu[i])
            mu_new = mu_opt(eps0 * tau ** (i + 1), g, tv.D(), lmaxX, lmaxA)
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
g = 0.9  # 3.14159

px = 6
py = 1
pz = 1
shape=(pz, py, px)
p = np.prod(shape)
n = 5

rr = RidgeRegression()
l1 = L1()
tv = TotalVariation(shape=(pz, py, px))

a = 1.0
Sigma = a * np.eye(p) + (1.0 - a) * np.ones((p, p))
Mu = np.zeros(p)
M = np.random.multivariate_normal(Mu, Sigma, n)
e = np.random.randn(n, 1)
density=0.5

eps = 1e-3
mu = 0.9 * (2.0 * eps / p)
print "mu:", mu
conts = 1000
maxit = 1000

snr = 100.0

ps = int(round(p * density))
beta1D = np.zeros((p, 1))
for i in xrange(p):
    if i < ps:
        beta1D[i, 0] = U(0, 1) * snr / np.sqrt(ps)
    else:
        beta1D[i, 0] = 0.0

beta1D = np.flipud(np.sort(beta1D, axis=0))

#p = M.shape[1]
##px = shape[1]
##py = shape[0]
##print "p:", p, ", px:", px, ", py:", py
##print "density * p:", density * p
#s = np.sqrt(density * p / (px * py))
##print "s:", s
#part = s * s * px * py / p
##print "part:", part
#pys = int(round(py * s))
#pxs = int(round(px * s))
## Search for better approximation of px and py
#best_x = 0
#best_y = 0
#best = float("inf")
#for i in xrange(-2, 3):
#    for j in xrange(-2, 3):
#        diff = abs(((pys + i) * (pxs + j) / float(p)) - part)
#        if diff < best:
#            best = diff
#            best_x = j
#            best_y = i
##        print "%f = diff < best = %f, px = %d, py = %d" \
##                % (diff, best, (pys + i), (pxs + j))
#pys += best_y
#pxs += best_x
#
#print "pxs:", pxs
#print "pys:", pys
#print "px:", px
#print "py:", py
#
#beta2D = np.zeros((py, px))
#for i in xrange(py):
#    for j in xrange(px):
#        if i >= pys or j >= pxs:
#            beta2D[i, j] = 0.0
#        else:
#            beta2D[i, j] = U(0, 1) * snr / np.sqrt(pys * pxs)
#beta2D = np.fliplr(np.sort(np.flipud(np.sort(beta2D, axis=0)), axis=1))
#print p
#print beta2D.shape
#beta2D = np.reshape(beta2D, (p, 1))

betastar = beta1D
#betastar = beta2D
print betastar
#tv_grad = tv.grad(g, betastar, mu)
Aa = tv.Aa(tv.alpha(betastar, mu))
#X, y = l1_l2_tvmu.load(l, k, g, betastar, M, e, Aa)
#X, y = l1_l2_tv.load(l, k, g, betastar, M, e)
#X, y = l1_l2_tv_2D.load(l, k, g, betastar, M, e, shape)
X, y = l1_l2_tv.load(l, k, g, betastar, M, e, snr)

print tv.Aa(tv.alpha(betastar, mu))
print tv.grad(g, betastar, mu)

print "err:", np.sum((np.dot(X, betastar) - y) ** 2.0)
print "e  :", np.sum(e ** 2.0)
print "Xb :", np.sum(np.dot(X, betastar) ** 2.0)
print "y  :", np.sum(y ** 2.0)
G = gap(X, y, l, k, g, betastar, 1e+3)
print "gap: ", G
G = gap(X, y, l, k, g, betastar, 1e-2)
print "gap: ", G
G = gap(X, y, l, k, g, betastar, 1e-1)
print "gap: ", G
G = gap(X, y, l, k, g, betastar, 1e-0)
print "gap: ", G
G = gap(X, y, l, k, g, betastar, 1e-1)
print "gap: ", G
G = gap(X, y, l, k, g, betastar, 1e-2)
print "gap: ", G
G = gap(X, y, l, k, g, betastar, mu_zero)
print "gap: ", G


beta = betastar

alpha_tv = tv.alpha(beta, mu)

P = rr.f(X, y, k, beta) \
  + l1.f(l, beta) \
  + tv.phi(X, y, g, beta, alpha_tv, mu)

gAa = g * tv.Aa(alpha_tv)

beta_hat = beta

def D(beta, alpha_l1, alpha_tv):
    D = rr.f(X, y, k, beta) \
      + l1.phi(l, beta, alpha_l1, mu_zero) \
      + tv.phi(X, y, g, beta, alpha_tv, mu)

#    D = rr.f(X, y, k, beta=beta_hat) \
#      + l1.f(l, beta_hat) \
#      + tv.phi(X, y, g, beta_hat, alpha_tv, mu)

    return D

print "D:", D
print "grad:", np.linalg.norm(np.dot(X.T, np.dot(X, beta_hat) - y) + k * beta_hat + gAa)

mu = mu_zero

#t = 1.0 / rr.Lipschitz(X, k)
t = 1.0 / Lipschitz(X, k, g, mu)
print "t:", t
print "l:", l
print "k:", k
print "g:", g
for i in range(10000000):
    beta_hat = l1.prox(l * t, beta_hat - t * (rr.grad(X, y, k, beta_hat) + gAa))

#    D = np.sum((np.dot(X, beta_hat) - y) ** 2.0)
    D = rr.f(X, y, k, beta=beta_hat) \
      + l1.f(l, beta_hat) \
      + tv.phi(X, y, g, beta_hat, alpha_tv, mu)

    if i % 100000 == 0:
        print "D:", D
        print "grad:", np.linalg.norm(np.dot(X.T, np.dot(X, beta_hat) - y) + k * beta_hat + gAa)
#    print "beta: ", beta_hat.T

#    print rr.f(X, y, k, beta) - rr.f(X, y, k, beta_hat)

        print "P - D: ", P - D

#beta_min = np.dot(np.linalg.pinv(X), y)
#print "min:", np.sum((np.dot(X, beta_min) - y) ** 2.0)
print beta_hat
print "Gap: ", P - D

G = gap(X, y, l, k, g, betastar, mu_zero)
print "gap: ", G









#beta0 = np.random.rand(*betastar.shape)
#step = 1.0 / Lipschitz(X, k, g, mu)
#
#v = []
#x = []
#fval = []
#beta_opt = 0
#
#num_lin = 13
##ls = np.maximum(0.0, np.linspace(l - l * 0.1, l + l * 0.1, num_lin))
##ks = np.maximum(0.0, np.linspace(k - k * 0.1, k + k * 0.1, num_lin))
#gs = np.maximum(0.0, np.linspace(g - g * 0.1, g + g * 0.1, num_lin))
#beta = beta0
#from time import time
#t__ = time()
#for i in range(len(gs)):
#    val = gs[i]
##    beta = beta0
##    beta, crit = FISTA(X, y, l, k, val, beta, step, mu=mu*100000.0, eps=eps, maxit=10000)
##    print "f:", f(X, y, l, k, val, beta, mu=mu)
##    beta, crit = FISTA(X, y, l, k, val, beta, step, mu=mu*10000.0, eps=eps, maxit=10000)
##    print "f:", f(X, y, l, k, val, beta, mu=mu)
##    beta, crit = FISTA(X, y, l, k, val, beta, step, mu=mu*1000.0, eps=eps, maxit=1000)
##    print "f:", f(X, y, l, k, val, beta, mu=mu)
##    beta, crit = FISTA(X, y, l, k, val, beta, step, mu=mu*100.0, eps=eps, maxit=1000)
##    print "f:", f(X, y, l, k, val, beta, mu=mu)
##    beta, crit = FISTA(X, y, l, k, val, beta, step, mu=mu*10.0, eps=eps, maxit=1000)
##    print "f:", f(X, y, l, k, val, beta, mu=mu)
##    beta, crit = FISTA(X, y, l, k, val, beta, step, mu=mu, eps=eps, maxit=conts * maxit)
##    print "f:", f(X, y, l, k, val, beta, mu=mu)
##    beta0 = beta
#
#    (beta, fval_, Gval_) = CONESTA(X, y, l, k, val, beta,
#                                   mumin=mu_zero, sigma=1.0, tau=0.5,
#                                   eps=utils.TOLERANCE,
#                                   conts=conts, maxit=maxit, dynamic=False)
#
##    print "FISTA:  ", f(X, y, l, k, g, beta, mu=mu)
##    print "CONESTA:", f(X, y, l, k, g, beta_test, mu=mu)
#
#    curr_val = np.sum((beta - betastar) ** 2.0)
#    print "curr_val: ", curr_val
#    f_ = f(X, y, l, k, g, beta, mu=mu)
#
#    print "rr:", rr.f(X, y, k, beta)
#    print "l1:", l1.f(l, beta)
#    print "tv:", tv.f(X, y, g, beta, mu)
#
#    v.append(curr_val)
#    x.append(val)
#    fval.append(f_)
#
#    if curr_val <= min(v):
#        beta_opt = beta
#        fbest = fval_
#        Gbest = Gval_
##        beta_test_opt = beta_test
#
#    print "true = %.5f => %.7f" % (val, curr_val)
#
#print "time:", (time() - t__)
#
#print "best  f:", f(X, y, l, k, g, betastar, mu=mu)
#print "found f:", f(X, y, l, k, g, beta_opt, mu=mu)
#print "least f:", min(fval)
#
#plot.subplot(3, 2, 1)
#plot.plot(x, v, '-b')
#plot.title("true: %.5f, min: %.5f" % (g, x[np.argmin(v)]))
#
#plot.subplot(3, 2, 3)
#plot.plot(x, fval, '-b')
#plot.title("true: %.5f, min: %.5f" % (g, x[np.argmin(fval)]))
#
#plot.subplot(3, 2, 5)
#plot.plot(betastar, '-g', beta_opt, '-r')
##plot.plot(betastar, '-g', beta_opt, '-r', beta_test_opt, '--b')
#
#plot.subplot(3, 2, 2)
#print fbest
#fbest = sum(fbest, [])
#print fbest
#plot.plot(fbest, '-b')
#
#plot.subplot(3, 2, 4)
#print Gbest
##Gbest = sum(Gbest, [])
#plot.plot(Gbest, '-b')
#
#plot.show()










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