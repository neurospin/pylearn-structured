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
#mu_zero = 1e-8


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

        if mu > 0.0:
            alpha = self.alpha(beta, mu)
            Aa = self.Aa(alpha)

            alpha_sqsum = 0.0
            for a in alpha:
                alpha_sqsum += np.sum(a ** 2.0)

            return g * (np.dot(Aa.T, beta)[0, 0] - (mu / 2.0) * alpha_sqsum)

        else:
            A = self.A()
            sqsum = np.sum(np.sqrt(A[0].dot(beta) ** 2.0 + \
                                   A[1].dot(beta) ** 2.0 + \
                                   A[2].dot(beta) ** 2.0))

            return g * sqsum

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

        return float(g) * lmax / mu

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
        pass

    """ Function value of L1.
    """
    def f(self, l, beta):
        return l * np.sum(np.abs(beta))

    def phi(self, l, beta, alpha, mu):
#        return l * np.dot(alpha.T, beta)[0, 0]
        return l * (np.dot(alpha[0].T, beta)[0, 0] \
                - 0.5 * mu * np.sum(alpha[0] ** 2.0))

    """ Proximal operator of the L1 norm
    """
    def prox(self, l, x):
        return (np.abs(x) > l) * (x - l * np.sign(x - l))

    ### Methods for the dual formulation ###

#    def fmu(self, beta, alpha, l, mu):
#        return l * (np.dot(alpha.T, beta)[0, 0] \
#                - 0.5 * mu * np.sum(alpha ** 2.0))

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


# Dual function value of Ridge regression and smoothed L1
def phi(X, y, k, g, beta, alpha, mu):
    if alpha == None:
        alpha = tv.alpha(beta, mu)
    return rr.phi(X, y, k, beta) + tv.phi(X, y, g, beta, alpha, mu)


# Gradient of Ridge regression + TV
def grad(X, y, k, g, beta, mu):
    return rr.grad(X, y, k, beta) + tv.grad(g, beta, mu)


# Proximal operator of the L1 norm
def prox(l, x):
    return l1.prox(l, x)


def betahat(X, y, k, g, beta, alpha=None, Aalpha=None):

    XXkI = np.dot(X.T, X) + k * np.eye(X.shape[1])
    if alpha != None:
        betahatk = np.dot(np.linalg.pinv(XXkI), np.dot(X.T, y) - g * tv.Aa(alpha))
    else:
        betahatk = np.dot(np.linalg.pinv(XXkI), np.dot(X.T, y) - g * Aalpha)

    return betahatk


def gap_function(X, y, k, g, beta, mu):

#    alphak = tv.alpha(beta, mu)
#
##    gradbetak = rr.grad(X, y, k, beta)
##    i = np.abs(beta) < utils.TOLERANCE
##    alphak[i] = (-1.0 / l) * gradbetak[i]
#
#    betahatk = betahat(X, y, k, g, beta, alphak)

    alphak = tv.alpha(beta, mu)
    Aak = tv.Aa(alphak)

    Aa = -(1.0 / g) * rr.grad(X, y, k, beta)
    i = np.abs(beta) < utils.TOLERANCE
    Aak[i] = Aa[i]

#    betahatk = betahat(X, y, k, g, beta, alphak)
    betahatk = betahat(X, y, k, g, beta, Aalpha=Aak)

    return phi(X, y, k, g, beta, alphak, mu) \
         - phi(X, y, k, g, betahatk, alphak, mu)
#    return phi(X, y, l, k, g, beta, alphak) \
#         - phi(X, y, l, k, g, betahatk, alphak)


def Lipschitz(X, k, g, mu):
    if mu > 0.0:
        return rr.Lipschitz(X, k) + tv.Lipschitz(g, mu)
    else:
        return rr.Lipschitz(X, k)


# L1
#def mu_plus(l, p, lmax, epsilon):
#    return (-p * l ** 2.0 + np.sqrt((p * l ** 2.0) ** 2.0 \
#            + 2.0 * p * epsilon * lmax * l ** 2.0)) / (p * lmax * l)

# TV
def mu_plus(eps, g, D, lmaxX, lmaxA):
    return (-2.0 * (g ** 2.0) * lmaxA * D \
            + np.sqrt((2.0 * (g ** 2.0) * lmaxA * D) ** 2.0 \
                       + 4.0 * (g ** 2.0) * lmaxX * D * eps * lmaxA)) \
           / (2.0 * g * lmaxX * D)


#def sinf(u):
#    unorm = np.abs(u)
#    i = unorm > 1.0
#    unorm_i = unorm[i]
#    u[i] = np.divide(u[i], unorm_i)
#    return u


# The fast iterative shrinkage threshold algorithm
def FISTA(X, y, l, k, g, beta, step, mu,
          eps=utils.TOLERANCE, maxit=utils.MAX_ITER):

    z = betanew = betaold = beta

    crit = [f(X, y, l, k, g, beta, mu=mu_zero)]
    critmu = [f(X, y, l, k, g, beta, mu=mu)]
    for i in xrange(1, maxit + 1):
        z = betanew + ((i - 2.0) / (i + 1.0)) * (betanew - betaold)
        betaold = betanew
        betanew = prox(step * l, z - step * grad(X, y, k, g, z, mu))
        crit.append(f(X, y, l, k, g, betanew, mu=mu_zero))
        critmu.append(f(X, y, l, k, g, beta, mu=mu))
        if math.norm1(betanew - z) < eps * step:
            break

    return (betanew, crit, critmu)


def conesmo(X, y, l, k, beta, eps=utils.TOLERANCE, conts=10, maxit=100):

    lambdamax = Lipschitz(X, l, k)
    mu = np.max(np.abs(math.corr(X, y)))
    print "start mu:", mu, ", eps:", eps, ", conts:", conts, "maxit:", maxit

    gap = gap_function(X, y, k, beta)
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
        gap = gap_function(X, y, k, beta)
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


np.random.seed(42)

l = 0.0  # 0.61803
k = 1.0  # 0.271828
g = 100.0  # 3.14159

px = 6
py = 1
pz = 1
p = px * py * pz
n = 6

rr = RidgeRegression()
l1 = L1()
tv = TotalVariation(shape=(pz, py, px))

a = 1.0
Sigma = a * np.eye(p) + (1.0 - a) * np.ones((p, p))
Mu = np.zeros(p)
M = np.random.multivariate_normal(Mu, Sigma, n)
e = np.random.randn(n, 1)

mu_zero = 1e-8
eps = 1e-6
mu = (0.9 * eps / (g * p)) * 1.0
print "mu:", mu
conts = 10000
maxit = 100

#    X, y, beta = lasso.load(l, density=0.7, snr=100.0, M=M, e=e)
#    X, y, beta = ridge.load(k, density=0.7, snr=100.0, M=M, e=e)
#    X, y, beta = l1_tv.load(l, gamma, density=0.7, snr=100.0, M=M, e=e)
#    X, y, beta = ridge_2D.load(k, density=0.7, snr=100.0, M=M, e=e,
#                               shape=(py, px))
#    X, y, beta = lasso_2D.load(l, density=0.7, snr=100.0, M=M, e=e,
#                               shape=(py, px))
#    X, y, beta = l1_l2.load(l, k, density=0.7, snr=100.0, M=M, e=e)
#    X, y, beta = l1_l2_2D.load(l, k, density=0.7, snr=100.0, M=M, e=e,
#                               shape=(py, px))
#X, y, betastar = l1_l2_tv.load(l, k, g, density=0.50, snr=100.0,
#                               M=M, e=e)
X, y, betastar = l1_l2_tvmu.load(l, k, g, density=0.50, snr=100.0, M=M, e=e,
                                 tv=tv, mu=mu)

print betastar
for A in tv.A():
    print A.todense()
print tv.alpha(betastar, mu)
print tv.Aa(tv.alpha(betastar, mu))

beta0 = np.random.rand(*betastar.shape)
step = 1.0 / Lipschitz(X, k, g, mu)

v = []
x = []
fval = []
beta_opt = 0

num_lin = 25
#ls = np.maximum(0.0, np.linspace(l - l * 0.1, l + l * 0.1, num_lin))
#ks = np.maximum(0.0, np.linspace(k - k * 0.1, k + k * 0.1, num_lin))
gs = np.maximum(0.0, np.linspace(g - g * 0.1, g + g * 0.1, num_lin))
for i in range(len(gs)):
    val = gs[i]
    beta = beta0
    beta, crit, critmu = FISTA(X, y, l, k, val, beta, step, mu=mu*100000.0, eps=eps, maxit=1000)
    print "f:", f(X, y, l, k, val, beta, mu=mu)
    beta, crit, critmu = FISTA(X, y, l, k, val, beta, step, mu=mu*10000.0, eps=eps, maxit=1000)
    print "f:", f(X, y, l, k, val, beta, mu=mu)
    beta, crit, critmu = FISTA(X, y, l, k, val, beta, step, mu=mu*1000.0, eps=eps, maxit=1000)
    print "f:", f(X, y, l, k, val, beta, mu=mu)
    beta, crit, critmu = FISTA(X, y, l, k, val, beta, step, mu=mu*100.0, eps=eps, maxit=1000)
    print "f:", f(X, y, l, k, val, beta, mu=mu)
    beta, crit, critmu = FISTA(X, y, l, k, val, beta, step, mu=mu*10.0, eps=eps, maxit=1000)
    print "f:", f(X, y, l, k, val, beta, mu=mu)
    beta, crit, critmu = FISTA(X, y, l, k, val, beta, step, mu=mu, eps=eps, maxit=conts * maxit)
    print "f:", f(X, y, l, k, val, beta, mu=mu)
#    beta0 = beta

#    print "f(betastar) = ", f(X, y, l, k, g, betastar, mu=mu_zero)
#    print "f(beta) = ", f(X, y, l, k, g, beta, mu=mu_zero)
#    print "f(betastar, mu) = ", f(X, y, l, k, g, betastar, mu=mu)
#    print "f(beta, mu) = ", f(X, y, l, k, g, beta, mu=mu)
#    fstar = f(X, y, l, k, g, betastar, mu=mu_zero)
#    print "err:", f(X, y, l, k, g, beta, mu=mu_zero) - fstar

    curr_val = np.sum((beta - betastar) ** 2.0)
    f_ = f(X, y, l, k, val, beta, mu=mu)

    print "rr:", rr.f(X, y, k, beta)
    print "l1:", l1.f(l, beta)
    print "tv:", tv.f(X, y, g, beta, mu)

    v.append(curr_val)
    x.append(val)
    fval.append(f_)

    if curr_val <= min(v):
        beta_opt = beta

    print "true = %.5f => %.7f" % (val, curr_val)

print "best  f:", f(X, y, l, k, g, betastar, mu=mu)
print "found f:", f(X, y, l, k, g, beta_opt, mu=mu)
print "least f:", min(fval)

plot.subplot(2, 1, 1)
plot.plot(x, v, '-b')
plot.title("true: %.5f, min: %.5f" % (g, x[np.argmin(v)]))
plot.subplot(2, 1, 2)
plot.plot(betastar, '-g', beta_opt, '-r')
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