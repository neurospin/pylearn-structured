# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 12:07:50 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: TBD
"""

import numpy as np
import scipy.sparse as sparse
#import strukturerad.algorithms as algorithms
import strukturerad.utils as utils
from strukturerad.utils import math

import strukturerad.datasets.simulated.l1_l2_tv as l1_l2_tv
import strukturerad.datasets.simulated.l1_l2_tvmu as l1_l2_tvmu
import strukturerad.datasets.simulated.l1mu_l2_tvmu as l1mu_l2_tvmu

import strukturerad.datasets.simulated.beta as generate_beta

from time import time
import matplotlib.pyplot as plot
import matplotlib.cm as cm


np.random.seed(42)

mu_zero = 5e-8
TOLERANCE = 5e-8
MAX_ITER = 1000


def FastSVD(X, max_iter=100, start_vector=None):
    """A kernel SVD implementation.

    Performs SVD of given matrix. This is always faster than np.linalg.svd.
    Particularly, this is a lot faster than np.linalg.svd when M << N or
    M >> N, for an M-by-N matrix.

    Arguments:
    ---------
    X : The matrix to decompose

    Returns:
    -------
    v : The right singular vector.
    """
    M, N = X.shape
    if M < 80 and N < 80:  # Very arbitrary threshold for my computer ;-)
        _, _, V = np.linalg.svd(X, full_matrices=True)
        v = V[[0], :].T
    elif M < N:
        Xt = X.T
        K = np.dot(X, Xt)
        # TODO: Use module for this!
        t = np.random.rand(X.shape[0], 1)
#        t = start_vectors.RandomStartVector().get_vector(Xt)
        for it in xrange(max_iter):
            t_ = t
            t = np.dot(K, t_)
            t /= np.sqrt(np.sum(t_ ** 2.0))

            if np.sqrt(np.sum((t_ - t) ** 2.0)) < TOLERANCE:
                break

        v = np.dot(Xt, t)
        v /= np.sqrt(np.sum(v ** 2.0))

    else:
        K = np.dot(X.T, X)
        # TODO: Use module for this!
        v = np.random.rand(X.shape[1], 1)
#        v = start_vectors.RandomStartVector().get_vector(X)
        for it in xrange(max_iter):
            v_ = v
            v = np.dot(K, v_)
            v /= np.sqrt(np.sum(v ** 2.0))

            if np.sqrt(np.sum((v_ - v) ** 2.0)) < TOLERANCE:
                break

    return v


def FastSparseSVD(X, max_iter=100, start_vector=None):
    """A kernel SVD implementation for sparse CSR matrices.

    This is usually faster than np.linalg.svd when density < 20% and when
    M << N or N << M (at least one order of magnitude). When M = N >= 10000 it
    is faster when the density < 1% and always faster regardless of density
    when M = N < 10000.

    These are ballpark estimates that may differ on your computer.

    Arguments:
    ---------
    X : The matrix to decompose

    Returns:
    -------
    v : The right singular vector.
    """
    M, N = X.shape
    # TODO: Use module for this!
    p = np.random.rand(X.shape[1], 1)
#    p = start_vectors.RandomStartVector().get_vector(X)
    if M < N:
        Xt = X.T
        K = X.dot(Xt)
        t = X.dot(p)
        for it in xrange(max_iter):
            t_ = t
            t = K.dot(t_)
            t /= np.sqrt(np.sum(t_ ** 2.0))

            if (np.sum((t_ - t) ** 2.0)) < TOLERANCE:
                break

        p = Xt.dot(t)
        normp = np.sqrt(np.sum(p ** 2.0))
        # Is the solution significantly different from zero (or TOLERANCE)?
        if normp >= TOLERANCE:
            p /= normp
        else:
            p = np.zeros(p.shape) / np.sqrt(p.shape[0])

    else:
        K = X.T.dot(X)
        for it in xrange(max_iter):
            p_ = p
            p = K.dot(p_)
            normp = np.sqrt(np.sum(p ** 2.0))
            if normp > TOLERANCE:
                p /= normp
            else:
                p = np.zeros(p.shape) / np.sqrt(p.shape[0])

            if (np.sum((p_ - p) ** 2.0)) < TOLERANCE:
                break

    return p


class RidgeRegression(object):

    def __init__(self, k):

        self.k = float(k)

        self.reset()

    def reset(self):

        self._lambda_max = None
        self._lambda_min = None

    """ Function value of Ridge regression.
    """
    def f(self, X, y, beta):

        return (1.0 / 2.0) * np.sum((np.dot(X, beta) - y) ** 2.0) \
             + (self.k / 2.0) * np.sum(beta ** 2.0)

    """ Gradient of Ridge regression
    """
    def grad(self, X, y, beta):

        return np.dot((np.dot(X, beta) - y).T, X).T + self.k * beta

    def Lipschitz(self, X):

        return self.lambda_max(X)

    def lambda_max(self, X):

        if self._lambda_max == None:
            s = np.linalg.svd(X, full_matrices=False, compute_uv=False)

            self._lambda_max = np.max(s) ** 2.0

            if len(s) < X.shape[1]:
                self._lambda_min = 0.0
            else:
                self._lambda_min = np.min(s) ** 2.0

        return self._lambda_max + self.k

    def lambda_min(self, X):

        if self._lambda_min == None:
            s = np.linalg.svd(X, full_matrices=False, compute_uv=False)

            self._lambda_max = np.max(s) ** 2.0

            if len(s) < X.shape[1]:
                self._lambda_min = 0.0
            else:
                self._lambda_min = np.min(s) ** 2.0

        return self._lambda_min + self.k

#    ### Methods for the dual formulation ###
#
#    def phi(self, X, y, beta):
#
#        return self.f(X, y, self.k, beta)


class L1(object):
    """The proximal operator of the L1 loss function

        f(x) = l * ||x||_1,

    where ||x||_1 is the L1 loss function.
    """
    def __init__(self, l):

        self.l = float(l)

    """ Function value of L1.
    """
    def f(self, beta, mu=None):

        return self.l * np.sum(np.abs(beta))

    """ Proximal operator of the L1 norm
    """
    def prox(self, x, factor=1.0):

        l = self.l * factor

        return (np.abs(x) > l) * (x - l * np.sign(x - l))


class SmoothedL1(object):
    """The proximal operator of the smoothed L1 loss function

        f(x) = l * L1mu(x),

    where L1mu(x) is the smoothed L1 loss function.
    """
    def __init__(self, l, p):

        self.l = float(l)
        self._p = p
        self._A = None

    """ Function value of L1.
    """
    def f(self, beta, mu=0.0):

        if self.l < TOLERANCE:
            return 0.0

        if mu == 0.0:
            return self.l * np.sum(np.abs(beta))
        else:
            alpha = self.alpha(beta, mu)
            return self.phi(beta, alpha, mu)

    def phi(self, beta, alpha, mu):

        if self.l < TOLERANCE:
            return 0.0

        return self.l * (np.dot(alpha[0].T, beta)[0, 0] \
                            - (mu / 2.0) * np.sum(alpha[0] ** 2.0))

    def grad(self, beta, mu):

        alpha = self.alpha(beta, mu)

        return self.l * alpha[0]

    def Lipschitz(self, mu):

        return self.l / mu

    def A(self):

        if self._A == None:
            self._A = sparse.eye(self._p, self._p)

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

    def M(self):

        A = self.A()
        return A[0].shape[0] / 2.0

    def mu(self, beta):

        return np.max(np.absolute(beta))


class TotalVariation(object):

    def __init__(self, g, shape):

        self.g = float(g)
        self._A = self.precompute(shape, mask=None, compress=False)
        self._lambda_max = None

    """ Function value of Ridge regression.
    """
    def f(self, beta, mu=0.0):

        if self.g < TOLERANCE:
            return 0.0

        if mu > TOLERANCE:
            alpha = self.alpha(beta, mu)
            return self.phi(beta, alpha, mu)
        else:
            A = self.A()
            return self.g * np.sum(np.sqrt(A[0].dot(beta) ** 2.0 + \
                                           A[1].dot(beta) ** 2.0 + \
                                           A[2].dot(beta) ** 2.0))

    def phi(self, beta, alpha, mu):

        if self.g < TOLERANCE:
            return 0.0

        Aa = self.Aa(alpha)

        alpha_sqsum = 0.0
        for a in alpha:
            alpha_sqsum += np.sum(a ** 2.0)

        return self.g * (np.dot(Aa.T, beta)[0, 0] \
                            - (mu / 2.0) * alpha_sqsum)

    """ Gradient of Total variation.
    """
    def grad(self, beta, mu, alpha=None):

        if self.g < TOLERANCE:
            return 0.0

        if alpha == None:
            alpha = self.alpha(beta, mu)
        grad = self.Aa(alpha)

        return self.g * grad

    def Lipschitz(self, mu, max_iter=10):

        if self.g < TOLERANCE:
            return 0.0

        lmaxA = self.lambda_max(mu, max_iter=max_iter)

        return self.g * lmaxA / mu

    def lambda_max(self, mu, max_iter=10):

        # Note that we can save the state here since lmax(A) does not change.
        if self._lambda_max == None:
            A = sparse.vstack(self.A())
            v = FastSparseSVD(A, max_iter=max_iter)
            us = A.dot(v)
            self._lambda_max = np.sum(us ** 2.0)

        return self._lambda_max

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

    def M(self):

        return self._A[0].shape[0] / 2.0

    def mu(self, beta):

        SS = 0
        A = self.A()
        for i in xrange(len(A)):
            SS += A[i].dot(beta) ** 2.0

        return np.max(np.sqrt(SS))

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


class OLSL2_L1_TV(object):

    def __init__(self, k, l, g, shape):

        self.rr = RidgeRegression(k)
        self.l1 = L1(l)
        self.tv = TotalVariation(g, shape=shape)

    """ Function value of Ridge regression, L1 and TV.
    """
    def f(self, X, y, beta, mu):

        return self.rr.f(X, y, beta) \
             + self.l1.f(beta) \
             + self.tv.f(beta, mu)

    """ Gradient of the differentiable part with Ridge regression + TV.
    """
    def grad(self, X, y, beta, mu):

        return self.rr.grad(X, y, beta) \
             + self.tv.grad(beta, mu)

    def Lipschitz(self, X, mu, max_iter=100):

        return self.rr.Lipschitz(X) \
             + self.tv.Lipschitz(mu, max_iter=max_iter)

    """ Proximal operator of the L1 norm.
    """
    def prox(self, beta, factor=1.0):

        return self.l1.prox(beta, factor)

    def mu(self, beta):

        return self.tv.mu(beta)

    def mu_opt(self, eps, X):

        gM = self.tv.g * self.tv.M()
        gA2 = self.tv.Lipschitz(1.0)  # Gamma is in here!
        Lg = self.rr.Lipschitz(X)

        return (-gM * gA2 + np.sqrt((gM * gA2) ** 2.0 \
                    + gM * Lg * gA2 * eps)) / (gM * Lg)

    def eps_opt(self, mu, X):

        gM = self.tv.g * self.tv.M()
        gA2 = self.tv.Lipschitz(1.0)  # Gamma is in here!
        Lg = self.rr.Lipschitz(X)

        return (2.0 * gM * gA2 * mu + gM * Lg * mu ** 2.0) / gA2

#    """ Returns the beta that minimises the dual function.
#    """
#    def betahat(self, X, y, alpha):
#
#        l = self.l1.l
#        k = self.rr.k
#        g = self.tv.g
#        grad_l1 = l * self.l1.Aa(alpha[0])
#        grad_tv = g * self.tv.Aa(alpha[1:])
#
#        XXkI = np.dot(X.T, X) + k * np.eye(X.shape[1])
#        beta = np.dot(np.linalg.inv(XXkI), np.dot(X.T, y) - grad_l1 - grad_tv)
#
#        return beta

    def gap(self, X, y, beta, mu, eps=utils.TOLERANCE, max_iter=100):

        alpha = self.tv.alpha(beta, mu)

#        if smooth_l1:
#            alpha_l1 = self.l1.alpha(beta, mu)
#
#            P = self.rr.f(X, y, beta) \
#              + self.l1.phi(beta, alpha_l1, mu) \
#              + self.tv.phi(beta, alpha_tv, mu)
#
##            Aa_l1 = self.l1.Aa(alpha_l1)
##            Aa_tv = self.tv.Aa(alpha_tv)
#
##            lAa_l1 = self.l1.l * Aa_l1
##            gAa_tv = self.tv.g * Aa_tv
##            gAa = lAa_l1 + gAa_tv
#            beta_hat = self.betahat(X, y, [alpha_l1, alpha_tv])
#
#            D = self.rr.f(X, y, beta_hat) \
#              + self.l1.phi(beta_hat, alpha_l1, mu) \
#              + self.tv.phi(beta_hat, alpha_tv, mu)
#
#        else:
        P = self.rr.f(X, y, beta) \
          + self.l1.f(beta) \
          + self.tv.phi(beta, alpha, mu)

        t = 1.0 / self.Lipschitz(X, mu)
        beta_old = beta_new = beta

        # TODO: Use the FISTA function instead!!
        for i in xrange(1, max_iter):
            z = beta_new + ((i - 2.0) / (i + 1.0)) * (beta_new - beta_old)
            beta_old = beta_new

            beta_new = self.prox(z - t * (self.rr.grad(X, y, z) \
                                        + self.tv.grad(z, mu, alpha)), t)

            D = self.rr.f(X, y, beta_new) \
              + self.l1.f(beta_new) \
              + self.tv.phi(beta_new, alpha, mu)

            if (1.0 / t) * math.norm(beta_new - z) < eps and P - D >= 0:
                print "Broke after %d iterations" % (i,)
                break

        return P - D


class SmoothedL1TV(object):

    def __init__(self, l, g, shape):

        self.l1 = SmoothedL1(l, np.prod(shape))
        self.tv = TotalVariation(g, shape)

        self._lambda_max = None

    def f(self, beta, mu=0.0):

        return self.l1.f(beta, mu) \
             + self.tv.f(beta, mu)

    def phi(self, beta, alpha, mu):

        return self.l1.phi(beta, alpha[0], mu) \
             + self.tv.phi(beta, alpha[1:], mu)

    def grad(self, beta, mu):

        return self.l1.grad(beta, mu) \
             + self.tv.grad(beta, mu)

#    def grad(self, alpha):
#
#        return self.l1.l * self.l1.Aa(alpha[0]) \
#             + self.tv.g * self.tv.Aa(alpha[1:])

    def Lipschitz(self, mu, max_iter=100):

        return self.l1.Lipschitz(mu) \
             + self.tv.g * self.tv.Lipschitz(mu, max_iter=max_iter)

#    def lambda_max(self, mu, max_iter=10):
#
#        # Note that we can save the state here since lmax(A) does not change.
#        if self._lambda_max == None:
#            A = sparse.vstack(self.A())
#            v = FastSparseSVD(X, max_iter=max_iter)
#            us = A.dot(v)
#            self._lambda_max = np.sum(us ** 2.0)
#
#        return self._lambda_max

    def A(self):

#        A = self.l1.A() + self.tv.A()
#
#        A[0] *= self.l1.l
#        A[1] *= self.l1.g
#        A[2] *= self.l1.g
#        A[3] *= self.l1.g

        return self.l1.A() + self.tv.A()

#    def gAa(self, alpha):
#
#        return self.l1.l * self.l1.Aa(alpha[0]) \
#             + self.tv.g * self.tv.Aa(alpha[1:])

    def alpha(self, beta, mu):

        A = self.A()

        a = [0] * len(A)

        a[0] = (self.l1.l / mu) * A[0].dot(beta)
        a[1] = (self.tv.g / mu) * A[1].dot(beta)
        a[2] = (self.tv.g / mu) * A[2].dot(beta)
        a[3] = (self.tv.g / mu) * A[3].dot(beta)

        return self.project(a)

    def project(self, a):

        return self.l1.project(a[:1]) \
             + self.tv.project(a[1:])

    def M(self):

        return self.l1.M() \
             + self.tv.M()

#    def mu(self, beta):
#
#        return max(self.l1.mu(beta), self.tv.mu(beta))

    def V(self, u, beta, L):

#        a = self.alpha(beta, 1.0, project=False)

        A = self.A()
        a = [0] * len(A)
        a[0] = (self.l1.l / L) * A[0].dot(beta)
        a[1] = (self.tv.g / L) * A[1].dot(beta)
        a[2] = (self.tv.g / L) * A[2].dot(beta)
        a[3] = (self.tv.g / L) * A[3].dot(beta)

        u_new = [0] * len(u)
        for i in xrange(len(u)):
#            u_new[i] = u[i] + g[i] * A[i].dot(beta) / L
#            u_new[i] = u[i] + a[i] / L
            u_new[i] = u[i] + a[i]

        return self.project(u_new)


class OLSL2_SmoothedL1TV(object):

    def __init__(self, k, l, g, shape):

        self.g = RidgeRegression(k)
        self.h = SmoothedL1TV(l, g, shape)

    """ Function value of Ridge regression and TV.
    """
    def f(self, X, y, beta, mu):

        return self.g.f(X, y, beta) \
             + self.h.f(beta, mu)

#    """ Gradient of the differentiable part with Ridge regression + TV.
#    """
#    def grad(self, X, y, beta, mu):
#
#        return self.g.grad(X, y, beta) \
#             + self.h.grad(beta, mu)

    def Lipschitz(self, X, mu, max_iter=100):

        return self.h.Lipschitz(mu, max_iter=max_iter) / self.g.lambda_min(X)

#    def mu(self, beta):
#
#        return self.h.mu(beta)

    def V(self, u, beta, L):

        return self.h.V(u, beta, L)

    """ Returns the beta that minimises the dual function.
    """
    def betahat(self, X, y, alpha):

        A = self.h.A()
        grad = self.h.l1.l * A[0].T.dot(alpha[0])
        grad += self.h.tv.g * A[1].T.dot(alpha[1])
        grad += self.h.tv.g * A[2].T.dot(alpha[2])
        grad += self.h.tv.g * A[3].T.dot(alpha[3])

#        XXkI = np.dot(X.T, X) + self.g.k * np.eye(X.shape[1])

        Xty_grad = (np.dot(X.T, y) - grad) / self.g.k

#        t = time()
#        XXkI = np.dot(X.T, X)
#        index = np.arange(min(XXkI.shape))
#        XXkI[index, index] += self.g.k
#        invXXkI = np.linalg.inv(XXkI)
#        print "t:", time() - t
#        beta = np.dot(invXXkI, Xty_grad)

#        t = time()
        XXtkI = np.dot(X, X.T)
        index = np.arange(min(XXtkI.shape))
        XXtkI[index, index] += self.g.k
        invXXtkI = np.linalg.inv(XXtkI)
        beta = (Xty_grad - np.dot(X.T, np.dot(invXXtkI, np.dot(X, Xty_grad))))
#        print "t:", time() - t

        return beta


# The fast iterative shrinkage threshold algorithm
def FISTA(X, y, function, beta, step, mu,
          eps=utils.TOLERANCE, max_iter=utils.MAX_ITER):

    z = betanew = betaold = beta

    crit = []
    for i in xrange(1, max_iter + 1):
        z = betanew + ((i - 2.0) / (i + 1.0)) * (betanew - betaold)
        betaold = betanew
        betanew = function.prox(z - step * function.grad(X, y, z, mu), step)

        crit.append(function.f(X, y, betanew, mu=mu))

        if math.norm(betanew - z) < eps * step:
            break

    return (betanew, crit)


def CONESTA(X, y, function, beta, mumin=utils.TOLERANCE, sigma=1.0, tau=0.5,
            dynamic=True, eps=utils.TOLERANCE, conts=50, max_iter=1000):

    mu = [0.9 * function.mu(beta)]
    print "mu:", mu[0]
    eps0 = function.eps_opt(mu[0], X)
    print "eps:", eps0

    t = []
    tmin = 1.0 / function.Lipschitz(X, mumin, max_iter=100)

    f = []
    Gval = []

    i = 0
    while True:
        t.append(1.0 / function.Lipschitz(X, mu[i], max_iter=100))
        eps_plus = max(eps, function.eps_opt(mu[i], X))
        print "eps_plus: ", eps_plus
        (beta, crit) = FISTA(X, y, function, beta, t[i], mu[i],
                             eps=eps_plus, max_iter=max_iter)
        print "crit: ", crit[-1]
        print "it: ", len(crit)
        f.append(crit)

        mumin = min(mumin, mu[i])
        tmin = min(tmin, t[i])
        beta_tilde = function.prox(beta - tmin * function.grad(X, y,
                                                               beta, mumin),
                                   tmin)

        if (1.0 / tmin) * math.norm(beta - beta_tilde) < eps or i >= conts:
            break

        if dynamic:
            G = function.gap(X, y, beta, mu[i],
                             eps=eps_plus, max_iter=max_iter)
        else:
            G = eps0 * tau ** (i + 1)

        G = abs(G) / sigma

        if G <= utils.TOLERANCE and mu[i] <= utils.TOLERANCE:
            break

        Gval.append(G)
        mu_new = min(mu[i], function.mu_opt(G, X))
        mu.append(max(mumin, mu_new))
#        mu.append(mu_new)

        print "mu:", mu[i + 1]

        i = i + 1

    return (beta, f, Gval)


def ExcessiveGapMethod(X, y, function, eps=TOLERANCE, max_iter=MAX_ITER):
    """ The excessive gap method for strongly convex functions.

    Parameters
    ----------
    function : The function to minimise. It contains two parts, function.g is
            the strongly convex part and function.h is the smoothed part of the
            function.
    """

    A = function.h.A()

    u = [0] * len(A)
    for i in xrange(len(A)):
        u[i] = np.zeros((A[i].shape[0], 1))

    L = function.Lipschitz(X, 1.0, max_iter=100)

    mu = [L]
    beta0 = function.betahat(X, y, u)  # u is zero here
    beta = beta0
    alpha = function.V(u, beta, L)  # u is zero here

    k = 0

    f = []  # function.f(X, y, beta[0], mu[0])]
    # mu[0] * function.h.D()
    ulim = [4.0 * L * function.h.M() / ((k + 1.0) * (k + 2.0))]

    while True:
        tau = 2.0 / (float(k) + 3.0)

        alpha_hat = function.h.alpha(beta, mu[k])
        for i in xrange(len(alpha_hat)):
            u[i] = (1.0 - tau) * alpha[i] + tau * alpha_hat[i]

        mu.append((1.0 - tau) * mu[k])
        betahat = function.betahat(X, y, u)
        beta = ((1.0 - tau) * beta + tau * betahat)
        alpha = function.V(u, betahat, L)

        if mu[k] * function.h.M() < eps or k >= max_iter:
            break

        f.append(function.f(X, y, beta, TOLERANCE))
#        ulim.append(mu[k + 1] * function.h.D())
        ulim.append(4.0 * L * function.h.M() / ((k + 1.0) * (k + 2.0)))

        k = k + 1

    return (beta, f, ulim, beta0)


np.random.seed(42)

k = 0.5  # 0.271828
l = 0.5  # 0.61803
g = 10.0  # 3.14159

px = 6
py = 6
pz = 6
shape = (pz, py, px)
p = np.prod(shape)
n = 25

function_egm = OLSL2_SmoothedL1TV(k, l, g, shape)
function_pgm = OLSL2_L1_TV(k, l, g, shape)

a = 1.0
Sigma = a * np.eye(p) + (1.0 - a) * np.ones((p, p))
Mu = np.zeros(p)
M = np.random.multivariate_normal(Mu, Sigma, n)
e = np.random.randn(n, 1)
e = e / math.norm(e)
density = 0.5

eps = 1e-3
mu = 0.9 * (2.0 * eps / p)
print "mu:", mu
conts = 20
maxit = 2500

snr = 100.0

betastar = generate_beta.rand(shape, density=density, sort=True)
#betastar = np.random.rand(*shape).reshape((p, 1))
#print betastar

#betastart = np.random.rand(*betastar.shape)

X, y, betastar = l1_l2_tv.load(l, k, g, betastar, M, e, snr, shape)

t = time()
beta_egm, f_egm, ulim, betastart = ExcessiveGapMethod(X, y, function_egm,
                                                  eps=eps,
                                                  max_iter=conts * maxit)
print "time:", time() - t

#step = 1.0 / (function_test.g.Lipschitz(X) \
#            + function_test.h.Lipschitz(mu, max_iter=100))
#t = time()
#step = 1.0 / function_pgm.Lipschitz(X, mu, max_iter=100)
#beta_fista, f_fista = FISTA(X, y, function_pgm,
#                            betastart, step, mu=mu,
#                            eps=eps, max_iter=conts * maxit * 10)
#print "time:", time() - t
beta_fista = betastart
f_fista = [0, 0]

t = time()
(beta_conesta, f_conesta, G_conesta) = CONESTA(X, y, function_pgm, betastart,
                                               mumin=mu_zero, sigma=1.0,
                                               tau=0.5, dynamic=True,
                                               eps=eps,
                                               conts=conts, max_iter=maxit)
print "time:", time() - t

best_f = function_egm.f(X, y, betastar, mu=0.0)
found_f = function_egm.f(X, y, beta_egm, mu=0.0)
fista_f = function_egm.f(X, y, beta_fista, mu=0.0)
conesta_f = function_egm.f(X, y, beta_conesta, mu=0.0)
print "egm err      : %.6f" % (found_f - best_f)
print "egm best    f:", best_f
print "egm egm     f:", found_f
print "egm conesta f:", conesta_f
print "egm fista   f:", fista_f

best_f = function_pgm.f(X, y, betastar, mu=0.0)
found_f = function_pgm.f(X, y, beta_egm, mu=0.0)
fista_f = function_pgm.f(X, y, beta_fista, mu=0.0)
conesta_f = function_pgm.f(X, y, beta_conesta, mu=0.0)
print "pgm err      : %.6f" % (found_f - best_f)
print "pgm best    f:", best_f
print "pgm egm     f:", found_f
print "pgm conesta f:", conesta_f
print "pgm fista   f:", fista_f

plot.subplot(3, 1, 1)
plot.plot(betastar + 0.01, '-g', beta_fista, '.-k',
                                 beta_conesta, '-b',
                                 beta_egm, '-r')
plot.title("True $\\beta^*$ (green) and EGM $\\beta^{(k)}$ (red)")

plot.subplot(3, 1, 2)
plot.plot(f_egm,  '-b')
plot.plot([0, len(f_egm) - 1], [best_f, best_f], 'g')
plot.title("Function value of EGM as a function of iteration number")

plot.subplot(3, 1, 3)
err = [f_egm[i] - best_f for i in range(len(f_egm))]
plot.plot(err, '-r')
plot.plot(ulim, '-g')
plot.title("Error at $k$th iteration and theoretical upper bound on the error")

plot.show()

#print "snr = %.5f = %.5f = |X.b| / |e| = %.5f / %.5f" \
#       % (snr, np.linalg.norm(np.dot(X, betastar)) / np.linalg.norm(e),
#          np.linalg.norm(np.dot(X, betastar)), np.linalg.norm(e))
#
#
#beta0 = np.random.rand(*betastar.shape)
#
#v = []
#x = []
#fval = []
#beta_opt = 0
#
#num_lin = 7
#val = g
#vals = np.maximum(0.0, np.linspace(val - val * 0.1, val + val * 0.1, num_lin))
#beta = beta0
#step = 1.0 / rrl1tv.Lipschitz(X, mu)
#beta, _ = FISTA(X, y, rrl1tv, beta, step, mu=100.0 * mu, eps=eps, maxit=10000)
#
#t__ = time()
#for i in range(len(vals)):
#    val = vals[i]
#    l_ = l
#    k_ = k
#    g_ = val
#
##    beta = beta0    
##    beta, crit = FISTA(X, y, l_, k_, g_, beta, step, mu=mu*100000.0, eps=eps, maxit=10000)
##    print "f:", f(X, y, l_, k_, g_, beta, mu=mu)
##    beta, crit = FISTA(X, y, l_, k_, g_, beta, step, mu=mu*10000.0, eps=eps, maxit=10000)
##    print "f:", f(X, y, l_, k_, g_, beta, mu=mu)
##    beta, crit = FISTA(X, y, l_, k_, g_, beta, step, mu=mu*1000.0, eps=eps, maxit=1000)
##    print "f:", f(X, y, l_, k_, g_, beta, mu=mu)
##    beta, crit = FISTA(X, y, l_, k_, g_, beta, step, mu=mu*100.0, eps=eps, maxit=1000)
##    print "f:", f(X, y, l_, k_, g_, beta, mu=mu)
##    beta, crit = FISTA(X, y, l_, k_, g_, beta, step, mu=mu*10.0, eps=eps, maxit=1000)
##    print "f:", f(X, y, l_, k_, g_, beta, mu=mu)
##    step = 1.0 / Lipschitz(X, k_, g_, mu)
##    beta, crit = FISTA(X, y, l_, k_, g_, beta, step, mu=mu, eps=eps, maxit=conts * maxit)
##    print "f:", rrl1tv.f(X, y, l_, k_, g_, beta, mu=mu)
##    beta0 = beta
#
#    rrl1tv.rr.k = k_
#    rrl1tv.l1.l = l_
#    rrl1tv.tv.g = g_
#    (beta, fval_, Gval_) = CONESTA(X, y, rrl1tv, beta,
#                                   mumin=mu_zero, sigma=1.01, tau=0.5,
#                                   eps=utils.TOLERANCE,
#                                   conts=conts, maxit=maxit, dynamic=True)
#
##    print "FISTA:  ", f(X, y, l, k, g, beta, mu=mu)
##    print "CONESTA:", f(X, y, l, k, g, beta_test, mu=mu)
#
#    curr_val = np.sum((beta - betastar) ** 2.0)
#    rrl1tv.rr.k = k
#    rrl1tv.l1.l = l
#    rrl1tv.tv.g = g
#    f_ = rrl1tv.f(X, y, beta, mu=mu)
#
#    rrl1tv.rr.k = k_
#    rrl1tv.l1.l = l_
#    rrl1tv.tv.g = g_
#    print "rr:", rrl1tv.rr.f(X, y, beta)
#    print "l1:", rrl1tv.l1.f(beta)
#    print "tv:", rrl1tv.tv.f(X, y, beta, mu)
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
#rrl1tv.rr.k = k
#rrl1tv.l1.l = l
#rrl1tv.tv.g = g
#print "best  f:", rrl1tv.f(X, y, betastar, mu=mu)
#print "found f:", rrl1tv.f(X, y, beta_opt, mu=mu)
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
#print "Gbest:", Gbest
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