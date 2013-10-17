# -*- coding: utf-8 -*-
"""
The :mod:`structured.functions` module contains several functions used
throughout the package. These represent mathematical functions and should thus
have any corresponding properties used by the algorithms.

Loss functions should be stateless. Loss functions may be shared and copied and
should therefore not hold anythig that cannot be recomputed the next time it is
called.

Created on Mon Apr 22 10:54:29 2013

@author:  Tommy Löfstedt, Vincent Guillemot and Fouad Hadj Selem
@email:   tommy.loefstedt@cea.fr
@license: TBD.
"""

import numpy as np
import scipy.sparse as sparse

import structured.utils as utils
from structured.utils import math
import structured.algorithms as algorithms

__all__ = ['RidgeRegression', 'L1', 'SmoothedL1', 'TotalVariation']


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

        if self.l < utils.TOLERANCE:
            return 0.0

        if mu == 0.0:
            return self.l * np.sum(np.abs(beta))
        else:
            alpha = self.alpha(beta, mu)
            return self.phi(beta, alpha, mu)

    def phi(self, beta, alpha, mu):

        if self.l < utils.TOLERANCE:
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

#    def Aa(self, alpha):
#
#        return alpha[0]

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

        if self.g < utils.TOLERANCE:
            return 0.0

        if mu > utils.TOLERANCE:
            alpha = self.alpha(beta, mu)
            return self.phi(beta, alpha, mu)
        else:
            A = self.A()
            return self.g * np.sum(np.sqrt(A[0].dot(beta) ** 2.0 + \
                                           A[1].dot(beta) ** 2.0 + \
                                           A[2].dot(beta) ** 2.0))

    def phi(self, beta, alpha, mu):

        if self.g < utils.TOLERANCE:
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

        if self.g < utils.TOLERANCE:
            return 0.0

        if alpha == None:
            alpha = self.alpha(beta, mu)
        grad = self.Aa(alpha)

        return self.g * grad

    def Lipschitz(self, mu, max_iter=10):

        if self.g < utils.TOLERANCE:
            return 0.0

        lmaxA = self.lambda_max(mu, max_iter=max_iter)

        return self.g * lmaxA / mu

    def lambda_max(self, mu, max_iter=100):

        # Note that we can save the state here since lmax(A) does not change.
        if self._lambda_max == None:
            A = sparse.vstack(self.A())
            v = algorithms.FastSparseSVD(A, max_iter=max_iter)
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

    def reset(self):

        self.rr.reset()

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

            if (1.0 / t) * math.norm(beta_new - z) < eps and P - D >= 0 \
                    and i > 100:
                print "Broke after %d iterations" % (i,)
                break

        return P - D


class SmoothedL1TV(object):

    def __init__(self, l, g, shape):

#        self.l1 = SmoothedL1(l, np.prod(shape))
#        self.tv = TotalVariation(g, shape)

        self.l = l
        self.g = g

        self._p = np.prod(shape)

        Atv = TotalVariation.precompute(shape, mask=None, compress=False)
        self._A = [l * sparse.eye(self._p, self._p),
                   g * Atv[0],
                   g * Atv[1],
                   g * Atv[2]]

        self._lambda_max = None

    """ Function value of Ridge regression.
    """
    def f(self, beta, mu=0.0):

        if self.l < utils.TOLERANCE and self.g < utils.TOLERANCE:
            return 0.0

        if mu > utils.TOLERANCE:
            alpha = self.alpha(beta, mu)
            return self.phi(beta, alpha, mu)
        else:
            A = self.A()
            return math.norm1(A[0].dot(beta)) + \
                   np.sum(np.sqrt(A[1].dot(beta) ** 2.0 + \
                                  A[2].dot(beta) ** 2.0 + \
                                  A[3].dot(beta) ** 2.0))

    def phi(self, beta, alpha, mu):

        if self.l < utils.TOLERANCE and self.g < utils.TOLERANCE:
            return 0.0

        Aa = self.A(alpha=alpha)

        alpha_sqsum = 0.0
        for a in alpha:
            alpha_sqsum += np.sum(a ** 2.0)

        return np.dot(Aa.T, beta)[0, 0] - (mu / 2.0) * alpha_sqsum

#    def Lipschitz(self, max_iter=100):
#
#        return self.lambda_max(max_iter=max_iter)

    def lambda_max(self, max_iter=100):

        # Note that we can save the state here since lmax(A) does not change.
        if self._lambda_max == None:
            A = sparse.vstack(self.A())
            v = algorithms.FastSparseSVD(A, max_iter=max_iter)
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

        A = self.A()

        a = [0] * len(A)

        a[0] = (1.0 / mu) * A[0].dot(beta)
        a[1] = (1.0 / mu) * A[1].dot(beta)
        a[2] = (1.0 / mu) * A[2].dot(beta)
        a[3] = (1.0 / mu) * A[3].dot(beta)

        return self.project(a)

    def project(self, a):

        # L1
        al1 = a[0]
        anorm_l1 = np.abs(al1)
        i_l1 = anorm_l1 > 1.0
        anorm_l1_i = anorm_l1[i_l1]
        al1[i_l1] = np.divide(al1[i_l1], anorm_l1_i)

        # TV
        ax = a[1]
        ay = a[2]
        az = a[3]
        anorm_tv = ax ** 2.0 + ay ** 2.0 + az ** 2.0
        i_tv = anorm_tv > 1.0

        anorm_tv_i = anorm_tv[i_tv] ** 0.5  # Square root is taken here. Faster.
        ax[i_tv] = np.divide(ax[i_tv], anorm_tv_i)
        ay[i_tv] = np.divide(ay[i_tv], anorm_tv_i)
        az[i_tv] = np.divide(az[i_tv], anorm_tv_i)

        return [al1, ax, ay, az]

    def M(self):

        A = self.A()

        return (A[0].shape[0] / 2.0) \
             + (A[1].shape[0] / 2.0)


class OLSL2_SmoothedL1TV(object):

    def __init__(self, k, l, g, shape):

        self.g = RidgeRegression(k)
        self.h = SmoothedL1TV(l, g, shape)

        self.reset()

    def reset(self):

        self.g.reset()
#        self.h.reset()

        self._Xy = None
        self._XtinvXXtkI = None

    """ Function value of Ridge regression and TV.
    """
    def f(self, X, y, beta, mu):

        return self.g.f(X, y, beta) \
             + self.h.f(beta, mu)

    def Lipschitz(self, X, max_iter=100):

        print "self.h.lambda_max(max_iter=max_iter):", self.h.lambda_max(max_iter=max_iter)
        print "self.g.lambda_min(X):", self.g.lambda_min(X)
        return self.h.lambda_max(max_iter=max_iter) / self.g.lambda_min(X)

    def V(self, u, beta, L):

        A = self.h.A()
        a = [0] * len(A)
        a[0] = (1.0 / L) * A[0].dot(beta)
        a[1] = (1.0 / L) * A[1].dot(beta)
        a[2] = (1.0 / L) * A[2].dot(beta)
        a[3] = (1.0 / L) * A[3].dot(beta)

        u_new = [0] * len(u)
        for i in xrange(len(u)):
            u_new[i] = u[i] + a[i]

        return self.h.project(u_new)

    """ Returns the beta that minimises the dual function.
    """
    def betahat(self, X, y, alpha):

        A = self.h.A()
        grad = A[0].T.dot(alpha[0])
        grad += A[1].T.dot(alpha[1])
        grad += A[2].T.dot(alpha[2])
        grad += A[3].T.dot(alpha[3])

#        XXkI = np.dot(X.T, X) + self.g.k * np.eye(X.shape[1])

        if self._Xy == None:
            self._Xy = np.dot(X.T, y)

        Xty_grad = (self._Xy - grad) / self.g.k

#        t = time()
#        XXkI = np.dot(X.T, X)
#        index = np.arange(min(XXkI.shape))
#        XXkI[index, index] += self.g.k
#        invXXkI = np.linalg.inv(XXkI)
#        print "t:", time() - t
#        beta = np.dot(invXXkI, Xty_grad)

#        t = time()
        if self._XtinvXXtkI == None:
            XXtkI = np.dot(X, X.T)
            index = np.arange(min(XXtkI.shape))
            XXtkI[index, index] += self.g.k
            invXXtkI = np.linalg.inv(XXtkI)
            self._XtinvXXtkI = np.dot(X.T, invXXtkI)

        beta = (Xty_grad - np.dot(self._XtinvXXtkI, np.dot(X, Xty_grad)))
#        print "t:", time() - t

        return beta