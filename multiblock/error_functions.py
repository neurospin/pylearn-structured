# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 10:54:29 2013

@author: tl236864
"""

__all__ = ['ErrorFunction', 'ConvexErrorFunction',
           'DifferentiableErrorFunction', 'NonDifferentiableErrorFunction']

import abc
import numpy as np
import scipy.sparse as sparse

from utils import norm, norm1, TOLERANCE
import prox_ops
import warnings


class ErrorFunction(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        super(ErrorFunction, self).__init__()


class ConvexErrorFunction(ErrorFunction):

    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        super(ConvexErrorFunction, self).__init__(**kwargs)

    @abc.abstractmethod
    def f(self, *args, **kwargs):
        raise NotImplementedError('Abstract method "f" must be specialised!')


class ProximalOperatorErrorFunction(ErrorFunction):

    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        super(ProximalOperatorErrorFunction, self).__init__(**kwargs)

    @abc.abstractmethod
    def f(self, *args, **kwargs):
        raise NotImplementedError('Abstract method "f" must be specialised!')

    @abc.abstractmethod
    def prox(self, x):
        raise NotImplementedError('Abstract method "prox" must be ' \
                                  'specialised!')


class DifferentiableErrorFunction(ErrorFunction):

    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        super(DifferentiableErrorFunction, self).__init__(**kwargs)

    @abc.abstractmethod
    def f(self, *args, **kwargs):
        raise NotImplementedError('Abstract method "f" must be specialised!')

    @abc.abstractmethod
    def grad(self, *args, **kwargs):
        raise NotImplementedError('Abstract method "grad" must be ' \
                                  'specialised!')


class NonDifferentiableErrorFunction(ErrorFunction):

    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        super(NonDifferentiableErrorFunction, self).__init__(**kwargs)

    @abc.abstractmethod
    def f(self, *args, **kwargs):
        raise NotImplementedError('Abstract method "f" must be specialised!')


class CompinedDifferentiableErrorFunction(DifferentiableErrorFunction,
                                          ConvexErrorFunction):

    def __init__(self, a, b, **kwargs):
        super(CompinedDifferentiableErrorFunction, self).__init__(**kwargs)

        self.a = a
        self.b = b

    def f(self, *args, **kwargs):
        return a.f(*args, **kwargs) + b.f(*args, **kwargs)

    def grad(self, *args, **kwargs):
        return a.grad(*args, **kwargs) + b.grad(*args, **kwargs)


class ZeroErrorFunction(ConvexErrorFunction, DifferentiableErrorFunction,
                        ProximalOperatorErrorFunction):

    def __init__(self):
        super(ZeroErrorFunction, self).__init__()

    def f(self, *args, **kwargs):
        return 0

    def grad(self, *args, **kwargs):
        return 0

    def prox(self, beta, *args, **kwargs):
        return beta


class MeanSquareRegressionError(DifferentiableErrorFunction,
                                ConvexErrorFunction):

    def __init__(self, X, y):
        super(MeanSquareRegressionError, self).__init__()

        self.X = X
        self.y = y

    def f(self, beta):
        return norm(self.y - np.dot(self.X, beta)) ** 2

    def grad(self, beta):
        return 2 * np.dot(self.X.T, np.dot(self.X, beta) - self.y)


class L1(ProximalOperatorErrorFunction, ConvexErrorFunction):

    def __init__(self, l):
        super(L1, self).__init__()

        self.l = l

#        if prox_op == None:
#            self.prox_op = prox_ops.L1(self.l)
#        else:
#            self.prox_op = prox_op

    def f(self, beta):
        return self.l * norm1(beta)

#    def prox(self, beta):
#        return self.prox_op.prox(beta)

    def prox(self, x, factor=1, allow_empty=False):

        xorig = x.copy()
        lorig = factor * self.l
        l = lorig

        warn = False
        while True:
            x = xorig

            sign = np.sign(x)
            np.absolute(x, x)
            x -= l
            x[x < 0] = 0
            x = np.multiply(sign, x)

            if norm(x) > TOLERANCE or allow_empty:
                break
            else:
                warn = True
                # TODO: Improved this!
                l *= 0.95  # Reduce by 5 % until at least one significant

        if warn:
            warnings.warn('Soft threshold was too large (all variables ' \
                          'purged). Threshold reset to %f (was %f)'
                          % (l, lorig))
        return x


class TV(DifferentiableErrorFunction,
         ConvexErrorFunction):

    def __init__(self, shape, gamma, mu):
        super(TV, self).__init__()

        self.shape = shape
        self.gamma = gamma
        self.mu = mu
        self.beta_id = None

        self.precompute()

    def f(self, beta):

        if self.beta_id != id(beta):
            self.compute_alpha(beta)
            self.beta_id = id(beta)

        return np.dot(self.gAalpha, beta) - \
                (self.gamma * self.mu / 2.0) * (norm(self.asx) ** 2.0 +
                                                norm(self.asy) ** 2.0 +
                                                norm(self.asz) ** 2.0)

    def grad(self, beta):

        if self.beta_id != id(beta):
            self.compute_alpha(beta)
            self.beta_id = id(beta)

        return self.gAalpha

    def set_mu(self, mu):
        self.mu = mu

    def compute_alpha(self, beta):

        # Compute a* for each dimension
        q = self.gamma / self.mu
        print self.Ax.shape
        print beta.shape
        self.asx = q * self.Ax.dot(beta)
        self.asy = q * self.Ay.dot(beta)
        self.asz = q * self.Az.dot(beta)

        # Apply projection
        asnorm = self.asx ** 2.0 + self.asy ** 2.0 + self.asz ** 2.0
        asnorm = np.sqrt(asnorm)  # TODO: Optimise by removing the square root
        i = asnorm > 1
        self.asx[i] = np.divide(self.asx[i], asnorm[i])
        self.asy[i] = np.divide(self.asy[i], asnorm[i])
        self.asz[i] = np.divide(self.asz[i], asnorm[i])

        self.gAalpha = self.gamma * (np.dot(self.Ax.T, self.asx) + \
                                     np.dot(self.Ay.T, self.asy) + \
                                     np.dot(self.Az.T, self.asz))

    def precompute(self):

        M = self.shape[0]
        N = self.shape[1]
        O = self.shape[2]
        p = M * N * O

        self.Ax = sparse.eye(p, p, 1, format="csr") \
                - sparse.eye(p, p, format="csr")
        self.Ay = sparse.eye(p, p, M, format="csr") \
                - sparse.eye(p, p, format="csr")
        self.Az = sparse.eye(p, p, M * N, format="csr") \
                - sparse.eye(p, p, format="csr")

        ind = np.reshape(xrange(p), (O, M, N))
        xind = ind[:, :, -1].flatten().tolist()
        yind = ind[:, -1, :].flatten().tolist()
        zind = ind[-1, :, :].flatten().tolist()

    #    Ax.data[Ax.indptr[Xxind]] = 0
        for i in xrange(len(xind)):
            self.Ax.data[self.Ax.indptr[xind[i]]: \
                         self.Ax.indptr[xind[i] + 1]] = 0
        self.Ax.eliminate_zeros()

        for i in xrange(len(yind)):
            self.Ay.data[self.Ay.indptr[yind[i]]: \
                         self.Ay.indptr[yind[i] + 1]] = 0
        self.Ay.eliminate_zeros()

    #    Az.data[Az.indptr[M * N] : ] = 0
        for i in xrange(len(zind)):
            self.Az.data[self.Az.indptr[zind[i]]: \
                         self.Az.indptr[zind[i] + 1]] = 0
        self.Az.eliminate_zeros()