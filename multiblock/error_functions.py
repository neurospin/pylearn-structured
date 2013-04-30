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
import warnings


class ErrorFunction(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        super(ErrorFunction, self).__init__(**kwargs)

    @abc.abstractmethod
    def f(self, *args, **kwargs):
        raise NotImplementedError('Abstract method "f" must be specialised!')


class ConvexErrorFunction(ErrorFunction):

    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        print "CxEF created"
        super(ConvexErrorFunction, self).__init__(**kwargs)


class ProximalOperatorErrorFunction(ErrorFunction):

    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        super(ProximalOperatorErrorFunction, self).__init__(**kwargs)

    @abc.abstractmethod
    def prox(self, x):
        raise NotImplementedError('Abstract method "prox" must be ' \
                                  'specialised!')


class DifferentiableErrorFunction(ErrorFunction):

    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        print "DEF created"
        super(DifferentiableErrorFunction, self).__init__(**kwargs)

    @abc.abstractmethod
    def grad(self, *args, **kwargs):
        raise NotImplementedError('Abstract method "grad" must be ' \
                                  'specialised!')


class NesterovErrorFunction(DifferentiableErrorFunction, ConvexErrorFunction):
    """An error function approximated using the Nesterov technique.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, mus=None, **kwargs):

        print "NEF created", mus
        super(NesterovErrorFunction, self).__init__(**kwargs)

        if mus != None:
            self.mus = mus
            self.mu = mus[0]

    @abc.abstractmethod
    def precompute(self, *args, **kwargs):
        raise NotImplementedError('Abstract method "precompute" must be ' \
                                  'specialised!')

    def get_mus(self):
        return self.mus

    def set_mus(self, mus):
        if not isinstance(mus, (tuple, list)):
            mus = [mus]

        self.mus = mus

    def get_mu(self):
        return self.mu

    def set_mu(self, mu):
        self.mu = mu


class CombinedErrorFunction(DifferentiableErrorFunction, ConvexErrorFunction):

    def __init__(self, a=None, b=None, **kwargs):

        if a == None or b == None:
            raise ValueError('Two error functions must be given as ' \
                             'arguments to the constructor')

        print "CEF created", a, b
        super(CombinedErrorFunction, self).__init__(**kwargs)

        self.a = a
        self.b = b

    def f(self, *args, **kwargs):
        return self.a.f(*args, **kwargs) + self.b.f(*args, **kwargs)

    def grad(self, *args, **kwargs):
        return self.a.grad(*args, **kwargs) + self.b.grad(*args, **kwargs)


class CombinedNesterovErrorFunction(CombinedErrorFunction,
                                    NesterovErrorFunction):
    """An error function contructed as the sum of two error functions, where
    at least one of them is a Nesterov error function, i.e. an error function
    computed using the Nestrov technique.

    This function is of the form

        g = g1 + g2,

    where either g1 or g2 is a Nesterov function.
    """

    def __init__(self, a, b, mus):
        print "CNEF created", a, b, mus
        super(CombinedNesterovErrorFunction, self).__init__(a=a, b=b)

        self.set_mus(mus)
        self.set_mu(mus[-1])

    def f(self, *args, **kwargs):
        return self.a.f(*args, **kwargs) + self.b.f(*args, **kwargs)

    def grad(self, *args, **kwargs):
        return self.a.grad(*args, **kwargs) + self.b.grad(*args, **kwargs)

    def precompute(self, *args, **kwargs):
        if isinstance(self.a, NesterovErrorFunction):
            self.a.precompute(*args, **kwargs)
        if isinstance(self.b, NesterovErrorFunction):
            self.b.precompute(*args, **kwargs)

    def get_mus(self):
        if isinstance(self.a, NesterovErrorFunction):
            return self.a.get_mus()
        else:
            return self.b.get_mus()

    def set_mus(self, mus):
        if not isinstance(mus, (tuple, list)):
            mus = [mus]

        if isinstance(self.a, NesterovErrorFunction):
            self.a.set_mus(mus)
        if isinstance(self.b, NesterovErrorFunction):
            self.b.set_mus(mus)

    def get_mu(self):
        if isinstance(self.a, NesterovErrorFunction):
            return self.a.get_mu()
        else:
            return self.b.get_mu()

    def set_mu(self, mu):
        if isinstance(self.a, NesterovErrorFunction):
            self.a.set_mu(mu)
        if isinstance(self.b, NesterovErrorFunction):
            self.b.set_mu(mu)


class ZeroErrorFunction(ConvexErrorFunction, DifferentiableErrorFunction,
                        ProximalOperatorErrorFunction):

    def __init__(self, **kwargs):
        super(ZeroErrorFunction, self).__init__(**kwargs)

    def f(self, *args, **kwargs):
        return 0

    def grad(self, *args, **kwargs):
        return 0

    def prox(self, beta, *args, **kwargs):
        return beta


class SumSqRegressionError(DifferentiableErrorFunction,
                           ConvexErrorFunction):

    def __init__(self, X, y, **kwargs):
        super(SumSqRegressionError, self).__init__(**kwargs)

        self.X = X
        self.y = y

    def f(self, beta, **kwargs):
        return norm(self.y - np.dot(self.X, beta)) ** 2

    def grad(self, beta, **kwargs):
        return 2 * np.dot(self.X.T, np.dot(self.X, beta) - self.y)


class L1(ProximalOperatorErrorFunction, ConvexErrorFunction):

    def __init__(self, l, **kwargs):
        super(L1, self).__init__(**kwargs)

        self.l = l

    def f(self, beta):
        return self.l * norm1(beta)

    def prox(self, x, factor=1, allow_empty=False):

#        xorig = x.copy()
        lorig = factor * self.l
        l = lorig

        return (np.abs(x) > l) * (x - l * np.sign(x - l))

#        warn = False
#        while True:
#            x = xorig
#
#            sign = np.sign(x)
#            np.absolute(x, x)
#            x -= l
#            x[x < 0] = 0
#            x = np.multiply(sign, x)
#
##            print "HERE!!!!"
#
##            if norm(x) > TOLERANCE or allow_empty:
#            break
##            else:
##                warn = True
##                # TODO: Improved this!
##                l *= 0.95  # Reduce by 5 % until at least one significant
#
#        if warn:
#            warnings.warn('Soft threshold was too large (all variables ' \
#                          'purged). Threshold reset to %f (was %f)'
#                          % (l, lorig))
#        return x


class TV(NesterovErrorFunction):

    def __init__(self, shape, gamma, mu, **kwargs):

        super(TV, self).__init__(**kwargs)

        self.shape = shape
        self.gamma = gamma
        self.set_mu(mu)

        self.beta_id = None
        self.mu_id = None

        self.precompute()

    def f(self, beta, mu=None):

        if (mu == None):
            mu = self.get_mus()[-1]

#        print "f:", self.get_mus()

#        if self.beta_id != id(beta) or self.mu_id != id(mu):
        self.compute_alpha(beta, mu)
#            self.beta_id = id(beta)
#            self.mu_id = id(mu)

        return np.dot(self.Aalpha.T, beta)[0, 0] - \
                (mu / 2.0) * (norm(self.asx) ** 2.0 +
                              norm(self.asy) ** 2.0 +
                              norm(self.asz) ** 2.0)

    def grad(self, beta):

#        print "grad:", self.get_mus()

#        if self.beta_id != id(beta) or self.mu_id != id(self.get_mu()):
        self.compute_alpha(beta, self.get_mu())
#            self.beta_id = id(beta)
#            self.mu_id = id(self.get_mu())

        return self.Aalpha

    def compute_alpha(self, beta, mu):

        # Compute a* for each dimension
        q = self.gamma / mu
        self.asx = q * self.Ax.dot(beta)
        self.asy = q * self.Ay.dot(beta)
        self.asz = q * self.Az.dot(beta)

        # Apply projection
        asnorm = self.asx ** 2.0 + self.asy ** 2.0 + self.asz ** 2.0
        asnorm = np.sqrt(asnorm)  # TODO: Speed up by removing the square root
        i = asnorm > 1

        self.asx[i] = np.divide(self.asx[i], asnorm[i])
        self.asy[i] = np.divide(self.asy[i], asnorm[i])
        self.asz[i] = np.divide(self.asz[i], asnorm[i])

        self.Aalpha = self.Ax.T.dot(self.asx) + \
                      self.Ay.T.dot(self.asy) + \
                      self.Az.T.dot(self.asz)

    def precompute(self):

        M = self.shape[0]
        N = self.shape[1]
        O = self.shape[2]
        p = M * N * O

#        from time import time
#        start = time()
        self.Ax = sparse.eye(p, p, 1, format="csr") \
                - sparse.eye(p, p, format="csr")
#        print "Ax sparse:", (time() - start)
#        start = time()
        self.Ay = sparse.eye(p, p, N, format="csr") \
                - sparse.eye(p, p, format="csr")
#        print "Ay sparse:", (time() - start)
#        start = time()
        self.Az = sparse.eye(p, p, M * N, format="csr") \
                - sparse.eye(p, p, format="csr")
#        print "Az sparse:", (time() - start)

#        start = time()
        ind = np.reshape(xrange(p), (O, M, N))
#        print "reshape xrange p time:", (time() - start)
#        start = time()
        xind = ind[:, :, -1].flatten().tolist()
#        print "x slice flatten tolist:", (time() - start)
#        start = time()
        yind = ind[:, -1, :].flatten().tolist()
#        print "y slice flatten tolist:", (time() - start)
#        start = time()
        zind = ind[-1, :, :].flatten().tolist()
#        print "z slice flatten tolist:", (time() - start)

#        start = time()
    #    Ax.data[Ax.indptr[Xxind]] = 0
        for i in xrange(len(xind)):
            self.Ax.data[self.Ax.indptr[xind[i]]: \
                         self.Ax.indptr[xind[i] + 1]] = 0
        self.Ax.eliminate_zeros()

#        print "x remove zero rows:", (time() - start)
#        start = time()

        for i in xrange(len(yind)):
            self.Ay.data[self.Ay.indptr[yind[i]]: \
                         self.Ay.indptr[yind[i] + 1]] = 0
        self.Ay.eliminate_zeros()

#        print "y remove zero rows:", (time() - start)
#        start = time()

    #    Az.data[Az.indptr[M * N] : ] = 0
        for i in xrange(len(zind)):
            self.Az.data[self.Az.indptr[zind[i]]: \
                         self.Az.indptr[zind[i] + 1]] = 0
        self.Az.eliminate_zeros()
#        print "z remove zero rows:", (time() - start)