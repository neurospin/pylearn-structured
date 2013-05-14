# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 10:54:29 2013

@author: tl236864
"""

__all__ = ['ErrorFunction', 'ConvexErrorFunction',
           'ProximalOperatorErrorFunction', 'DifferentiableErrorFunction',
           'NesterovErrorFunction', 'CombinedErrorFunction',
           'CombinedNesterovErrorFunction',

           'ZeroErrorFunction', 'SumSqRegressionError',
           'LogisticRegressionError',
           'L1',
           'TotalVariation', 'GroupLassoOverlap']

import abc
import numpy as np
import scipy.sparse as sparse
import math
from time import time
import warnings

import algorithms
from utils import warning, norm, norm1, TOLERANCE


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
        super(DifferentiableErrorFunction, self).__init__(**kwargs)

    @abc.abstractmethod
    def grad(self, *args, **kwargs):
        raise NotImplementedError('Abstract method "grad" must be ' \
                                  'specialised!')

    @abc.abstractmethod
    def Lipschitz(self):
        raise NotImplementedError('Abstract method "grad" must be ' \
                                  'specialised!')


class NesterovErrorFunction(DifferentiableErrorFunction, ConvexErrorFunction):
    """An error function approximated using the Nesterov technique.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, mus=None, **kwargs):
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

        super(CombinedErrorFunction, self).__init__(**kwargs)

        self.a = a
        self.b = b

    def f(self, *args, **kwargs):
        return self.a.f(*args, **kwargs) + self.b.f(*args, **kwargs)

    def grad(self, *args, **kwargs):
        return self.a.grad(*args, **kwargs) + self.b.grad(*args, **kwargs)

    def Lipschitz(self):
        return self.a.Lipschitz() + self.b.Lipschitz()


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
        super(CombinedNesterovErrorFunction, self).__init__(a=a, b=b)

        self.set_mus(mus)
        self.set_mu(mus[-1])

#    def f(self, *args, **kwargs):
#        return self.a.f(*args, **kwargs) + self.b.f(*args, **kwargs)
#
#    def grad(self, *args, **kwargs):
#        return self.a.grad(*args, **kwargs) + self.b.grad(*args, **kwargs)

    def precompute(self, *args, **kwargs):
#        if isinstance(self.a, NesterovErrorFunction):
        if hasattr(self.a, 'precompute'):
            self.a.precompute(*args, **kwargs)
#        if isinstance(self.b, NesterovErrorFunction):
        if hasattr(self.b, 'precompute'):
            self.b.precompute(*args, **kwargs)

    def get_mus(self):
#        if isinstance(self.a, NesterovErrorFunction):
        if hasattr(self.a, 'get_mus'):
            return self.a.get_mus()
        else:
            return self.b.get_mus()

    def set_mus(self, mus):
        if not isinstance(mus, (tuple, list)):
            mus = [mus]

#        if isinstance(self.a, NesterovErrorFunction):
        if hasattr(self.a, 'set_mus'):
            self.a.set_mus(mus)
#        if isinstance(self.b, NesterovErrorFunction):
        if hasattr(self.b, 'set_mus'):
            self.b.set_mus(mus)

    def get_mu(self):
#        if isinstance(self.a, NesterovErrorFunction):
        if hasattr(self.a, 'get_mu'):
            return self.a.get_mu()
        else:
            return self.b.get_mu()

    def set_mu(self, mu):
#        if isinstance(self.a, NesterovErrorFunction):
        if hasattr(self.a, 'set_mu'):
            self.a.set_mu(mu)
#        if isinstance(self.b, NesterovErrorFunction):
        if hasattr(self.b, 'set_mu'):
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

    def Lipschitz(self):
        return 0


class SumSqRegressionError(DifferentiableErrorFunction,
                           ConvexErrorFunction):

    def __init__(self, X, y, **kwargs):
        super(SumSqRegressionError, self).__init__(**kwargs)

        self.X = X
        self.y = y

#        self._Xy = np.dot(self.X.T, self.y)

#        D, V = np.linalg.eig(np.dot(self.X.T, self.X))
#        self.t = np.max(D.real)
        _, s, _ = np.linalg.svd(X, full_matrices=False)  # False == faster
        self.t = np.max(s) ** 2.0

    def f(self, beta, **kwargs):
        return norm(self.y - np.dot(self.X, beta)) ** 2

    def grad(self, beta, **kwargs):
        return 2 * np.dot(self.X.T, np.dot(self.X, beta) - self.y)
#        return 2 * (np.dot(self.X.T, np.dot(self.X, beta)) - self._Xy)

    def Lipschitz(self):
        return self.t


class LogisticRegressionError(DifferentiableErrorFunction,
                              ConvexErrorFunction):

    def __init__(self, X, y, **kwargs):
        super(LogisticRegressionError, self).__init__(**kwargs)

        self.X = X
        self.y = y

        V = 0.5 * np.eye(X.shape[0])  # pi(x) * (1 - pi(x)) <= 0.25 = 0.5 * 0.5
        VX = np.dot(V, X)
        _, s, _ = np.linalg.svd(VX, full_matrices=False)  # False == faster
        self.t = np.max(s) ** 2.0

    def f(self, beta, **kwargs):
        logit = np.dot(self.X, beta)
        expt = np.exp(logit)
        return -np.sum(np.multiply(self.y, logit) + np.log(1 + expt))

    def grad(self, beta, **kwargs):
        logit = np.dot(self.X, beta)
        expt = np.exp(logit)
        pix = np.divide(expt, expt + 1)
        return -np.dot(self.X.T, self.y - pix)

    def hessian(self, beta, **kwargs):
        logit = np.dot(self.X, beta)
        expt = np.exp(logit)
        pix = np.divide(expt, expt + 1)
        pixpix = np.multiply(pix, 1 - pix)

        V = np.diag(pixpix.flatten())
        XVX = np.dot(np.dot(self.X.T, V), self.X)

        return XVX

    def Lipschitz(self):
        return self.t


class L1(ProximalOperatorErrorFunction, ConvexErrorFunction):

    def __init__(self, l, **kwargs):
        super(L1, self).__init__(**kwargs)

        self.l = l

    def f(self, beta):
        return self.l * norm1(beta)

    def prox(self, x, factor=1, allow_empty=False):

        l = factor * self.l
        return (np.abs(x) > l) * (x - l * np.sign(x - l))

#        xorig = x.copy()
#        lorig = factor * self.l
#        l = lorig
#
#        warn = False
#        while True:
#            x = xorig

#        sign = np.sign(x)
#        np.absolute(x, x)
#        x -= l
#        x[x < 0] = 0
#        x = np.multiply(sign, x)

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
        return x


class TotalVariation(NesterovErrorFunction):

    def __init__(self, shape, gamma, mu, **kwargs):

        super(TotalVariation, self).__init__(**kwargs)

        self.shape = shape
        self.gamma = gamma
        self.set_mu(mu)

        self.beta_id = None
        self.mu_id = None

        self.precompute()

        A = sparse.vstack((self.Ax, self.Ay, self.Az))
        v = algorithms.SparseSVD(max_iter=10).run(A)
        u = A.dot(v)
        self.lambda_max = np.sum(u ** 2.0)

    def f(self, beta, mu=None):

        if self.gamma <= TOLERANCE:
            return 0

        if (mu == None):
            mu = self.get_mus()[-1]

        if self.beta_id != id(beta) or self.mu_id != id(mu):
            self.compute_alpha(beta, mu)
            self.beta_id = id(beta)
            self.mu_id = id(mu)

        return np.dot(self.Aalpha.T, beta)[0, 0] - \
                (mu / 2.0) * (np.sum(self.asx ** 2.0) +
                              np.sum(self.asy ** 2.0) +
                              np.sum(self.asz ** 2.0))

    def grad(self, beta):

        if self.gamma <= TOLERANCE:
            return np.zeros((np.prod(self.shape), 1))

        if self.beta_id != id(beta) or self.mu_id != id(self.get_mu()):
            self.compute_alpha(beta, self.get_mu())
            self.beta_id = id(beta)
            self.mu_id = id(self.get_mu())

        return self.Aalpha

    def Lipschitz(self):

        if self.gamma < TOLERANCE:
            return 0

        return self.lambda_max / self.get_mu()

    def compute_alpha(self, beta, mu):

        # Compute a* for each dimension
        q = self.gamma / mu
        self.asx = q * self.Ax.dot(beta)
        self.asy = q * self.Ay.dot(beta)
        self.asz = q * self.Az.dot(beta)

        # Apply projection
        asnorm = self.asx ** 2.0 + self.asy ** 2.0 + self.asz ** 2.0  # )**0.5
#        asnorm = np.sqrt(asnorm)
        i = asnorm > 1

        asnorm_i = asnorm[i] ** 0.5  # Square root is taken here. Faster.
        self.asx[i] = np.divide(self.asx[i], asnorm_i)
        self.asy[i] = np.divide(self.asy[i], asnorm_i)
        self.asz[i] = np.divide(self.asz[i], asnorm_i)

#        self.Aalpha = self.Ax.T.dot(self.asx) + \
#                      self.Ay.T.dot(self.asy) + \
#                      self.Az.T.dot(self.asz)
#        self.Aalpha = np.add(np.add(self.Ax.T.dot(self.asx),
#                                    self.Ay.T.dot(self.asy), self.buff),
#                                    self.Az.T.dot(self.asz), self.buff)
        self.Aalpha = np.add(np.add(self.Axt.dot(self.asx),
                                    self.Ayt.dot(self.asy), self.buff),
                                    self.Azt.dot(self.asz), self.buff)

    def precompute(self):

        M = self.shape[0]
        N = self.shape[1]
        O = self.shape[2]
        p = M * N * O

        smtype = 'csr'
        self.Ax = sparse.eye(p, p, 1, format=smtype) \
                - sparse.eye(p, p)
        self.Ay = sparse.eye(p, p, N, format=smtype) \
                - sparse.eye(p, p)
        self.Az = sparse.eye(p, p, M * N, format=smtype) \
                - sparse.eye(p, p)

        ind = np.reshape(xrange(p), (O, M, N))
        xind = ind[:, :, -1].flatten().tolist()
        yind = ind[:, -1, :].flatten().tolist()
        zind = ind[-1, :, :].flatten().tolist()

        for i in xrange(len(xind)):
            self.Ax.data[self.Ax.indptr[xind[i]]: \
                         self.Ax.indptr[xind[i] + 1]] = 0
        self.Ax.eliminate_zeros()

        for i in xrange(len(yind)):
            self.Ay.data[self.Ay.indptr[yind[i]]: \
                         self.Ay.indptr[yind[i] + 1]] = 0
        self.Ay.eliminate_zeros()

#        for i in xrange(len(zind)):
#            self.Az.data[self.Az.indptr[zind[i]]: \
#                         self.Az.indptr[zind[i] + 1]] = 0
        self.Az.data[self.Az.indptr[zind[0]]: \
                     self.Az.indptr[zind[-1] + 1]] = 0
        self.Az.eliminate_zeros()

        self.Axt = self.Ax.T
        self.Ayt = self.Ay.T
        self.Azt = self.Az.T

        self.buff = np.zeros((self.Ax.shape[0], 1))


class GroupLassoOverlap(NesterovErrorFunction):

    def __init__(self, num_variables, groups, gamma, mu, weights=None,
                 **kwargs):
        """
        Parameters:
        ----------
        groups: A list of list, with the outer list being the groups and the
                inner list the variables in the group. E.g. [[1,2],[2,3]]
                contains two groups ([1,2] and [2,3]) with variable 1 and 2
                in the first group and variables 2 and 3 in the second
                group.
        """

        super(GroupLassoOverlap, self).__init__(**kwargs)

        self.num_variables = num_variables
        self.groups = groups
        self.gamma = gamma
        self.set_mu(mu)

        if weights == None:
            self.weights = [1] * len(groups)
        else:
            self.weights = weights

        self.beta_id = None
        self.mu_id = None

        self.precompute()

    def f(self, beta, mu=None):

        if (mu == None):
            mu = self.get_mus()[-1]

        if self.beta_id != id(beta) or self.mu_id != id(mu):
            self.compute_alpha(beta, mu)
            self.beta_id = id(beta)
            self.mu_id = id(mu)

        sumastar = 0
        for g in xrange(len(self.astar)):
            sumastar += np.sum(self.astar[g] ** 2.0)

        return np.dot(self.Aalpha.T, beta)[0, 0] - (mu / 2.0) * sumastar

    def grad(self, beta):

#        print "grad:", self.get_mus()

        if self.beta_id != id(beta) or self.mu_id != id(self.get_mu()):
            self.compute_alpha(beta, self.get_mu())
            self.beta_id = id(beta)
            self.mu_id = id(self.get_mu())

        return self.Aalpha

    def Lipschitz(self):

        return self.max_col_norm / self.mu

    def compute_alpha(self, beta, mu):

        # Compute a* for each dimension
        self.Aalpha = 0
        q = self.gamma / mu
        for g in xrange(len(self.A)):
            astar = q * self.A[g].dot(beta)
            normas = np.sqrt(np.dot(astar.T, astar))
            if normas > 1:
                astar /= normas

            self.astar[g] = astar

#            self.Aalpha += self.A[g].T.dot(astar)
            self.Aalpha += self.At[g].dot(astar)

    def precompute(self):

        self.A = list()
        self.At = list()

        powers = np.zeros(self.num_variables)
        for g in xrange(len(self.groups)):
            Gi = self.groups[g]
            lenGi = len(Gi)
            Ag = sparse.lil_matrix((lenGi, self.num_variables))
            for i in xrange(lenGi):
                w = self.weights[g]
                Ag[i, Gi[i]] = w
                powers[Gi[i]] += w ** 2.0

            # Matrix operations are a lot faster when the sparse matrix is csr
            self.A.append(Ag.tocsr())
            self.At.append(self.A[-1].T)

        self.max_col_norm = np.sqrt(np.max(powers))

        self.astar = [0] * len(self.groups)