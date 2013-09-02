# -*- coding: utf-8 -*-
"""
Loss functions should, as far as possible, be stateless. Loss functions may be
shared and copied and should therefore not hold anythig that cannot be
recomputed the next time it is called.

Created on Mon Apr 22 10:54:29 2013

@author:  Tommy Löfstedt, Vincent Guillemot and Fouad Hadj Selem
@email:   tommy.loefstedt@cea.fr
@license: TBD.
"""

__all__ = ['LossFunction', 'LipschitzContinuous', 'Differentiable',
           'DataDependent',

           'Convex',
           'LinearRegressionError', 'LogisticRegressionError',

           'StronglyConvex',
           'RidgeRegression',

           'LinearLossFunction',

           'ProximalOperator',
           'ZeroErrorFunction',
           'L1', 'L2', 'ElasticNet',

           'NesterovFunction',
           'TotalVariation', 'SmoothL1', 'GroupLassoOverlap',

           'CombinedLossFunction',
           'CombinedNesterovLossFunction']

import abc
import numpy as np
import scipy.sparse as sparse
import copy

import algorithms
#from utils import norm, norm1, TOLERANCE, delete_sparse_csr_row
import utils
from utils import math


class LossFunction(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):

        super(LossFunction, self).__init__(**kwargs)

    @abc.abstractmethod
    def f(self, *args, **kwargs):

        raise NotImplementedError('Abstract method "f" must be specialised!')


class LipschitzContinuous(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):

        super(LipschitzContinuous, self).__init__(**kwargs)

    @abc.abstractmethod
    def Lipschitz(self, *args, **kwargs):

        raise NotImplementedError('Abstract method "Lipschitz" must be ' \
                                  'specialised!')


class Differentiable(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):

        super(Differentiable, self).__init__(**kwargs)

    @abc.abstractmethod
    def grad(self, *args, **kwargs):

        raise NotImplementedError('Abstract method "grad" must be ' \
                                  'specialised!')


class DataDependent(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):

        super(DataDependent, self).__init__(**kwargs)

    @abc.abstractmethod
    def set_data(self, *X):

        raise NotImplementedError('Abstract method "set_data" must be ' \
                                  'specialised!')

    @abc.abstractmethod
    def get_data(self):

        raise NotImplementedError('Abstract method "get_data" must be ' \
                                  'specialised!')

    @abc.abstractmethod
    def free_data(self):

        raise NotImplementedError('Abstract method "free_data" must be ' \
                                  'specialised!')


class Convex(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):

        super(Convex, self).__init__(**kwargs)


class StronglyConvex(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):

        super(StronglyConvex, self).__init__(**kwargs)

    @abc.abstractmethod
    def lambda_min(self):

        raise NotImplementedError('Abstract method "lambda_min" must be ' \
                                  'specialised!')


class ProximalOperator(LossFunction, Convex):

    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):

        super(ProximalOperator, self).__init__(**kwargs)

    @abc.abstractmethod
    def prox(self, x):

        raise NotImplementedError('Abstract method "prox" must be ' \
                                  'specialised!')


class NesterovFunction(LossFunction,
                       Convex,
                       Differentiable,
                       LipschitzContinuous):
    """A loss function approximated using the Nesterov smoothing technique.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, gamma=None, mu=None, **kwargs):
        """Construct a Nesterov loss function.

        Parameters
        ----------
        gamma : The regularisation parameter for the nesterov penality.

        mu : The Nesterov function regularisation parameter. Must be provided
                unless you are using ContinuationRun.
        """
        super(NesterovFunction, self).__init__(**kwargs)

        if gamma != None:
            self.gamma = float(gamma)
        else:
            self.gamma = gamma

        if mu != None:
            self.set_mu(mu)

        self._A = None  # The linear operator
        self._At = None  # The linear operator transposed
        self._lambda_max = None  # The largest eigenvalue of A'.A
        self._num_compacts = None

    def f(self, beta, mu=None, **kwargs):

        if self.gamma < utils.TOLERANCE:
            return 0.0

        alpha = self._compute_alpha(beta, mu)
        grad = self._compute_grad(alpha)

        alpha_sqsum = 0.0
        for a in alpha:
            alpha_sqsum += np.sum(a ** 2.0)

        if mu == None:
            mu = self.get_mu()

        return self.gamma * (np.dot(beta.T, grad)[0, 0] \
                             - (mu / 2.0) * alpha_sqsum)

    def phi(self, beta, alpha, mu=None):

        grad = self._compute_grad(alpha)

        if mu == None:

            return self.gamma * np.dot(beta.T, grad)[0, 0]

        else:
#            alpha_sqsum = 0.0
#            for a in alpha:
#                alpha_sqsum += np.sum(a ** 2.0)

#            print "alpha: ", alpha_sqsum

            return self.gamma * (np.dot(beta.T, grad)[0, 0])  # \
#                                 - (mu / 2.0) * alpha_sqsum)

    def grad(self, beta, alpha=None, mu=None):

        if self.gamma < utils.TOLERANCE:
            return np.zeros(beta.shape)

        if alpha == None:
            alpha = self._compute_alpha(beta, mu)

        grad = self._compute_grad(alpha)

        return self.gamma * grad

    def Lipschitz(self, mu=None):
        """The Lipschitz constant of the gradient.

        In general the Lipschitz constant is the largest eigenvalue of A'.A
        divided by mu, but should be specialised in subclasses if faster
        approaches exist.
        """
        if self.gamma < utils.TOLERANCE:
            return 0.0

        if self._lambda_max == None:  # The squared largest singular value
            A = sparse.vstack(self.A())
            v = algorithms.SparseSVD(max_iter=100).run(A)
            us = A.dot(v)
            self._lambda_max = np.sum(us ** 2.0)

        if mu != None:
            return (self.gamma ** 2.0) * self._lambda_max / mu
        else:
            return (self.gamma ** 2.0) * self._lambda_max / self.get_mu()

    def A(self):
        """Returns a list of the A blocks that constitute the A matrix, the
        linear operator.
        """
        return self._A

    def At(self):
        """Returns a list of the transposed A blocks that constitute the A
        matrix, the linear operator.
        """
        if self._At == None:
            self._At = []
            for A in self.A():
                self._At.append(A.T)

        return self._At

    def alpha(self, beta, mu=None):

        if self.gamma < utils.TOLERANCE:
            ret = []
            for A in self.A():
                ret.append(np.zeros((A.shape[0], 1)))

            return ret

        alpha = self._compute_alpha(beta, mu)

        return alpha

    def get_mu(self):
        """Returns the Nesterov regularisation constant mu.
        """
        return self.mu

    def set_mu(self, mu):
        """Sets the Nesterov regularisation constant mu.
        """
        self.mu = float(mu)

    def num_groups(self):
        """Returns the number of groups, i.e. the number of A blocks.
        """
        return len(self.A())

    def _count_compacts(self, A):

        rowsum = np.asarray(A[0].sum(axis=1)).ravel().tolist()
        return sum([1 for x in rowsum if x > 0.0])

    def num_compacts(self):
        """The smoothness gap between f and f_mu is mu * num_compacts() / 2.

        Note that the constant D = num_compacts() / 2.
        """
        if self._num_compacts == None:
            self._num_compacts = self._count_compacts(self.A())

        return self._num_compacts

    def _compute_grad(self, alpha):

        At = self.At()
        grad = At[0].dot(alpha[0])
        for i in xrange(1, len(At)):
            grad += At[i].dot(alpha[i])

        return grad

    @abc.abstractmethod
    def _compute_alpha(self, beta, mu=None):

        raise NotImplementedError('Abstract method "_compute_alpha" must be ' \
                                  'specialised!')

    @abc.abstractmethod
    def project(self, *alpha):

        raise NotImplementedError('Abstract method "project" must be ' \
                                  'specialised!')

    @abc.abstractmethod
    def precompute(*args, **kwargs):

        raise NotImplementedError('Abstract method "precompute" must be ' \
                                  'specialised!')


class CombinedLossFunction(LossFunction, LipschitzContinuous, Differentiable,
                           DataDependent):

    def __init__(self, a, b, **kwargs):

        super(CombinedLossFunction, self).__init__(**kwargs)

        self.a = a
        self.b = b

    def f(self, *args, **kwargs):

        return self.a.f(*args, **kwargs) + self.b.f(*args, **kwargs)

    def grad(self, *args, **kwargs):

        return self.a.grad(*args, **kwargs) + self.b.grad(*args, **kwargs)

    def Lipschitz(self):

        if isinstance(self.a, LipschitzContinuous) \
                and isinstance(self.b, LipschitzContinuous):

            return self.a.Lipschitz() + self.b.Lipschitz()

        elif isinstance(self.a, LipschitzContinuous):

            return self.a.Lipschitz()

        elif isinstance(self.b, LipschitzContinuous):

            return self.b.Lipschitz()

        return 0

    def set_data(self, *args, **kwargs):

        if hasattr(self.a, 'set_data'):

            self.a.set_data(*args, **kwargs)

        if hasattr(self.b, 'set_data'):

            self.b.set_data(*args, **kwargs)

    def get_data(self):

        data = tuple()
        if hasattr(self.a, 'get_data'):

            data += tuple(self.a.get_data())

        if hasattr(self.b, 'get_data'):

            data += tuple(self.b.get_data())

        # Remove doubles, preserve order
        ids = []
        unique = []
        for d in data:
            if id(d) not in ids:
                ids.append(id(d))
                unique.append(d)

        return tuple(unique)


class CombinedNesterovLossFunction(NesterovFunction, DataDependent):
    """A loss function contructed as the sum of two loss functions, where
    at least one of them is a Nesterov function, i.e. a loss function computed
    using the Nestrov technique.

    This function is of the form

        g = g1 + g2,

    where at least one of g1 or g2 is a Nesterov functions.
    """
    def __init__(self, a, b, **kwargs):

        super(CombinedNesterovLossFunction, self).__init__(**kwargs)

        self.a = a
        self.b = b

    def f(self, *args, **kwargs):

        return self.a.f(*args, **kwargs) + self.b.f(*args, **kwargs)

    def phi(self, beta, alpha, *args, **kwargs):

        if isinstance(self.a, NesterovFunction) \
                and isinstance(self.b, NesterovFunction):

            groups_a = self.a.num_groups()

            return self.a.phi(beta, alpha[:groups_a], *args, **kwargs) \
                    + self.b.phi(beta, alpha[groups_a:], *args, **kwargs)

        elif isinstance(self.a, NesterovFunction):

            return self.a.phi(beta, alpha, *args, **kwargs) \
                    + self.b.f(beta, *args, **kwargs)

        elif isinstance(self.b, NesterovFunction):

            return self.a.f(beta, *args, **kwargs) \
                    + self.b.phi(beta, alpha, *args, **kwargs)

        else:

            return self.a.f(beta, *args, **kwargs) \
                    + self.b.f(beta, *args, **kwargs)

    def grad(self, *args, **kwargs):

        return self.a.grad(*args, **kwargs) + self.b.grad(*args, **kwargs)

    def Lipschitz(self, mu=None):

        if isinstance(self.a, NesterovFunction) \
                and isinstance(self.b, NesterovFunction):

            return self.a.Lipschitz(mu=mu) + self.b.Lipschitz(mu=mu)

        elif isinstance(self.a, NesterovFunction):

            return self.a.Lipschitz(mu=mu) + self.b.Lipschitz()

        elif isinstance(self.b, NesterovFunction):

            return self.a.Lipschitz() + self.b.Lipschitz(mu=mu)

        else:

            return self.a.Lipschitz() + self.b.Lipschitz()

    def precompute(self, *args, **kwargs):

#        if hasattr(self.a, 'precompute'):
#
#            self.a.precompute(*args, **kwargs)
#
#        if hasattr(self.b, 'precompute'):
#
#            self.b.precompute(*args, **kwargs)
        raise ValueError("Has changed. Update!")

    def alpha(self, *args, **kwargs):

        if hasattr(self.a, 'alpha') and hasattr(self.b, 'alpha'):

            return self.a.alpha(*args, **kwargs) \
                    + self.b.alpha(*args, **kwargs)

        elif hasattr(self.a, 'alpha'):

            return self.a.alpha(*args, **kwargs)

        elif hasattr(self.b, 'alpha'):

            return self.b.alpha(*args, **kwargs)

        else:

            raise ValueError('At least one loss function must be Nesterov')

    def A(self):

        if hasattr(self.a, 'A') and hasattr(self.b, 'A'):

            return self.a.A() + self.b.A()

        elif hasattr(self.a, 'A'):

            return self.a.A()

        elif hasattr(self.b, 'A'):

            return self.b.A()

        else:

            raise ValueError('At least one loss function must be Nesterov')

    def At(self):

        if hasattr(self.a, 'At') and hasattr(self.b, 'At'):

            return self.a.At() + self.b.At()

        elif hasattr(self.a, 'At'):

            return self.a.At()

        elif hasattr(self.b, 'At'):

            return self.b.At()

        else:

            raise ValueError('At least one loss function must be Nesterov')

    def num_compacts(self):

        if hasattr(self.a, 'num_compacts') and hasattr(self.b, 'num_compacts'):

            return self.a.num_compacts() + self.b.num_compacts()

        elif hasattr(self.a, 'num_compacts'):

            return self.a.num_compacts()

        elif hasattr(self.b, 'num_compacts'):

            return self.b.num_compacts()

        else:

            raise ValueError('At least one loss function must be Nesterov')

    def num_groups(self):

        if hasattr(self.a, 'num_groups') and hasattr(self.b, 'num_groups'):

            return self.a.num_groups() + self.b.num_groups()

        elif hasattr(self.a, 'num_groups'):

            return self.a.num_groups()

        elif hasattr(self.b, 'num_groups'):

            return self.b.num_groups()

        else:

            raise ValueError('At least one loss function must be Nesterov')

    def project(self, alpha):

        if hasattr(self.a, 'project') and hasattr(self.b, 'project'):

            groups_a = self.a.num_groups()

            return self.a.project(alpha[:groups_a]) \
                    + self.b.project(alpha[groups_a:])

        elif hasattr(self.a, 'project'):

            return self.a.project(alpha)

        elif hasattr(self.b, 'project'):

            return self.b.project(alpha)

        else:

            raise ValueError('At least one loss function must be Nesterov')

    def _compute_alpha(self, beta, mu=None):

        if hasattr(self.a, '_compute_alpha') \
                and hasattr(self.b, '_compute_alpha'):

            return self.a._compute_alpha(beta, mu) \
                    + self.b._compute_alpha(beta, mu)

        elif hasattr(self.a, '_compute_alpha'):

            return self.a._compute_alpha(beta, mu)

        elif hasattr(self.b, '_compute_alpha'):

            return self.b._compute_alpha(beta, mu)

        else:

            raise ValueError('At least one loss function must be Nesterov')

    def _compute_grad(self, alpha):

        if hasattr(self.a, '_compute_grad') \
                and hasattr(self.b, '_compute_grad'):

            groups_a = self.a.num_groups()

            return self.a._compute_grad(alpha[:groups_a]) \
                    + self.b._compute_grad(alpha[groups_a:])

        elif hasattr(self.a, '_compute_grad'):

            return self.a._compute_grad(alpha)

        elif hasattr(self.b, '_compute_grad'):

            return self.b._compute_grad(alpha)

        else:

            raise ValueError('At least one loss function must be Nesterov')

    def get_mu(self):

        if hasattr(self.a, 'get_mu') and hasattr(self.b, 'get_mu'):

            # TODO: Problematic if the mus are different ...
            return min(self.a.get_mu(), self.b.get_mu())

        elif hasattr(self.b, 'get_mu'):

            return self.b.get_mu()

        else:

            return self.a.get_mu()

    def set_mu(self, mu):

        if hasattr(self.a, 'set_mu'):

            self.a.set_mu(mu)

        if hasattr(self.b, 'set_mu'):

            self.b.set_mu(mu)

    def set_data(self, *args, **kwargs):

        if hasattr(self.a, 'set_data'):

            self.a.set_data(*args, **kwargs)

        if hasattr(self.b, 'set_data'):

            self.b.set_data(*args, **kwargs)

    def get_data(self):

        data = tuple()
        if hasattr(self.a, 'get_data'):

            data += tuple(self.a.get_data())

        if hasattr(self.b, 'get_data'):

            data += tuple(self.b.get_data())

        # Remove doubles, preserve order
        ids = []
        unique = []
        for d in data:
            if id(d) not in ids:
                ids.append(id(d))
                unique.append(d)

        return tuple(unique)

    def free_data(self):

        if hasattr(self.a, 'free_data'):

            self.a.free_data()

        if hasattr(self.b, 'free_data'):

            self.b.free_data()


class ZeroErrorFunction(Differentiable,
                        LipschitzContinuous,
                        DataDependent,
                        ProximalOperator):

    def __init__(self, **kwargs):

        super(ZeroErrorFunction, self).__init__(**kwargs)

    def f(self, *args, **kwargs):
        return 0.0

    def grad(self, *args, **kwargs):
        return 0.0

    def Lipschitz(self, *args, **kwargs):
        return 0.0

    def set_data(self, *args, **kwargs):
        pass

    def get_data(self):
        return tuple()

    def free_data(self):
        pass

    def prox(self, beta, *args, **kwargs):
        return beta


class LinearRegressionError(LossFunction,
                            Convex,
                            Differentiable,
                            LipschitzContinuous,
                            DataDependent):
    """Loss function for linear regression. Represents the function:

        f(b) = (1 / 2) * ||X*b - y||²,

    where ||.||² is the L2 norm.
    """
    def __init__(self, **kwargs):

        super(LinearRegressionError, self).__init__(**kwargs)

    def set_data(self, X, y):

        self.X = X
        self.y = y

        self._lipschitz = None

    def get_data(self):

        return self.X, self.y

    def free_data(self):

        del self.X
        del self.y

        self._lipschitz = None

    def f(self, beta, **kwargs):

        return 0.5 * np.sum((self.y - np.dot(self.X, beta)) ** 2.0)

    def grad(self, beta, *args, **kwargs):

        return np.dot((np.dot(self.X, beta) - self.y).T, self.X).T

    def Lipschitz(self, *args, **kwargs):

        if self._lipschitz == None:  # Squared largest singular value
            v = algorithms.FastSVD(max_iter=100).run(self.X)
            us = np.dot(self.X, v)
            self._lipschitz = np.sum(us ** 2.0)

        return self._lipschitz


class RidgeRegression(LossFunction,
                      StronglyConvex,
                      Differentiable,
                      LipschitzContinuous,
                      DataDependent):
    """Loss function for ridge regression. Represents the function:

        f(b) = (1 / 2) * ||X.b - y||² + (l / 2) * ||b||²,

    where ||.||² is the L2 norm.
    """
    def __init__(self, l, **kwargs):

        super(RidgeRegression, self).__init__(**kwargs)

        self.l = l

    def set_data(self, X, y):

        self.X = X
        self.y = y

        self._l_max = None
        self._l_min = None

    def get_data(self):

        return self.X, self.y

    def free_data(self):

        del self.X
        del self.y

        self._l_max = None
        self._l_min = None

    def f(self, beta, **kwargs):

        return 0.5 * (np.sum((np.dot(self.X, beta) - self.y) ** 2.0) \
                        + self.l * np.sum(beta ** 2.0))

    def grad(self, beta, *args, **kwargs):

        return  np.dot((np.dot(self.X, beta) - self.y).T, self.X).T \
                + self.l * beta

    def Lipschitz(self, *args, **kwargs):

        if self._l_max == None:
            _, s, _ = np.linalg.svd(self.X, full_matrices=False)
            self._l_max = np.max(s) ** 2.0 + self.l
            self._l_min = np.min(s) ** 2.0 + self.l

        return self._l_max

    def lambda_min(self):

        if self._l_min == None:
            _, s, _ = np.linalg.svd(self.X, full_matrices=False)
            self._l_max = np.max(s) ** 2.0 + self.l
            self._l_min = np.min(s) ** 2.0 + self.l

        return self._l_min


class LinearLossFunction(LossFunction,
                         Convex,
                         Differentiable,
                         LipschitzContinuous,
                         DataDependent):
    """Loss function for a linear loss. Represents the function:

        f(b) = a'.b.
    """
    def __init__(self, a=None, **kwargs):

        super(LinearLossFunction, self).__init__(**kwargs)

        if a != None:
            self.a = a

    def set_data(self, a):

        self.a = a

    def get_data(self):

        return self.a

    def free_data(self):

        del self.a

    def f(self, beta, **kwargs):

        return np.dot(self.a.T, beta)[0, 0]

    def grad(self, *args, **kwargs):

        return self.a

    def Lipschitz(self, *args, **kwargs):

        return 0.0


class L1(ProximalOperator):
    """The proximal operator of the L1 loss function

        f(x) = l * ||x||_1,

    where ||x||_1 is the L1 loss function.
    """
    def __init__(self, l, **kwargs):

        super(L1, self).__init__(**kwargs)

        self.l = l

    def f(self, beta):

        return self.l * math.norm1(beta)

    def prox(self, x, factor=1.0, allow_empty=False):

        l = factor * self.l
        return (np.abs(x) > l) * (x - l * np.sign(x - l))


class L2(ProximalOperator):
    """The proximal operator of the L2 loss function

        f(x) = (l / 2.0) * ||x||²,

    where ||x||² is the L2 loss function.
    """
    def __init__(self, l, **kwargs):

        super(L2, self).__init__(**kwargs)

        self.l = l

    def f(self, x):

        return (self.l / 2.0) * np.sum(x ** 2.0)

    def prox(self, x, factor=1.0):

        l = factor * self.l
        return x / (1.0 + l)


class L1L2(ProximalOperator):
    """The proximal operator of the function

        f(x) = l * ||x||_1 + (k / 2) * ||x||²,

    where ||x||_1 is the L1 loss function and ||x||² is the L2 loss function.

    Parameters
    ----------
    l    : The L1 regularisation parameter.

    k    : The L2 regularisation parameter.
    """
    def __init__(self, l, k, **kwargs):

        super(L1L2, self).__init__(**kwargs)

        self.l = l
        self.k = k

    def f(self, x):
        return self.l * math.norm1(x) + (self.k / 2.0) * np.sum(x ** 2.0)

    def prox(self, x, factor=1.0, allow_empty=False):

        l = factor * self.l
        k = factor * self.k
        l1 = L1(l)

        return l1.prox(x, factor=1.0, allow_empty=allow_empty) \
                / (1.0 + k)


class ElasticNet(L1L2):
    """The proximal operator of the function

      f(x) = l * ||x||_1 + ((1 - l) / 2) * ||x||²,

    where ||x||_1 is the L1 loss function and ||x||² is the L2 loss function.

    Parameters
    ----------
    l : The L1 and L2 regularisation parameter. Must be in the interval [0, 1].
    """
    def __init__(self, l, **kwargs):

        l = max(0.0, min(l, 1.0))
        super(ElasticNet, self).__init__(l, 1.0 - l, **kwargs)


#class TotalVariation(NesterovFunction):
#
#    def __init__(self, gamma, mu=None, shape=None, A=None, mask=None,
#                 compress=True, **kwargs):
#        """Construct a TotalVariation loss function.
#
#        Parameters
#        ----------
#        gamma : The regularisation parameter for the TV penality.
#
#        mu : The Nesterov function regularisation parameter. Must be provided
#                unless you are using ContinuationRun.
#
#        shape : The shape of the unraveled data. This is either a integer, or a
#                tuple. If a tuple, it represents the shape of the image. If a
#                3D image, the shape must be a 3-tuple on the form (Z, Y, X). If
#                a 2D image, the shape must be a 2-tuple of the form (Y, X). If
#                a 1-tuple or an integer it represents the number of variables.
#
#                Equivalently, if the image is 2D, you may let the Z dimension
#                be 1, and if the "image" is 1D, you may let both the Y and Z
#                dimensions be 1.
#
#                Either shape or A must be given, but not both.
#
#        A : If the linear operators are known already, provide them to the
#                constructor to create a TV object reusing those matrices. In TV
#                it is assumed that A is a list of three elements, the matrices
#                Ax, Ay and Az. Either shape or A must be given, but not both.
#
#        mask : A 1-dimensional mask representing the image mask. Must be a list
#                or 1-dimensional array of booleans.
#
#        compress: The matrix A and the dual alpha is automatically pruned to
#                speed-up computations. This is not compatible with all smoothed
#                functions (really, only compatible with image-related
#                functions), and may therefore be turned off. Default is True,
#                set to False to keep all rows of A and alpha.
#        """
#        super(TotalVariation, self).__init__(gamma=gamma, mu=mu, **kwargs)
#
#        self.mask = mask
#        self.compress = compress
#
#        if shape != None:
#            if isinstance(shape, int):
#                shape = (1, 1, shape)
#            elif len(shape) == 1:
#                shape = (1, 1) + tuple(shape)
#            elif len(shape) == 2:
#                shape = (1,) + tuple(shape)
#
#        if A != None:
#            self._A = [A[0], A[1], A[2]]
#
#        else:
#            self._A = self.precompute(gamma, shape, mask=mask,
#                                      compress=compress)
#
#        self._At = None
#
#        # TODO: This is only true if zero-rows have been removed!
#        self._num_compacts = self._A[0].shape[0]
#
#        if mu == None:
#            # TODO: May be updated if we put gamma outside A!
#            mu = max(utils.TOLERANCE,
#                     2.0 * utils.TOLERANCE / self.num_compacts())
#            self.set_mu(mu)
#
#    def f(self, beta, mu=None, smooth=True):
#
#        if smooth:
#
#            return super(TotalVariation, self).f(beta, mu)
#
#        else:
#            if self.gamma < utils.TOLERANCE:
#                return 0.0
#
#            A = self.A()
#            sqsum = np.sum(np.sqrt(A[0].dot(beta) ** 2.0 + \
#                                   A[1].dot(beta) ** 2.0 + \
#                                   A[2].dot(beta) ** 2.0))
#
#            return sqsum  # Gamma is already incorporated in A
#
#    def project(self, alpha):
#
#        asx = alpha[0]
#        asy = alpha[1]
#        asz = alpha[2]
#        asnorm = asx ** 2.0 + asy ** 2.0 + asz ** 2.0
#        i = asnorm > 1.0
#
#        asnorm_i = asnorm[i] ** 0.5  # Square root is taken here. Faster.
#        asx[i] = np.divide(asx[i], asnorm_i)
#        asy[i] = np.divide(asy[i], asnorm_i)
#        asz[i] = np.divide(asz[i], asnorm_i)
#
#        return [asx, asy, asz]
#
#    def _compute_alpha(self, beta, mu=None):
#
#        if mu == None:
#            mu = self.get_mu()
#
#        # Compute a*
#        A = self.A()
#        alpha = [0] * len(A)
#        for i in xrange(len(A)):
#            alpha[i] = A[i].dot(beta) / mu
#
#        # Apply projection
#        alpha = self.project(alpha)
#
#        return alpha
#
#    @staticmethod
#    def precompute(gamma, shape, mask=None, compress=True):
#
#        def _find_mask_ind(mask, ind):
#
#            xshift = np.concatenate((mask[:, :, 1:], -np.ones((mask.shape[0],
#                                                              mask.shape[1],
#                                                              1))),
#                                    axis=2)
#            yshift = np.concatenate((mask[:, 1:, :], -np.ones((mask.shape[0],
#                                                              1,
#                                                              mask.shape[2]))),
#                                    axis=1)
#            zshift = np.concatenate((mask[1:, :, :], -np.ones((1,
#                                                              mask.shape[1],
#                                                              mask.shape[2]))),
#                                    axis=0)
#
#            xind = ind[(mask - xshift) > 0]
#            yind = ind[(mask - yshift) > 0]
#            zind = ind[(mask - zshift) > 0]
#
#            return xind.flatten().tolist(), \
#                   yind.flatten().tolist(), \
#                   zind.flatten().tolist()
#
#        Z = shape[0]
#        Y = shape[1]
#        X = shape[2]
#        p = X * Y * Z
#
#        smtype = 'csr'
#        Ax = gamma * (sparse.eye(p, p, 1, format=smtype) \
#                       - sparse.eye(p, p))
#        Ay = gamma * (sparse.eye(p, p, X, format=smtype) \
#                       - sparse.eye(p, p))
#        Az = gamma * (sparse.eye(p, p, X * Y, format=smtype) \
#                       - sparse.eye(p, p))
#
#        ind = np.reshape(xrange(p), (Z, Y, X))
#        if mask != None:
#            _mask = np.reshape(mask, (Z, Y, X))
#            xind, yind, zind = _find_mask_ind(_mask, ind)
#        else:
#            xind = ind[:, :, -1].flatten().tolist()
#            yind = ind[:, -1, :].flatten().tolist()
#            zind = ind[-1, :, :].flatten().tolist()
#
#        for i in xrange(len(xind)):
#            Ax.data[Ax.indptr[xind[i]]: \
#                    Ax.indptr[xind[i] + 1]] = 0
#        Ax.eliminate_zeros()
#
#        for i in xrange(len(yind)):
#            Ay.data[Ay.indptr[yind[i]]: \
#                    Ay.indptr[yind[i] + 1]] = 0
#        Ay.eliminate_zeros()
#
##        for i in xrange(len(zind)):
##            Az.data[Az.indptr[zind[i]]: \
##                    Az.indptr[zind[i] + 1]] = 0
#        Az.data[Az.indptr[zind[0]]: \
#                Az.indptr[zind[-1] + 1]] = 0
#        Az.eliminate_zeros()
#
#        # Remove rows corresponding to indices excluded in all dimensions
#        if compress:
#            toremove = list(set(xind).intersection(yind).intersection(zind))
#            toremove.sort()
#            # Remove from the end so that indices are not changed
#            toremove.reverse()
#            for i in toremove:
#                utils.delete_sparse_csr_row(Ax, i)
#                utils.delete_sparse_csr_row(Ay, i)
#                utils.delete_sparse_csr_row(Az, i)
#
#        # Remove columns of A corresponding to masked-out variables
#        if mask != None:
#            Ax = Ax.T.tocsr()
#            Ay = Ay.T.tocsr()
#            Az = Az.T.tocsr()
#            for i in reversed(xrange(p)):
#                # TODO: Mask should be boolean!
#                if mask[i] == 0:
#                    utils.delete_sparse_csr_row(Ax, i)
#                    utils.delete_sparse_csr_row(Ay, i)
#                    utils.delete_sparse_csr_row(Az, i)
#
#            Ax = Ax.T
#            Ay = Ay.T
#            Az = Az.T
#
#        return [Ax, Ay, Az]


class TotalVariation(NesterovFunction):

    def __init__(self, gamma, mu=None, shape=None, A=None, mask=None,
                 compress=True, **kwargs):
        """Construct a TotalVariation loss function.

        Parameters
        ----------
        gamma : The regularisation parameter for the TV penality.

        mu : The Nesterov function regularisation parameter. Must be provided
                unless you are using ContinuationRun.

        shape : The shape of the unraveled data. This is either a integer, or a
                tuple. If a tuple, it represents the shape of the image. If a
                3D image, the shape must be a 3-tuple on the form (Z, Y, X). If
                a 2D image, the shape must be a 2-tuple of the form (Y, X). If
                a 1-tuple or an integer it represents the number of variables.

                Equivalently, if the image is 2D, you may let the Z dimension
                be 1, and if the "image" is 1D, you may let both the Y and Z
                dimensions be 1.

                Either shape or A must be given, but not both.

        A : If the linear operator is known already, provide it to the
                constructor to create a TV object reusing thiat matrix. In TV
                it is assumed that A is a list of three elements, the matrices
                Ax, Ay and Az. Either A or shape must be given, but not both.

        mask : A 1-dimensional mask representing the image mask. Must be a list
                or 1-dimensional array of booleans.

        compress: The matrix A and the dual alpha is automatically pruned to
                speed-up computations. This is not compatible with all smoothed
                functions (really, only compatible with image-related
                functions), and may therefore be turned off. Default is True,
                set to False to keep all rows of A and alpha.
        """
        super(TotalVariation, self).__init__(gamma=gamma, mu=mu, **kwargs)

        self.mask = mask
        self.compress = compress

        if shape != None:
            if isinstance(shape, int):
                shape = (1, 1, shape)
            elif len(shape) == 1:
                shape = (1, 1) + tuple(shape)
            elif len(shape) == 2:
                shape = (1,) + tuple(shape)

        if A != None:
            self._A = [A[0], A[1], A[2]]

        else:
            self._A = self.precompute(shape, mask=mask, compress=compress)

        self._At = None

        if compress:
            self._num_compacts = self._A[0].shape[0]
        else:
            # Count the number of non.zero rows
            self._num_compacts = self._count_compacts(self._A)

        if mu == None:
            # The lower limit on mu.
            mu = max(utils.TOLERANCE,
                     2.0 * utils.TOLERANCE / self.num_compacts())
            self.set_mu(mu)

    def f(self, beta, mu=None, smooth=True):

        if smooth:

            return super(TotalVariation, self).f(beta, mu)

        else:
            if self.gamma < utils.TOLERANCE:
                return 0.0

            A = self.A()
            sqsum = np.sum(np.sqrt(A[0].dot(beta) ** 2.0 + \
                                   A[1].dot(beta) ** 2.0 + \
                                   A[2].dot(beta) ** 2.0))

            return self.gamma * sqsum

    def project(self, alpha):

        asx = alpha[0]
        asy = alpha[1]
        asz = alpha[2]
        asnorm = asx ** 2.0 + asy ** 2.0 + asz ** 2.0
        i = asnorm > 1.0

        asnorm_i = asnorm[i] ** 0.5  # Square root is taken here. Faster.
        asx[i] = np.divide(asx[i], asnorm_i)
        asy[i] = np.divide(asy[i], asnorm_i)
        asz[i] = np.divide(asz[i], asnorm_i)

        return [asx, asy, asz]

    def _compute_alpha(self, beta, mu=None):

        if mu == None:
            mu = self.get_mu()

        # Compute a*
        A = self.A()
        alpha = [0] * len(A)
        for i in xrange(len(A)):
            alpha[i] = A[i].dot(beta) / mu

        # Apply projection
        alpha = self.project(alpha)

        return alpha

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


class SmoothL1(NesterovFunction):

    def __init__(self, gamma, mu=None, num_variables=None, A=None, mask=None,
                 compress=True, **kwargs):
        """Construct an L1 loss function, smoothed using the Nesterov
        technique.

        Parameters
        ----------
        gamma : The regularisation parameter for the L1 penality.

        mu : The Nesterov function regularisation parameter. Must be provided
                unless you are using ContinuationRun.

        num_variables : The total number of variables (length of the resulting
                beta vector).

        A : If the linear operator is known already, provide them to the
                constructor to create a Nesterov object reusing those matrices.
                In smoothed L1 it is assumed that A is a list of just one
                element. Either num_variables or A must be given, but not both.

        mask : A 1-dimensional mask representing the 3D image mask. Must be a
                list or 1-dimensional array of booleans.

        compress: The matrix A and the dual alpha is automatically pruned to
                speed-up computations. This is not compatible with all smoothed
                functions (really, only compatible with image-related
                functions), and may therefore be turned off. Default is True,
                set to False to keep all rows of A and alpha.
        """
        super(SmoothL1, self).__init__(gamma=gamma, mu=mu, **kwargs)

        self.mask = mask
        self.compress = compress
        self.num_variables = num_variables
        self.l = self.gamma

        if A != None:

            self._A = [A[0], A[1], A[2]]

        else:
            self._A = self.precompute(num_variables, mask, compress)

        self._At = None

        if compress:
            self._num_compacts = self._A[0].shape[0]
        else:
            # Count the number of non.zero rows
            self._num_compacts = self._count_compacts(self._A)

        if mu == None:
            # The lower limit on mu.
            mu = max(utils.TOLERANCE,
                     2.0 * utils.TOLERANCE / self.num_compacts())
            self.set_mu(mu)

    def f(self, beta, mu=None, smooth=True):

        if smooth:

            return super(SmoothL1, self).f(beta, mu)

        else:

            if self.gamma < utils.TOLERANCE:
                return 0.0

            return self.gamma * math.norm1(beta)

    def Lipschitz(self, mu=None):

        if self.gamma < utils.TOLERANCE:
            return 0.0

        self._lambda_max = 1.0
#        if self._lambda_max == None:
#            A = sparse.vstack(self.A())
#            v = algorithms.SparseSVD(max_iter=100).run(A)
#            us = A.dot(v)
#            self.lambda_max = np.sum(us ** 2.0)
#
#            # TODO: May change if gamma is put outside of A
#            self._lambda_max = self.gamma ** 2.0
#        print self.lambda_max, " == ", lambda_max, "?"

        if mu != None:
            return (self.gamma ** 2.0) * self._lambda_max / mu
        else:
            return (self.gamma ** 2.0) * self._lambda_max / self.get_mu()

    def project(self, alpha):

        a = alpha[0]
        asnorm = np.abs(a)
        i = asnorm > 1.0
        asnorm_i = asnorm[i]

        a[i] = np.divide(a[i], asnorm_i)

        return [a]

    def _compute_alpha(self, beta, mu=None):

        if mu == None:
            mu = self.get_mu()

        # Compute a*
        alpha = [0]
#        A = self.A()
#        alpha[0] = A[0].dot(beta) / mu
        alpha[0] = (1.0 / mu) * beta

        # Apply projection
        alpha = self.project(alpha)

        return alpha

    def _compute_grad(self, alpha):

        # Compute a*
#        grad = self._At[0].dot(alpha[0])
        grad = alpha[0]

        return grad

    @staticmethod
    def precompute(num_variables, mask, compress):

        A = sparse.eye(num_variables, num_variables, format='csr')

        # Remove rows corresponding to masked-out variables
        if compress and mask != None:
            for i in reversed(xrange(num_variables)):
                # TODO: Mask should be boolean!
                if mask[i] == 0:
                    utils.delete_sparse_csr_row(A, i)

        # Remove columns of A corresponding to masked-out variables
        if mask != None:
            A = A.T.tocsr()
            for i in reversed(xrange(num_variables)):
                # TODO: Mask should be boolean!
                if mask[i] == 0:
                    utils.delete_sparse_csr_row(A, i)

            A = A.T

        return [A]