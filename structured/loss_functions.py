# -*- coding: utf-8 -*-
"""
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

           'ProximalOperator',
           'ZeroErrorFunction',
           'L1', 'L2', 'ElasticNet',

           'NesterovFunction',
           'TotalVariation', 'GroupLassoOverlap',

           'CombinedLossFunction',
           'CombinedNesterovLossFunction']

import abc
import numpy as np
import scipy.sparse as sparse
import copy

import algorithms
#from utils import norm, norm1, TOLERANCE, delete_sparse_csr_row
import utils


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
    """A loss function approximated using the Nesterov technique.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, gamma=1.0, mu=None, **kwargs):
        """Construct a Nesterov loss function.

        Parameters
        ----------
        gamma : The regularisation parameter for the nesterov penality.

        mu : The Nesterov function regularisation parameter. Must be provided
                unless you are using ContinuationRun.
        """
        super(NesterovFunction, self).__init__(**kwargs)

        self.gamma = float(gamma)

        if mu != None:
            self.set_mu(mu)

        self._A = None  # The linear operator
        self._At = None  # The linear operator transposed
        self._alpha = None  # The dual variable
        self._grad = None  # The function's gradient
        self.lambda_max = None  # The largest eigenvalue of A'.A

    def Lipschitz(self, mu=None):
        """The Lipschitz constant of the gradient.

        In general the Lipschitz constant is the largest eigenvalue of A'.A
        divided by mu, but should be specialised in subclasses if faster
        approaches exist.
        """
        if self.gamma < utils.TOLERANCE:
            return 0.0

        if self.lambda_max == None:
            A = sparse.vstack(self.A())
            v = algorithms.SparseSVD(max_iter=100).run(A)
            us = A.dot(v)
            self.lambda_max = np.sum(us ** 2.0)

        if mu != None:
            return self.lambda_max / mu
        else:
            return self.lambda_max / self.get_mu()

    def A(self):
        """Returns a list of the A blocks that constitute the A matrix, the
        linear operator.
        """
        return self._A

    def At(self):
        """Returns a list of the transposed A blocks that constitute the A
        matrix, the linear operator.
        """
        return self._At

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
        return len(self._A)

    @abc.abstractmethod
    def num_compacts(self):
        """The smoothness gap between f and f_mu is mu * num_compacts().

        Note that the constant D = num_compacts() / 2.
        """
        raise NotImplementedError('Abstract method "num_compacts" must be ' \
                                  'specialised!')

    @abc.abstractmethod
    def projection(self, *alpha):

        raise NotImplementedError('Abstract method "projection" must be ' \
                                  'specialised!')

    @abc.abstractmethod
    def alpha(self, beta=None, mu=None):

        raise NotImplementedError('Abstract method "alpha" must be ' \
                                  'specialised!')

    @abc.abstractmethod
    def precompute(self, *args, **kwargs):

        raise NotImplementedError('Abstract method "precompute" must be ' \
                                  'specialised!')


class CombinedLossFunction(LossFunction, DataDependent):

    def __init__(self, a, b, **kwargs):

        super(CombinedLossFunction, self).__init__(**kwargs)

        self.a = a
        self.b = b

    def set_data(self, *args, **kwargs):

        if hasattr(self.a, 'set_data'):
            self.a.set_data(*args, **kwargs)
        if hasattr(self.b, 'set_data'):
            self.b.set_data(*args, **kwargs)

    def f(self, *args, **kwargs):
        return self.a.f(*args, **kwargs) + self.b.f(*args, **kwargs)


class CombinedNesterovLossFunction(CombinedLossFunction, NesterovFunction):
    """A loss function contructed as the sum of two loss functions, where
    at least one of them is a Nesterov function, i.e. a loss function computed
    using the Nestrov technique.

    This function is of the form

        g = g1 + g2,

    where at least one of g1 or g2 is a Nesterov functions.
    """

    def __init__(self, a, b):

        super(CombinedNesterovLossFunction, self).__init__(a=a, b=b)

    def grad(self, *args, **kwargs):

        return self.a.grad(*args, **kwargs) + self.b.grad(*args, **kwargs)

    def Lipschitz(self, mu=None):

        if isinstance(self.a, LipschitzContinuous) \
                and isinstance(self.b, LipschitzContinuous):
            return self.a.Lipschitz(mu=mu) + self.b.Lipschitz(mu=mu)
        elif isinstance(self.a, LipschitzContinuous):
            return self.a.Lipschitz(mu=mu) + self.b.Lipschitz()
        elif isinstance(self.b, LipschitzContinuous):
            return self.a.Lipschitz() + self.b.Lipschitz(mu=mu)
        else:
            raise ValueError('At least one loss function must be ' \
                             'LipschitzContinuous')

    def precompute(self, *args, **kwargs):

        if hasattr(self.a, 'precompute'):
            self.a.precompute(*args, **kwargs)
        if hasattr(self.b, 'precompute'):
            self.b.precompute(*args, **kwargs)

    def alpha(self, beta=None, mu=None):

        if hasattr(self.a, 'alpha') and hasattr(self.b, 'alpha'):
            return self.a.alpha(beta) + self.b.alpha(beta)  # Merge
        elif hasattr(self.a, 'alpha'):
            return self.a.alpha(beta)
        elif hasattr(self.b, 'alpha'):
            return self.b.alpha(beta)
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

    def projection(self, *alpha):

        if hasattr(self.a, 'projection') and hasattr(self.b, 'projection'):

            groups_a = self.a.num_groups()

            return self.a.projection(*alpha[0:groups_a]) \
                 + self.b.projection(*alpha[groups_a:])

        elif hasattr(self.a, 'projection'):
            return self.a.projection(*alpha)
        elif hasattr(self.b, 'projection'):
            return self.b.projection(*alpha)
        else:
            raise ValueError('At least one loss function must be Nesterov')

    def get_mu(self):

        if hasattr(self.a, 'get_mu') and hasattr(self.b, 'get_mu'):
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


class ZeroErrorFunction(Differentiable, LipschitzContinuous, DataDependent,
                        ProximalOperator):

    def __init__(self, **kwargs):

        super(ZeroErrorFunction, self).__init__(**kwargs)

    def f(self, *args, **kwargs):
        return 0

    def grad(self, *args, **kwargs):
        return 0

    def Lipschitz(self, *args, **kwargs):
        return 0

    def set_data(self, *args, **kwargs):
        pass

    def prox(self, beta, *args, **kwargs):
        return beta


class LinearRegressionError(LossFunction,
                            Convex,
                            Differentiable,
                            LipschitzContinuous,
                            DataDependent):
    """Loss function for linear regression. Represents the function:

        f(b) = (1.0 / 2.0) * norm(y - X*b)²
    """
    def __init__(self, **kwargs):

        super(LinearRegressionError, self).__init__(**kwargs)

    def set_data(self, X, y):

        self.X = X
        self.y = y

        self.lipschitz = None

    def f(self, beta, **kwargs):

#        return norm(self.y - np.dot(self.X, beta)) ** 2
#        return np.sum((self.y - np.dot(self.X, beta)) ** 2.0)
        return 0.5 * np.sum((self.y - np.dot(self.X, beta)) ** 2.0)

    def grad(self, beta, *args, **kwargs):

#        return 2.0 * np.dot(self.X.T, np.dot(self.X, beta) - self.y)
#        return np.dot(self.X.T, np.dot(self.X, beta) - self.y)
        return np.dot((np.dot(self.X, beta) - self.y).T, self.X).T

    def Lipschitz(self, *args, **kwargs):

        if self.lipschitz == None:
            v = algorithms.FastSVD(max_iter=100).run(self.X)
            us = np.dot(self.X, v)
#            self.lipschitz = 2.0 * np.sum(us ** 2.0)
            self.lipschitz = np.sum(us ** 2.0)

        return self.lipschitz


class LogisticRegressionError(LossFunction,
                              Convex,
                              Differentiable,
                              LipschitzContinuous,
                              DataDependent):

    def __init__(self, **kwargs):

        super(LogisticRegressionError, self).__init__(**kwargs)

    def set_data(self, X, y):

        self.X = X
        self.y = y

        self.lipschitz = None

    def f(self, beta, *args, **kwargs):

        logit = np.dot(self.X, beta)
        expt = np.exp(logit)
        return -np.sum(np.multiply(self.y, logit) + np.log(1 + expt))

    def grad(self, beta, **kwargs):

        logit = np.dot(self.X, beta)
        expt = np.exp(logit)
        pix = np.divide(expt, expt + 1)
        return -np.dot(self.X.T, self.y - pix)

#    def hessian(self, beta, **kwargs):
#        logit = np.dot(self.X, beta)
#        expt = np.exp(logit)
#        pix = np.divide(expt, expt + 1)
#        pixpix = np.multiply(pix, 1 - pix)
#
#        V = np.diag(pixpix.flatten())
#        XVX = np.dot(np.dot(self.X.T, V), self.X)
#
#        return XVX

    def Lipschitz(self, *args, **kwargs):
        if self.lipschitz == None:
            # pi(x) * (1 - pi(x)) <= 0.25 = 0.5 * 0.5
            V = 0.5 * np.eye(self.X.shape[0])
            VX = np.dot(V, self.X)
            svd = algorithms.FastSVD(max_iter=100)
            t = np.dot(VX, svd.run(VX))
            self.lipschitz = np.sum(t ** 2.0)
#            _, s, _ = np.linalg.svd(VX, full_matrices=False)
#            self.t = np.max(s) ** 2.0

        return self.lipschitz


class RidgeRegression(LossFunction,
                      StronglyConvex,
                      Differentiable,
                      LipschitzContinuous,
                      DataDependent):
    """Loss function for ridge regression. Represents the function:

        #f(b) = norm(y - X*b)² + lambda*norm(b)²
        f(b) = (1.0 / 2.0) * norm(y - X*b)² + (lambda / 2.0) * norm(b)²
    """
    def __init__(self, l, **kwargs):

        super(RidgeRegression, self).__init__(**kwargs)

        self.l = l

    def set_data(self, X, y):

        self.X = X
        self.y = y
        self.Xty = np.dot(X.T, y)

        self.l_max = None
        self.l_min = None

    def f(self, beta, **kwargs):

#        return (np.sum((self.y - np.dot(self.X, beta)) ** 2.0) \
#                + self.l * np.sum(beta ** 2.0))
        return 0.5 * (np.sum((self.y - np.dot(self.X, beta)) ** 2.0) \
                + self.l * np.sum(beta ** 2.0))

    def grad(self, beta, *args, **kwargs):

#        return 2.0 * ((np.dot(self.X.T, np.dot(self.X, beta)) - self.Xty) \
#                + self.l * beta)
        return (np.dot(self.X.T, np.dot(self.X, beta)) - self.Xty) \
                + self.l * beta

    def Lipschitz(self, *args, **kwargs):

        if self.l_max == None:
            _, s, _ = np.linalg.svd(self.X, full_matrices=False)
            self.l_max = np.max(s) ** 2.0 + self.l
            self.l_min = np.min(s) ** 2.0 + self.l

#        return 2.0 * self.l_max
        return self.l_max

    def lambda_min(self):

        if self.l_min == None:
            _, s, _ = np.linalg.svd(self.X, full_matrices=False)
            self.l_max = np.max(s) ** 2.0 + self.l
            self.l_min = np.min(s) ** 2.0 + self.l

#        return 2.0 * self.l_min
        return self.l_min


class L1(ProximalOperator):

    def __init__(self, l, **kwargs):

        super(L1, self).__init__(**kwargs)

        self.l = l

    def f(self, beta):

        return self.l * utils.norm1(beta)

    def prox(self, x, factor=1.0, allow_empty=False):

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


class L2(ProximalOperator):
    """The proximal operator of the function

        f(x) = (lambda / 2.0) * norm(x) ** 2.0.
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

        f(x) = lambda * norm1(x) + (kappa / 2.0) * norm(x) ** 2.

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
        return self.l * utils.norm1(x) + (self.k / 2.0) * np.sum(x ** 2.0)

    def prox(self, x, factor=1.0, allow_empty=False):

        l = factor * self.l
        k = factor * self.k
        l1 = L1(l)

        return l1.prox(x, factor=1.0, allow_empty=allow_empty) \
                / (1.0 + k)


class ElasticNet(L1L2):
    """The proximal operator of the function

      f(x) = lambda * norm1(x) + ((1.0 - lambda) / 2.0) * norm(x) ** 2.

    Parameters
    ----------
    l : The regularisation parameter. Must be in the interval [0, 1].
    """
    def __init__(self, l, **kwargs):

        l = max(0.0, min(l, 1.0))
        super(ElasticNet, self).__init__(l, 1.0 - l, **kwargs)


class NesterovDualFunction(NesterovFunction):
    """Duplicates a Nesterov loss function, but holds the alpha of this
    function constant.

    I.e., this class can be used to find a beta that corresponds to a
    particular alpha.

    This class also holds the gradient constant, i.e. _grad and grad() are
    fixed, just like _alpha and alpha(). This means it assumes a constant A
    matrix.
    """
    def __init__(self, function, **kwargs):

        super(NesterovDualFunction, self).__init__(gamma=function.gamma,
                                                   mu=function.get_mu(),
                                                   **kwargs)

        # TODO: Since __getattr__ and __setattr__ handles all the fields of the
        #       Nesterov function, we do not need to keep references here as
        #       well!

        # Required fields
        self.lambda_max = function.Lipschitz(1.0)
        self._A = function.A()
        self._At = function.At()
        self.set_mu(function.get_mu())
        self._num_compacts = function.num_compacts()
        self._alpha = function.alpha()
        self._grad = function.grad()
        # Copies of the "true", constant, internal alpha and grad!
        self.__alpha = copy.deepcopy(self._alpha)
        self.__grad = copy.deepcopy(self._grad)

        # Optional fields
        if hasattr(function, '_buff'):
            self._buff = function._buff

        # Methods defined in the base class NesterovFunction
        self.Lipschitz = function.Lipschitz
        self.A = function.A
        self.At = function.At
        self.get_mu = function.get_mu
        self.set_mu = function.set_mu
        self.num_groups = function.num_groups

        # Methods defined in the subclasses
        self.f = function.f
        self.grad = function.grad

        # Abstract methods defined in the subclasses
        self.num_compacts = function.num_compacts
        self.projection = function.projection
        self.precompute = function.precompute

        # The loss function we duplicate
        self.function = function

    def f(self, *args, **kwargs):
        pass

    def num_compacts(self, *args, **kwargs):
        pass

    def precompute(self, *args, **kwargs):
        pass

    def projection(self, *args, **kwargs):
        pass

    def _compute_alpha_grad(self, *args, **kwargs):
        """Used by most (all?) Nesterov loss functions to compute a new value
        of _alpha and _grad. Since this class controlls those, and they are
        constant, this function should not do anything.
        """
        pass

#    def __setattr__(self, name, value):
#
#        if name != '_alpha' and name != 'alpha' \
#                and name != '_grad' and name != 'grad':
#
#            super(NesterovDualFunction, self).__setattr__(name, value)
##            self.__dict__[name] = value  # Set attribute of this class
##            self.function.__setattr__(name, value)

#    def __delattr__(self, name):
#
#        if name != '_alpha' and name != 'alpha' \
#                and name != '_grad' and name != 'grad':
#            super(NesterovDualFunction, self).__delattr__(name)
##            self.function.__delattr__(name)

#    def __getattr__(self, name):
#
#        if name == '_alpha':
#            return self.__alpha
#
#        if name == '_grad':
#            return self.__grad
#
#        return super(NesterovDualFunction, self).__getattr__(name)

    def alpha(self, *args, **kwargs):

        return self.__alpha

    def grad(self, *args, **kwargs):

        return self.__grad


class TotalVariation(NesterovFunction):

    def __init__(self, gamma, shape=None, mu=None, mask=None,
                 compress=True, A=None, **kwargs):
        """Construct a TotalVariation loss function.

        Parameters
        ----------
        gamma : The regularisation parameter for the TV penality.

        shape : The shape of the 3D image. Must be a 3-tuple. If the image is
                2D, let the Z dimension be 1, and if the "image" is 1D, let the
                Y and Z dimensions be 1. The tuple must be on the form
                (Z, Y, X). Either shape or A must be given, but not both.

        A : If the linear operators are known already, provide them to the
                constructor to create a TV object reusing those matrices. In TV
                it is assumed that A is a list of three elements, the matrices
                Ax, Ay and Az. Either shape or A must be given, but not both.

        mu : The Nesterov function regularisation parameter. Must be provided
                unless you are using ContinuationRun.

        mask : A 1-dimensional mask representing the 3D image mask.

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
            self.precompute(shape)
        else:
            self._A = [A[0], A[1], A[2]]
            self._At = [A[0].T, A[1].T, A[2].T]
            self._alpha = [0.0, 0.0, 0.0]

            # TODO: This is only true if zero-rows have been removed!
            self._num_compacts = A[0].shape[0]

            self._buff = np.zeros((A[0].shape[1], 1))

        if mu == None:
            mu = max(utils.TOLERANCE,
                     2.0 * utils.TOLERANCE / self.num_compacts())
            self.set_mu(mu)

    def f(self, beta, mu=None, smooth=True):

        if self.gamma < utils.TOLERANCE:
            return 0.0

        if smooth:
            self._compute_alpha_grad(beta, mu)

            alpha = self.alpha()
            alpha_sqsum = 0.0
            for a in alpha:
                alpha_sqsum += np.sum(a ** 2.0)

            if mu == None:
                mu = self.get_mu()

            return np.dot(beta.T, self.grad())[0, 0] - (mu / 2.0) * alpha_sqsum

        else:
#            sqsum = 0.0
#            for Ad in self._A:
#                sqsum += Ad.dot(beta) ** 2.0
#            sqsum = np.sqrt(sqsum)
            sqsum = np.sum(np.sqrt(self._A[0].dot(beta) ** 2.0 + \
                                   self._A[1].dot(beta) ** 2.0 + \
                                   self._A[2].dot(beta) ** 2.0))

            return sqsum  # Gamma is already incorporated in A

    def grad(self, beta=None, mu=None):

        if self.gamma < utils.TOLERANCE:
            return np.zeros(beta.shape)

        if beta != None:
            self._compute_alpha_grad(beta, mu)

        return self._grad

    def alpha(self, beta=None, mu=None):

        if self.gamma < utils.TOLERANCE:
            return [0.0, 0.0, 0.0]

        if beta != None:
            self._compute_alpha_grad(beta, mu)

        return self._alpha

    def num_compacts(self):

#        return np.prod(self.shape)  # Number of variables
        return self._num_compacts

    def projection(self, *alpha):

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

    def _compute_alpha_grad(self, beta, mu=None):

        if mu == None:
            mu = self.get_mu()

        # Compute a* for each dimension
#        self._alpha[0] = self._A[0].dot(beta) / mu
#        self._alpha[1] = self._A[1].dot(beta) / mu
#        self._alpha[2] = self._A[2].dot(beta) / mu
        for i in xrange(len(self._A)):
            self._alpha[i] = self._A[i].dot(beta) / mu

        # Apply projection
        self._alpha = self.projection(*self._alpha)

#        self._grad = np.add(np.add(self._At[0].dot(self._alpha[0]),
#                                    self._At[1].dot(self._alpha[1]), self._buff),
#                                    self._At[2].dot(self._alpha[2]), self._buff)
        self._grad = self._At[0].dot(self._alpha[0])
        for i in xrange(1, len(self._alpha)):
            self._grad += self._At[i].dot(self._alpha[i])

    def _find_mask_ind(self, mask, ind):

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

    def precompute(self, shape):

        Z = shape[0]
        Y = shape[1]
        X = shape[2]
        p = X * Y * Z

        smtype = 'csr'
        Ax = self.gamma * (sparse.eye(p, p, 1, format=smtype) \
                            - sparse.eye(p, p))
        Ay = self.gamma * (sparse.eye(p, p, X, format=smtype) \
                            - sparse.eye(p, p))
        Az = self.gamma * (sparse.eye(p, p, X * Y, format=smtype) \
                            - sparse.eye(p, p))

        ind = np.reshape(xrange(p), (Z, Y, X))
        if self.mask != None:
            mask = np.reshape(self.mask, (Z, Y, X))
            xind, yind, zind = self._find_mask_ind(mask, ind)
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
        if self.compress:
            toremove = list(set(xind).intersection(yind).intersection(zind))
            toremove.sort()
            # Remove from the end so that indices are not changed
            toremove.reverse()
            for i in toremove:
                utils.delete_sparse_csr_row(Ax, i)
                utils.delete_sparse_csr_row(Ay, i)
                utils.delete_sparse_csr_row(Az, i)

        # Remove columns of A corresponding to masked-out variables
        if self.mask != None:
            Axt = Ax.T.tocsr()
            Ayt = Ay.T.tocsr()
            Azt = Az.T.tocsr()
            for i in reversed(xrange(p)):
                if self.mask[i] == 0:
                    utils.delete_sparse_csr_row(Axt, i)
                    utils.delete_sparse_csr_row(Ayt, i)
                    utils.delete_sparse_csr_row(Azt, i)

            Ax = Axt.T
            Ay = Ayt.T
            Az = Azt.T
        else:
            Axt = Ax.T
            Ayt = Ay.T
            Azt = Az.T

        self._A = [Ax, Ay, Az]
        self._At = [Axt, Ayt, Azt]
        self._alpha = [0.0, 0.0, 0.0]

        # TODO: This is only true if zero-rows have been removed!
        self._num_compacts = Ax.shape[0]

        self._buff = np.zeros((Ax.shape[1], 1))


class SmoothL1(NesterovFunction):

    def __init__(self, gamma, num_variables, mu=None, mask=None, **kwargs):
        """Construct an L1 loss function, smoothed using the Nesterov
        technique.

        Parameters
        ----------
        gamma : The regularisation parameter for the L1 penality.

        num_variables : The total number of variables.

        mu : The Nesterov function regularisation parameter. Must be provided
                unless you are using ContinuationRun.

        mask : A 1-dimensional mask representing the 3D image mask. Must be a
                list of 1s and 0s.
        """
        super(SmoothL1, self).__init__(gamma=gamma, mu=mu, **kwargs)

        self.num_variables = num_variables
        self.mask = mask

        self.precompute()

        if mu == None:
            mu = max(utils.TOLERANCE,
                     2.0 * utils.TOLERANCE / self.num_compacts())
            self.set_mu(mu)

    def f(self, beta, mu=None, smooth=True):

        if self.gamma < utils.TOLERANCE:
            return 0.0

        if smooth:
            self._compute_alpha_grad(beta, mu)

            alpha = self.alpha()

            alpha_sqsum = 0.0
            for a in alpha:
                alpha_sqsum += np.sum(a ** 2.0)

            if mu == None:
                mu = self.get_mu()

            return np.dot(beta.T, self.grad())[0, 0] - (mu / 2.0) * alpha_sqsum

        else:
            return self.gamma * utils.norm1(beta)

    def grad(self, beta=None, mu=None):

        if self.gamma < utils.TOLERANCE:
            return np.zeros(beta.shape)

        if beta != None:
            self._compute_alpha_grad(beta, mu)

        return self._grad

    def Lipschitz(self, mu=None):

        if self.gamma < utils.TOLERANCE:
            return 0.0

        if self.lambda_max == None:
#            A = sparse.vstack(self.A())
#            v = algorithms.SparseSVD(max_iter=100).run(A)
#            us = A.dot(v)
#            self.lambda_max = np.sum(us ** 2.0)

            self.lambda_max = self.gamma ** 2.0
#        print self.lambda_max, " == ", lambda_max, "?"
#        print "L1!!!"

        if mu != None:
            return self.lambda_max / mu
        else:
            return self.lambda_max / self.get_mu()

    def alpha(self, beta=None, mu=None):

        if self.gamma < utils.TOLERANCE:
            return [0.0]

        if beta != None:
            self._compute_alpha_grad(beta, mu)

        return self._alpha

    def num_compacts(self):

        return self.num_variables

    def projection(self, *alpha):

        a = alpha[0]
        anorm = np.abs(a)
        i = anorm > 1.0
        asnorm_i = anorm[i]

        a[i] = np.divide(a[i], asnorm_i)

        return [a]

    def _compute_alpha_grad(self, beta, mu=None):

        if mu == None:
            mu = self.get_mu()

        # Compute a*
#        self._alpha[0] = self._A[0].dot(beta) / mu
        self._alpha[0] = (self.gamma / mu) * beta

        # Apply projection
        self._alpha = self.projection(*self._alpha)

#        self._grad = self._At[0].dot(self._alpha[0])
        self._grad = self.gamma * self._alpha[0]

    def precompute(self):

        A = self.gamma * sparse.eye(self.num_variables, self.num_variables,
                                    format='csr')

        # Find indices in the mask to remove
        if self.mask != None:
            At = A.T.tocsr()
            for i in reversed(xrange(self.num_variables)):
                if self.mask[i] == 0:
                    utils.delete_sparse_csr_row(At, i)

            A = At.T
        else:
            At = A.T

        self._A = [A]
        self._At = [At]
        self._alpha = [0.0]


class GroupLassoOverlap(NesterovFunction):

    def __init__(self, gamma, num_variables, groups, mu=None, weights=None,
                 **kwargs):
        """
        Parameters:
        ----------
        gamma : The GL regularisation parameter.

        num_variables : The total number of variables.

        groups : A list of lists, with the outer list being the groups and the
                inner lists the variables in the groups. E.g. [[1,2],[2,3]]
                contains two groups ([1,2] and [2,3]) with variable 1 and 2 in
                the first group and variables 2 and 3 in the second group.

        mu : The Nesterov function regularisation parameter. Must be provided
                unless you are using ContinuationRun.

        weights : Weights put on the groups. Default is weight 1 for each
                group.
        """
        super(GroupLassoOverlap, self).__init__(gamma=gamma, mu=mu, **kwargs)

        self.num_variables = num_variables
        self.groups = groups

        if weights == None:
            self.weights = [1.0] * len(groups)
        else:
            self.weights = weights

        self.precompute()

        if mu == None:
            mu = max(utils.TOLERANCE,
                     2.0 * utils.TOLERANCE / self.num_compacts())
            self.set_mu(mu)

    def f(self, beta, mu=None, smooth=True):

        if self.gamma < utils.TOLERANCE:
            return 0.0

        if smooth:
            self._compute_alpha_grad(beta, mu)

            alpha = self.alpha()
            alpha_sqsum = 0.0
            for a in alpha:
                alpha_sqsum += np.sum(a ** 2.0)

            if mu == None:
                mu = self.get_mu()

            return np.dot(beta.T, self.grad())[0, 0] - (mu / 2.0) * alpha_sqsum

        else:
            sqsum = 0.0
            for Ag in self._A:
                sqsum += utils.norm(Ag.dot(beta))  # A.dot(beta) ** 2.0

            return sqsum  # Gamma is already incorporated in A

    def grad(self, beta=None, mu=None):

        if self.gamma < utils.TOLERANCE:
            return np.zeros(beta.shape)

        if beta != None:
            self._compute_alpha_grad(beta, mu)

        return self._grad

    def Lipschitz(self, mu=None):

        if self.gamma < utils.TOLERANCE:
            return 0

#        if self.lambda_max == None:
#            A = sparse.vstack(self.A())
#            v = algorithms.SparseSVD(max_iter=100).run(A)
#            us = A.dot(v)
#            self.lambda_max = np.sum(us ** 2.0)

#        test = self.lambda_max / self.get_mu()

#        print "lambda_max : ", test
        if mu != None:
#            print "max col sum: ", (self.max_col_norm / mu)
            return self.max_col_norm / mu
        else:
#            print "max col sum: ", (self.max_col_norm / self.get_mu())
            return self.max_col_norm / self.get_mu()

    def alpha(self, beta=None, mu=None):

        if self.gamma < utils.TOLERANCE:
            return tuple([0.0] * len(self._alpha))

        if beta != None:
            self._compute_alpha_grad(beta, mu)

        return self._alpha

    def num_compacts(self):

        return self.num_groups()

    def projection(self, *alpha):

        alpha = list(alpha)

        for i in xrange(len(alpha)):
            astar = alpha[i]
            normas = np.sqrt(np.sum(astar ** 2.0))
            if normas > 1.0:
                astar /= normas
            alpha[i] = astar

        return alpha

    def _compute_alpha_grad(self, beta, mu=None):

        if mu == None:
            mu = self.get_mu()

        # Compute a* for each dimension
        self._grad = 0
        for g in xrange(len(self._A)):
            astar = self._A[g].dot(beta) / mu
            astar = self.projection(astar)[0]

            self._alpha[g] = astar

            self._grad += self._At[g].dot(astar)

    def precompute(self):

        self._A = list()
        self._At = list()

        powers = np.zeros(self.num_variables)
        for g in xrange(len(self.groups)):
            Gi = self.groups[g]
            lenGi = len(Gi)
            Ag = sparse.lil_matrix((lenGi, self.num_variables))
            for i in xrange(lenGi):
                w = self.gamma * self.weights[g]
                Ag[i, Gi[i]] = w
                powers[Gi[i]] += w ** 2.0

            # Matrix operations are a lot faster when the sparse matrix is csr
            self._A.append(Ag.tocsr())
            self._At.append(self._A[-1].T)

        self.max_col_norm = np.max(powers)
        self._alpha = [0] * len(self.groups)