# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 10:54:29 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD Style.

"""

__all__ = ['LossFunction', 'LipschitzContinuous', 'Differentiable',

           'ConvexLossFunction',
           'LinearRegressionError', 'LogisticRegressionError',

           'StronglyConvexLossFunction',
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

import algorithms
from utils import norm, norm1, TOLERANCE, delete_sparse_csr_row


class LossFunction(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):

        super(LossFunction, self).__init__(**kwargs)

    @abc.abstractmethod
    def f(self, *args, **kwargs):
        raise NotImplementedError('Abstract method "f" must be specialised!')

    def set_data(self, *args, **kwargs):
        pass


class LipschitzContinuous(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):

        super(LipschitzContinuous, self).__init__(**kwargs)

    @abc.abstractmethod
    def Lipschitz(self, **kwargs):
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


class ConvexLossFunction(LossFunction):

    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):

        super(ConvexLossFunction, self).__init__(**kwargs)


class StronglyConvexLossFunction(LossFunction):

    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):

        super(StronglyConvexLossFunction, self).__init__(**kwargs)

    @abc.abstractmethod
    def lambda_min(self):
        raise NotImplementedError('Abstract method "lambda_min" must be ' \
                                  'specialised!')


class ProximalOperator(ConvexLossFunction):

    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):

        super(ProximalOperator, self).__init__(**kwargs)

    @abc.abstractmethod
    def prox(self, x):
        raise NotImplementedError('Abstract method "prox" must be ' \
                                  'specialised!')


class NesterovFunction(Differentiable, LipschitzContinuous):
    """A loss function approximated using the Nesterov technique.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, mu=None, **kwargs):

        super(NesterovFunction, self).__init__(**kwargs)

        if mu != None:
            self.set_mu(mu)

    @abc.abstractmethod
    def precompute(self, *args, **kwargs):
        raise NotImplementedError('Abstract method "precompute" must be ' \
                                  'specialised!')

    @abc.abstractmethod
    def alpha(self, beta, mu=None):
        raise NotImplementedError('Abstract method "alpha" must be ' \
                                  'specialised!')

    @abc.abstractmethod
    def A(self):
        raise NotImplementedError('Abstract method "A" must be ' \
                                  'specialised!')

    @abc.abstractmethod
    def At(self):
        raise NotImplementedError('Abstract method "At" must be ' \
                                  'specialised!')

    @abc.abstractmethod
    def projection(self, alpha):
        raise NotImplementedError('Abstract method "projection" must be ' \
                                  'specialised!')

    def get_mu(self):
        return self.mu

    def set_mu(self, mu):
        self.mu = mu


class CombinedLossFunction(LossFunction):

    def __init__(self, a=None, b=None, **kwargs):

        super(CombinedLossFunction, self).__init__(**kwargs)

        if a == None or b == None:
            raise ValueError('Two loss functions must be given as ' \
                             'arguments to the constructor')
        self.a = a
        self.b = b

    def f(self, *args, **kwargs):
        return self.a.f(*args, **kwargs) + self.b.f(*args, **kwargs)


class CombinedNesterovLossFunction(CombinedLossFunction, NesterovFunction):
    """A loss function contructed as the sum of two loss functions, where
    at least one of them is a Nesterov function, i.e. a loss function computed
    using the Nestrov technique.

    This function is of the form

        g = g1 + g2,

    where at least one of g1 or g2 are Nesterov functions.
    """

    def __init__(self, a, b):

        super(CombinedNesterovLossFunction, self).__init__(a=a, b=b)

    def grad(self, *args, **kwargs):

        return self.a.grad(*args, **kwargs) + self.b.grad(*args, **kwargs)

    def Lipschitz(self):

        return self.a.Lipschitz() + self.b.Lipschitz()

    def set_data(self, *args, **kwargs):

        if hasattr(self.a, 'set_data'):
            self.a.set_data(*args, **kwargs)
        if hasattr(self.b, 'set_data'):
            self.b.set_data(*args, **kwargs)

    def precompute(self, *args, **kwargs):

        if hasattr(self.a, 'precompute'):
            self.a.precompute(*args, **kwargs)
        if hasattr(self.b, 'precompute'):
            self.b.precompute(*args, **kwargs)

    def alpha(self, beta, mu):

        if hasattr(self.b, 'alpha'):
            return self.b.alpha(beta, mu)
        elif hasattr(self.a, 'alpha'):
            return self.a.alpha(beta, mu)
        else:
            raise ValueError('At least one loss function must be Nesterov')

    def A(self):

        if hasattr(self.b, 'A'):
            return self.b.A()
        elif hasattr(self.a, 'A'):
            return self.a.A()
        else:
            raise ValueError('At least one loss function must be Nesterov')

    def At(self):

        if hasattr(self.b, 'At'):
            return self.b.At()
        elif hasattr(self.a, 'At'):
            return self.a.At()
        else:
            raise ValueError('At least one loss function must be Nesterov')

    def projection(self):

        if hasattr(self.b, 'projection'):
            return self.b.projection()
        elif hasattr(self.a, 'projection'):
            return self.a.projection()
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


class LinearRegressionError(ConvexLossFunction,
                            Differentiable,
                            LipschitzContinuous):

    def __init__(self, **kwargs):

        super(LinearRegressionError, self).__init__(**kwargs)

    def set_data(self, X, y):

        self.X = X
        self.y = y
        self.Xy = np.dot(X.T, y)

        v = algorithms.FastSVD(max_iter=100).run(X)
        us = np.dot(X, v)
        self.lipschitz = 2.0 * np.sum(us ** 2.0)
#        _, s, _ = np.linalg.svd(X, full_matrices=False)  # False == faster
#        self.lipschitz = np.max(s) ** 2.0

    def f(self, beta, **kwargs):
#        return norm(self.y - np.dot(self.X, beta)) ** 2
        return np.sum((self.y - np.dot(self.X, beta)) ** 2.0)

    def grad(self, beta, **kwargs):
#        return 2.0 * np.dot(self.X.T, np.dot(self.X, beta) - self.y)
        return 2.0 * (np.dot(self.X.T, np.dot(self.X, beta)) - self.Xy)

    def Lipschitz(self):
        return self.lipschitz


class LogisticRegressionError(ConvexLossFunction,
                              Differentiable,
                              LipschitzContinuous):

    def __init__(self, X, y, **kwargs):

        super(LogisticRegressionError, self).__init__(**kwargs)

        self.X = X
        self.y = y

        V = 0.5 * np.eye(X.shape[0])  # pi(x) * (1 - pi(x)) <= 0.25 = 0.5 * 0.5
        VX = np.dot(V, X)
        svd = algorithms.FastSVD(max_iter=100)
        t = np.dot(VX, svd.run(VX))
        self.lipschitz = np.sum(t ** 2.0)
#        _, s, _ = np.linalg.svd(VX, full_matrices=False)  # False == faster
#        self.t = np.max(s) ** 2.0

    def f(self, beta, **kwargs):
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

    def Lipschitz(self):
        return self.lipschitz


class RidgeRegression(StronglyConvexLossFunction,
                      Differentiable,
                      LipschitzContinuous):
    """Loss function for ridge regression. Represents the function:

        f(b) = norm(y - X*b)² + lambda*norm(b)²
        #f(b) = (1/2)*norm(y - X*b)² + (lambda/2)*norm(b)²
    """

    def __init__(self, X, y, l, **kwargs):

        super(RidgeRegression, self).__init__(**kwargs)

        self.X = X
        self.y = y
        self.Xty = np.dot(X.T, y)
        self.l = l

        _, s, _ = np.linalg.svd(X, full_matrices=False)  # False == faster
        self.l_max = np.max(s) ** 2.0 + l
        self.l_min = np.min(s) ** 2.0 + l

    def f(self, beta, **kwargs):
#        return 0.5 * (np.sum((self.y - np.dot(self.X, beta)) ** 2.0) \
#                + self.l * np.sum(beta ** 2.0))
        return (np.sum((self.y - np.dot(self.X, beta)) ** 2.0) \
                + self.l * np.sum(beta ** 2.0))

    def grad(self, beta, **kwargs):
#        return (np.dot(self.X.T, np.dot(self.X, beta)) - self.Xty) \
#                + self.l * beta
        return 2.0 * ((np.dot(self.X.T, np.dot(self.X, beta)) - self.Xty) \
                + self.l * beta)

    def Lipschitz(self):
#        return self.lambda_max
        return 2.0 * self.l_max

    def lambda_min(self):
#        return self.lambda_min
        return 2.0 * self.l_min


class ZeroErrorFunction(ProximalOperator, Differentiable, LipschitzContinuous):

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


class L1(ProximalOperator):

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


class L2(ProximalOperator):
    """The proximal operator for the function lambda*0.5*norm(x)**2.
    """

    def __init__(self, l, **kwargs):

        super(L2, self).__init__(**kwargs)

        self.l = l

    def f(self, x):
        return self.l * 0.5 * np.sum(x ** 2.0)

    def prox(self, x, factor=1):

        l = factor * self.l
        return x / (1.0 + l)


class ElasticNet(ProximalOperator):
    """The proximal operator of the function

    lambda * norm1(x) + (1.0 - lambda) * 0.5 * norm(x) ** 2.
    """
    def __init__(self, l, **kwargs):

        super(ElasticNet, self).__init__(**kwargs)

        self.l = l
        self.l1 = L1(l)

    def f(self, x):
        return self.l * norm1(x) + (1.0 - self.l) * 0.5 * np.sum(x ** 2.0)

    def prox(self, x, factor=1, allow_empty=False):

        l = factor * self.l
        return self.l1.prox(x, factor=factor, allow_empty=allow_empty) \
                / (2.0 - l)


class TotalVariation(ConvexLossFunction,
                     NesterovFunction,
                     LipschitzContinuous):

    def __init__(self, gamma, shape, mu=None, mask=None, **kwargs):
        """Construct a TotalVariation loss function.

        Parameters:
        ---------
        gamma : The regularisation parameter for the TV penality.

        shape : The shape of the 3D image. Must be a 3-tuple. If the image is
                2D, let the Z dimension be 1, and if the "image" is 1D, let the
                Y and Z dimensions be 1. The tuple must be of the form
                (Z, Y, X).

        mu    : The Nesterov function regularisation parameter.

        mask  : A 1-dimensional mask representing the 3D image mask.
        """
        super(TotalVariation, self).__init__(mu=mu, **kwargs)

        self.gamma = gamma
        self.shape = shape
        self.mask = mask

        self.beta_id = None
        self.mu_id = None

        self.precompute()
        self.lambda_max = None

    def f(self, beta, mu=None, true=False):

        if self.gamma <= TOLERANCE:
            return 0

        if (mu == None):
            mu = self.get_mu()

        if (self.beta_id != id(beta) or self.mu_id != id(mu)) and not true:
            self.compute_alpha(beta, mu)
            self.beta_id = id(beta)
            self.mu_id = id(mu)

        if true:
            beta_norm2 = self.Ax.dot(beta) ** 2.0 + \
                         self.Ay.dot(beta) ** 2.0 + \
                         self.Az.dot(beta) ** 2.0
            return np.sum(beta_norm2)
        else:
            return np.dot(self.Aalpha.T, beta)[0, 0] - \
                    (mu / 2.0) * (np.sum(self.asx ** 2.0) +
                                  np.sum(self.asy ** 2.0) +
                                  np.sum(self.asz ** 2.0))

    def grad(self, beta):

        if self.gamma <= TOLERANCE:
            return np.zeros(beta.shape)

        if self.beta_id != id(beta) or self.mu_id != id(self.get_mu()):
            self.compute_alpha(beta, self.get_mu())
            self.beta_id = id(beta)
            self.mu_id = id(self.get_mu())

        return self.Aalpha

    def alpha(self, beta, mu=None):
        if self.gamma <= TOLERANCE:
            return 0

        if (mu == None):
            mu = self.get_mu()

        if self.beta_id != id(beta) or self.mu_id != id(mu):
            self.compute_alpha(beta, mu)
            self.beta_id = id(beta)
            self.mu_id = id(mu)

        return self.asx, self.asy, self.asz

    def A(self):
        return self.Ax, self.Ay, self.Az

    def At(self):
        return self.Axt, self.Ayt, self.Azt

    def projection(self, alpha):
        asx = alpha[0]
        asy = alpha[1]
        asz = alpha[2]
        asnorm = asx ** 2.0 + asy ** 2.0 + asz ** 2.0  # )**0.5
#        asnorm = np.sqrt(asnorm)
        i = asnorm > 1

        asnorm_i = asnorm[i] ** 0.5  # Square root is taken here. Faster.
        asx[i] = np.divide(asx[i], asnorm_i)
        asy[i] = np.divide(asy[i], asnorm_i)
        asz[i] = np.divide(asz[i], asnorm_i)

        return asx, asy, asz

    def _compute_lambda_max(self):

        A = sparse.vstack(self.A())
        v = algorithms.SparseSVD(max_iter=100).run(A)
        us = A.dot(v)
        self.lambda_max = np.sum(us ** 2.0)

    def Lipschitz(self):

        if self.gamma < TOLERANCE:
            return 0

        if self.lambda_max == None:
            self._compute_lambda_max()

        return self.lambda_max / self.get_mu()

    def compute_alpha(self, beta, mu):

        # Compute a* for each dimension
        q = self.gamma / mu
        self.asx = q * self.Ax.dot(beta)
        self.asy = q * self.Ay.dot(beta)
        self.asz = q * self.Az.dot(beta)

        # Apply projection
        self.asx, self.asy, self.asz = self.projection((self.asx,
                                                        self.asy,
                                                        self.asz))
#        asnorm = self.asx ** 2.0 + self.asy ** 2.0 + self.asz ** 2.0  # )**0.5
##        asnorm = np.sqrt(asnorm)
#        i = asnorm > 1
#
#        asnorm_i = asnorm[i] ** 0.5  # Square root is taken here. Faster.
#        self.asx[i] = np.divide(self.asx[i], asnorm_i)
#        self.asy[i] = np.divide(self.asy[i], asnorm_i)
#        self.asz[i] = np.divide(self.asz[i], asnorm_i)

#        self.Aalpha = self.Axt.dot(self.asx) + \
#                      self.Ayt.dot(self.asy) + \
#                      self.Azt.dot(self.asz)
        self.Aalpha = np.add(np.add(self.Axt.dot(self.asx),
                                    self.Ayt.dot(self.asy), self.buff),
                                    self.Azt.dot(self.asz), self.buff)

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

    def precompute(self):

        Z = self.shape[0]
        Y = self.shape[1]
        X = self.shape[2]
        p = X * Y * Z

        smtype = 'csr'
        self.Ax = sparse.eye(p, p, 1, format=smtype) \
                - sparse.eye(p, p)
        self.Ay = sparse.eye(p, p, X, format=smtype) \
                - sparse.eye(p, p)
        self.Az = sparse.eye(p, p, X * Y, format=smtype) \
                - sparse.eye(p, p)

        ind = np.reshape(xrange(p), (Z, Y, X))
        if self.mask != None:
            mask = np.reshape(self.mask, (Z, Y, X))
            xind, yind, zind = self._find_mask_ind(mask, ind)
        else:
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

        # Remove rows corresponding to indices excluded in all dimensions
        toremove = list(set(xind).intersection(yind).intersection(zind))
        toremove.sort()
        toremove.reverse()  # Remove from end so that indices are not changed
        for i in toremove:
            delete_sparse_csr_row(self.Ax, i)
            delete_sparse_csr_row(self.Ay, i)
            delete_sparse_csr_row(self.Az, i)

        if self.mask != None:
            self.Axt = self.Ax.T.tocsr()
            self.Ayt = self.Ay.T.tocsr()
            self.Azt = self.Az.T.tocsr()
            for i in reversed(xrange(p)):
                if self.mask[i] == 0:
                    delete_sparse_csr_row(self.Axt, i)
                    delete_sparse_csr_row(self.Ayt, i)
                    delete_sparse_csr_row(self.Azt, i)

            self.Ax = self.Axt.T
            self.Ay = self.Ayt.T
            self.Az = self.Azt.T
        else:
            self.Axt = self.Ax.T
            self.Ayt = self.Ay.T
            self.Azt = self.Az.T

        self.buff = np.zeros((self.Ax.shape[1], 1))


class GroupLassoOverlap(ConvexLossFunction,
                        NesterovFunction,
                        LipschitzContinuous):

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
        super(GroupLassoOverlap, self).__init__(mu=mu, **kwargs)

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
            mu = self.get_mu()

        if self.beta_id != id(beta) or self.mu_id != id(mu):
            self.compute_alpha(beta, mu)
            self.beta_id = id(beta)
            self.mu_id = id(mu)

        sumastar = 0
        for g in xrange(len(self.astar)):
            sumastar += np.sum(self.astar[g] ** 2.0)

        return np.dot(self.Aalpha.T, beta)[0, 0] - (mu / 2.0) * sumastar

    def grad(self, beta):

        if self.beta_id != id(beta) or self.mu_id != id(self.get_mu()):
            self.compute_alpha(beta, self.get_mu())
            self.beta_id = id(beta)
            self.mu_id = id(self.get_mu())

        return self.Aalpha

    def alpha(self, beta, mu=None):
        if (mu == None):
            mu = self.get_mu()

        if self.beta_id != id(beta) or self.mu_id != id(mu):
            self.compute_alpha(beta, mu)
            self.beta_id = id(beta)
            self.mu_id = id(mu)

        return self.astar

    def A(self):
        return self.A

    def At(self):
        return self.At

    def projection(self, alpha):
        for i in xrange(len(alpha)):
            astar = alpha[i]
            normas = np.sqrt(np.dot(astar.T, astar))
            if normas > 1.0:
                astar /= normas
            alpha[i] = astar

        return alpha

    def Lipschitz(self):

        return self.max_col_norm / self.mu

    def compute_alpha(self, beta, mu):

        # Compute a* for each dimension
        self.Aalpha = 0
        q = self.gamma / mu
        for g in xrange(len(self.A)):
            astar = q * self.A[g].dot(beta)
            astar = self.projection([astar])[0]

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