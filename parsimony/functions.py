# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.functions` module contains several functions used
throughout the package. These represent mathematical functions and should thus
have properties used by the corresponding algorithms.

Loss functions should be stateless. Loss functions may be shared and copied and
should therefore not hold anythig that cannot be recomputed the next time it is
called.

Created on Mon Apr 22 10:54:29 2013

@author:  Tommy Löfstedt, Vincent Guillemot and Fouad Hadj Selem
@email:   tommy.loefstedt@cea.fr
@license: TBD.
"""
import abc
import numpy as np
import scipy.sparse as sparse
import math

import parsimony.utils.maths as maths
import parsimony.utils.consts as consts

__all__ = ['RidgeRegression', 'L1', 'SmoothedL1', 'TotalVariation',
           'RR_L1_TV',
           'SmoothedL1TV',
           'RR_SmoothedL1TV']


#mapping = {RidgeRegression.__class__.__name__: [algorithms.FISTA,
#                                                algorithms.CONESTA]}


class Function(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def f(self, *args, **kwargs):
        raise NotImplementedError('Abstract method "f" must be ' \
                                  'specialised!')

    def reset(self):
        pass

    def set_params(self, **kwargs):

        for k in kwargs:
            self.__setattr__(k, kwargs[k])


class AtomicFunction(Function):
    """ This is a function that is not in general supposed to be minimised by
    itself. Instead it should be combined with other atomic functions and
    composite functions into composite functions.
    """
    __metaclass__ = abc.ABCMeta


class CompositeFunction(Function):
    """ This is a function that is the combination (i.e. sum) of other
    composite or atomic functions.
    """
    __metaclass__ = abc.ABCMeta


class Regularisation(object):

    __metaclass__ = abc.ABCMeta


class Constraint(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def feasible(self):
        """Feasibility of the constraint.
        """
        raise NotImplementedError('Abstract method "feasible" must be ' \
                                  'specialised!')


class ProximalOperator(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def prox(self, beta):
        """The proximal operator corresponding to the function.
        """
        raise NotImplementedError('Abstract method "prox" must be ' \
                                  'specialised!')


class NesterovFunction(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, l, c=0.0, A=None, mu=consts.TOLERANCE):

        self.l = float(l)
        self.c = float(c)
        self._A = A
        self.mu = float(mu)

    def fmu(self, beta, mu=None):
        """Returns the smoothed function value.
        """
        if mu == None:
            mu = self.get_mu()

        alpha = self.alpha(beta)
        alpha_sqsum = 0.0
        for a in alpha:
            alpha_sqsum += np.sum(a ** 2.0)

        Aa = self.Aa(alpha)

        return self.l * ((np.dot(beta.T, Aa)[0, 0] \
                          - (mu / 2.0) * alpha_sqsum) - self.c)

    def grad(self, beta):
        """ Gradient of the function at beta.
        """
        if self.l < consts.TOLERANCE:
            return 0.0

        alpha = self.alpha(beta)

        return self.l * self.Aa(alpha)

    def get_mu(self):
        """Returns the regularisation constant for the smoothing.
        """
        return self.mu

    def set_mu(self, mu):
        """Sets the regularisation constant for the smoothing.

        Arguments:
        =========
        mu: The regularisation constant for the smoothing to use from now on.

        Returns:
        =======
        old_mu: The old regularisation constant for the smoothing that was
                overwritten and is no longer used.
        """
        old_mu = self.get_mu()

        self.mu = mu

        return old_mu

    @abc.abstractmethod
    def phi(self, alpha, beta):
        """ Function value with known alpha.
        """
        raise NotImplementedError('Abstract method "phi" must be ' \
                                  'specialised!')

    def alpha(self, beta):
        """ Dual variable of the Nesterov function.
        """
        A = self.A()
        mu = self.get_mu()
        alpha = [0] * len(A)
        for i in xrange(len(A)):
            alpha[i] = A[i].dot(beta) / mu

        # Apply projection
        alpha = self.project(alpha)

        return alpha

    def A(self):
        """ Linear operator of the Nesterov function.
        """
        return self._A

    def Aa(self, alpha):
        """ Computes A^\T\alpha.
        """
        A = self.A()
        Aa = A[0].T.dot(alpha[0])
        for i in xrange(1, len(A)):
            Aa += A[i].T.dot(alpha[i])

        return Aa

    @abc.abstractmethod
    def project(self, a):
        """ Projection onto the compact space of the Nesterov function.
        """
        raise NotImplementedError('Abstract method "project" must be ' \
                                  'specialised!')

    @abc.abstractmethod
    def M(self):
        """ The maximum value of the regularisation of the dual variable. We
        have

            M = max_{\alpha \in K} 0.5*|\alpha|²_2.
        """
        raise NotImplementedError('Abstract method "M" must be ' \
                                  'specialised!')

    @abc.abstractmethod
    def estimate_mu(self, beta):
        """ Computes a "good" value of \mu with respect to the given \beta.
        """
        raise NotImplementedError('Abstract method "mu" must be ' \
                                  'specialised!')


class Continuation(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def mu_opt(self, eps):
        """The optimal value of \mu given \epsilon.
        """
        raise NotImplementedError('Abstract method "mu_opt" must be ' \
                                  'specialised!')

    @abc.abstractmethod
    def eps_opt(self, mu):
        """The optimal value of \epsilon given \mu.
        """
        raise NotImplementedError('Abstract method "eps_opt" must be ' \
                                  'specialised!')

    @abc.abstractmethod
    def eps_max(self, mu):
        """The maximum value of \epsilon.
        """
        raise NotImplementedError('Abstract method "eps_max" must be ' \
                                  'specialised!')


class Gradient(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def grad(self, beta):
        """Gradient of the function.
        """
        raise NotImplementedError('Abstract method "grad" must be ' \
                                  'specialised!')


class Hessian(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def hessian(self, beta, vector=None):
        """Hessian (second derivative) of the function.

        Arguments:
        ---------
        beta : The point at which to evaluate the Hessian.

        vector : If not None, it is multiplied with the Hessian from the right.
        """
        raise NotImplementedError('Abstract method "hessian" must be ' \
                                  'specialised!')

    @abc.abstractmethod
    def hessian_inverse(self, beta, vector=None):
        """Inverse of the Hessian (second derivative) of the function.

        Sometimes this can be done efficiently if we know the structure of the
        Hessian. Also, if we multiply the Hessian by a vector, it is often
        possible to do efficiently.

        Arguments:
        ---------
        beta : The point at which to evaluate the Hessian.

        vector : If not None, it is multiplied with the inverse of the Hessian
                from the right.
        """
        raise NotImplementedError('Abstract method "hessian_inverse" must be '\
                                  'specialised!')


class LipschitzContinuousGradient(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def L(self):
        """Lipschitz constant of the gradient.
        """
        raise NotImplementedError('Abstract method "L" must be ' \
                                  'specialised!')


class GradientMap(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def V(self, alpha, beta, L):
        """The gradient map associated to the function.
        """
        raise NotImplementedError('Abstract method "V" must be ' \
                                  'specialised!')


class DualFunction(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def gap(self, beta, beta_hat):
        """Compute the duality gap.
        """
        raise NotImplementedError('Abstract method "gap" must be ' \
                                  'specialised!')

    @abc.abstractmethod
    def betahat(self, alpha, beta=None):
        """Returns the beta that minimises the dual function.
        """
        raise NotImplementedError('Abstract method "betahat" must be ' \
                                  'specialised!')


class Eigenvalues(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def lambda_max(self):
        """Largest eigenvalue of the corresponding covariance matrix.
        """
        raise NotImplementedError('Abstract method "lambda_max" must be ' \
                                  'specialised!')

    def lambda_min(self):
        """Smallest eigenvalue of the corresponding covariance matrix.
        """
        raise NotImplementedError('Abstract method "lambda_min" is not ' \
                                  'implemented!')


class RidgeRegression(CompositeFunction, Gradient, LipschitzContinuousGradient,
                      Eigenvalues):

    def __init__(self, X, y, k):

        self.X = X
        self.y = y
        self.k = float(k)

        self.reset()

    def reset(self):

        self._lambda_max = None
        self._lambda_min = None

    def f(self, beta):
        """Function value of Ridge regression.
        """
        return (1.0 / 2.0) * np.sum((np.dot(self.X, beta) - self.y) ** 2.0) \
             + (self.k / 2.0) * np.sum(beta ** 2.0)

    def grad(self, beta):
        """Gradient of the function at beta.

        From the interface "Gradient".
        """
        return np.dot((np.dot(self.X, beta) - self.y).T, self.X).T \
             + self.k * beta

    def L(self):
        """Lipschitz constant of the gradient.

        From the interface "LipschitzContinuousGradient".
        """
        return self.lambda_max()

    def lambda_max(self):
        """Largest eigenvalue of the corresponding covariance matrix.

        From the interface "Eigenvalues".
        """
        if self._lambda_max == None:
            s = np.linalg.svd(self.X, full_matrices=False, compute_uv=False)

            self._lambda_max = np.max(s) ** 2.0

            if len(s) < self.X.shape[1]:
                self._lambda_min = 0.0
            else:
                self._lambda_min = np.min(s) ** 2.0

        return self._lambda_max + self.k

    def lambda_min(self):
        """Smallest eigenvalue of the corresponding covariance matrix.

        From the interface "Eigenvalues".
        """
        if self._lambda_min == None:
            s = np.linalg.svd(self.X, full_matrices=False, compute_uv=False)

            self._lambda_max = np.max(s) ** 2.0

            if len(s) < self.X.shape[1]:
                self._lambda_min = 0.0
            else:
                self._lambda_min = np.min(s) ** 2.0

        return self._lambda_min + self.k


class L1(AtomicFunction, Constraint, ProximalOperator):
    """The proximal operator of the L1 function

        f(beta) = l * (||beta||_1 - c),

    where ||beta||_1 is the L1 loss function. The constrained version has the
    form

        f(beta) <= c.

    Parameters:
    ----------
    l : The Lagrange multiplier, or regularisation constant, of the function.

    c : The limit of the constraint. The function is feasible if f(beta) <= c.
            The default value is c=0, i.e. the default is a regularisation
            formulation.
    """
    def __init__(self, l=1.0, c=0.0):

        self.l = float(l)
        self.c = float(c)

    def f(self, beta):
        """Function value.
        """
        return self.l * (maths.norm1(beta) - self.c)

    def prox(self, beta, factor=1.0):
        """The corresponding proximal operator.

        From the interface "ProximalOperator".
        """
        l = self.l * factor

        return (np.abs(beta) > l) * (beta - l * np.sign(beta - l))

    def feasible(self, beta):
        """Feasibility of the constraint.

        From the interface "Constraint".
        """
        return self.f(beta) <= self.c


class SmoothedL1(AtomicFunction, NesterovFunction, Gradient,
                 LipschitzContinuousGradient):
    """The proximal operator of the smoothed L1 function

        f(beta) = l * (L1mu(beta) - c),

    where L1mu(beta) is the smoothed L1 function. The constrained version has
    the form

        f(beta) <= c.

    Parameters
    ----------
    l : The Lagrange multiplier, or regularisation constant, of the function.

    c : The limit of the constraint. The function is feasible if f(beta) <= c.
            The default value is c=0, i.e. the default is a regularisation
            formulation.

    A : The linear operator for the Nesterov formulation. May not be None.

    mu : The regularisation constant for the smoothing.
    """
    def __init__(self, l, c=0.0, A=None, mu=0.0):

        super(SmoothedL1, self).__init__(l, c, A, mu)

    def f(self, beta):
        """ Function value.
        """
        if self.l < consts.TOLERANCE:
            return 0.0

        return self.l * (maths.norm1(beta) - self.c)

    def phi(self, alpha, beta):
        """ Function value with known alpha.

        From the interface "NesterovFunction".
        """
        if self.l < consts.TOLERANCE:
            return 0.0

        return self.l * ((np.dot(alpha[0].T, beta)[0, 0] \
                         - (self.mu / 2.0) * np.sum(alpha[0] ** 2.0)) - self.c)

    def grad(self, beta):
        """ Gradient of the function at beta.

        From the interface "Gradient". Overloaded since we can be faster than
        the default.
        """
        alpha = self.alpha(beta)

        return self.l * alpha[0]

    def L(self):
        """ Lipschitz constant of the gradient.

        From the interface "LipschitzContinuousGradient".
        """
        return self.l / self.mu

#    """ Linear operator of the Nesterov function.
#
#    From the interface "NesterovFunction".
#    """
#    def A(self):
#
#        return self._A

    def alpha(self, beta):
        """ Dual variable of the Nesterov function.

        From the interface "NesterovFunction". Overloaded since we can be
        faster than the default.
        """
        alpha = self.project([beta / self.mu])

        return alpha

    @staticmethod
    def project(a):
        """ Projection onto the compact space of the Nesterov function.

        From the interface "NesterovFunction".
        """
        a = a[0]
        anorm = np.abs(a)
        i = anorm > 1.0
        anorm_i = anorm[i]
        a[i] = np.divide(a[i], anorm_i)

        return [a]

    def M(self):
        """ The maximum value of the regularisation of the dual variable. We
        have

            M = max_{\alpha \in K} 0.5*|\alpha|²_2.

        From the interface "NesterovFunction".
        """
        A = self.A()
        return A[0].shape[0] / 2.0

    def estimate_mu(self, beta):
        """ Computes a "good" value of \mu with respect to the given \beta.

        From the interface "NesterovFunction".
        """
        return np.max(np.absolute(beta))


class TotalVariation(AtomicFunction, NesterovFunction, Gradient,
                     LipschitzContinuousGradient):
    """The proximal operator of the smoothed Total variation (TV) function

        f(\beta) = l * (TV(\beta) - c),

    where TV(beta) is the smoothed L1 function. The constrained version has the
    form

        f(\beta) <= c.

    Parameters
    ----------
    l : The Lagrange multiplier, or regularisation constant, of the function.

    c : The limit of the constraint. The function is feasible if f(\beta) <= c.
            The default value is c=0, i.e. the default is a regularisation
            formulation.

    A : The linear operator for the Nesterov formulation. May not be None!

    mu : The regularisation constant for the smoothing.
    """
    def __init__(self, l, c=0.0, A=None, mu=0.0):

        super(TotalVariation, self).__init__(l, c, A, mu)
        self._p = A[0].shape[1]

        self.reset()

    def reset(self):

        self._lambda_max = None

    def f(self, beta):
        """ Function value.
        """
        if self.l < consts.TOLERANCE:
            return 0.0

        A = self.A()
        return self.l * (np.sum(np.sqrt(A[0].dot(beta) ** 2.0 + \
                                        A[1].dot(beta) ** 2.0 + \
                                        A[2].dot(beta) ** 2.0)) - self.c)

    def phi(self, alpha, beta):
        """Function value with known alpha.

        From the interface "NesterovFunction".
        """
        if self.l < consts.TOLERANCE:
            return 0.0

        Aa = self.Aa(alpha)

        alpha_sqsum = 0.0
        for a in alpha:
            alpha_sqsum += np.sum(a ** 2.0)

        return self.l * ((np.dot(beta.T, Aa)[0, 0] \
                          - (self.mu / 2.0) * alpha_sqsum) - self.c)

#    """ Gradient of the function at beta.
#
#    From the interface "Gradient".
#    """
#    def grad(self, beta):
#
#        if self.l < utils.TOLERANCE:
#            return 0.0
#
#        alpha = self.alpha(beta)
#
#        return self.l * self.Aa(alpha)

    def L(self):
        """ Lipschitz constant of the gradient.

        From the interface "LipschitzContinuousGradient".
        """
        if self.l < consts.TOLERANCE:
            return 0.0

        lmaxA = self.lambda_max()

        return self.l * lmaxA / self.mu

    def lambda_max(self):
        """ Largest eigenvalue of the corresponding covariance matrix.

        From the interface "Eigenvalues".
        """
        # Note that we can save the state here since lmax(A) does not change.
        if len(self._A) == 3 \
            and self._A[1].nnz == 0 and self._A[2].nnz == 0:
            # TODO: Instead of p, this should really be the number of non-zero
            # rows of A.
            self._lambda_max = 2.0 * (1.0 - math.cos(float(self._p - 1) \
                                                     * math.pi \
                                                     / float(self._p)))

        elif self._lambda_max == None:

            from parsimony.algorithms import FastSparseSVD

            A = sparse.vstack(self.A())
            # TODO: Add max_iter here!
            v = FastSparseSVD()(A)  # , max_iter=max_iter)
            us = A.dot(v)
            self._lambda_max = np.sum(us ** 2.0)

        return self._lambda_max

#    """ Linear operator of the Nesterov function.
#
#    From the interface "NesterovFunction".
#    """
#    def A(self):
#
#        return self._A

#    """ Computes A^\T\alpha.
#
#    From the interface "NesterovFunction".
#    """
#    def Aa(self, alpha):
#
#        A = self.A()
#        Aa = A[0].T.dot(alpha[0])
#        for i in xrange(1, len(A)):
#            Aa += A[i].T.dot(alpha[i])
#
#        return Aa

#    """ Dual variable of the Nesterov function.
#
#    From the interface "NesterovFunction".
#    """
#    def alpha(self, beta):
#
#        # Compute a*
#        A = self.A()
#        alpha = [0] * len(A)
#        for i in xrange(len(A)):
#            alpha[i] = A[i].dot(beta) / self.mu
#
#        # Apply projection
#        alpha = self.project(alpha)
#
#        return alpha

    def project(self, a):
        """ Projection onto the compact space of the Nesterov function.

        From the interface "NesterovFunction".
        """
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
        """ The maximum value of the regularisation of the dual variable. We
        have

            M = max_{\alpha \in K} 0.5*|\alpha|²_2.

        From the interface "NesterovFunction".
        """
        return self._A[0].shape[0] / 2.0

    """ Computes a "good" value of \mu with respect to the given \beta.

    From the interface "NesterovFunction".
    """
    def estimate_mu(self, beta):

        SS = 0
        A = self.A()
        for i in xrange(len(A)):
            SS += A[i].dot(beta) ** 2.0

        return np.max(np.sqrt(SS))

#    @staticmethod
#    def precompute(shape, mask=None, compress=True):
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
#        Ax = sparse.eye(p, p, 1, format=smtype) - sparse.eye(p, p)
#        Ay = sparse.eye(p, p, X, format=smtype) - sparse.eye(p, p)
#        Az = sparse.eye(p, p, X * Y, format=smtype) - sparse.eye(p, p)
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


class RR_L1_TV(CompositeFunction, Gradient, LipschitzContinuousGradient,
                 ProximalOperator, NesterovFunction, Continuation,
                 DualFunction):

    def __init__(self, X, y, k, l, g, A=None, mu=0.0):

        self.X = X
        self.y = y

        self.rr = RidgeRegression(X, y, k)
        self.l1 = L1(l)
        self.tv = TotalVariation(g, A=A, mu=0.0)

        self.reset()

    def reset(self):

        self.rr.reset()
        self.l1.reset()
        self.tv.reset()

        self._Xty = None
        self._invXXkI = None
        self._XtinvXXtkI = None

    def set_params(self, **kwargs):

        mu = kwargs.pop("mu", self.get_mu())
        self.set_mu(mu)

        super(RR_L1_TV, self).set_params(**kwargs)

    def get_mu(self):
        """Returns the regularisation constant for the smoothing.

        From the interface "NesterovFunction".
        """
        return self.tv.get_mu()

    def set_mu(self, mu):
        """Sets the regularisation constant for the smoothing.

        From the interface "NesterovFunction".

        Arguments:
        =========
        mu: The regularisation constant for the smoothing to use from now on.

        Returns:
        =======
        old_mu: The old regularisation constant for the smoothing that was
                overwritten and is no longer used.
        """
        return self.tv.set_mu(mu)

    def f(self, beta):
        """Function value.
        """
        return self.rr.f(beta) \
             + self.l1.f(beta) \
             + self.tv.f(beta)

    def phi(self, alpha, beta):
        """ Function value with known alpha.
        """
        return self.rr.f(beta) \
             + self.l1.f(beta) \
             + self.tv.phi(alpha, beta)

    def grad(self, beta):
        """Gradient of the differentiable part of the function.

        From the interface "Gradient".
        """
        return self.rr.grad(beta) \
             + self.tv.grad(beta)

    def L(self):
        """Lipschitz constant of the gradient.

        From the interface "LipschitzContinuousGradient".
        """
        return self.rr.L() \
             + self.tv.L()

    def prox(self, beta, factor=1.0):
        """The proximal operator of the non-differentiable part of the
        function.

        From the interface "ProximalOperator".
        """
        return self.l1.prox(beta, factor)

    def estimate_mu(self, beta):
        """Computes a "good" value of \mu with respect to the given \beta.

        From the interface "NesterovFunction".
        """
        return self.tv.estimate_mu(beta)

    def M(self):
        """The maximum value of the regularisation of the dual variable. We
        have

            M = max_{\alpha \in K} 0.5*|\alpha|²_2.

        From the interface "NesterovFunction".
        """
        return self.tv.M()

    def mu_opt(self, eps):
        """The optimal value of \mu given \epsilon.

        From the interface "Continuation".
        """
        gM = self.tv.l * self.tv.M()

        # Mu is set to 1.0, because it is in fact not here "anymore". It is
        # factored out in this solution.
        old_mu = self.tv.set_mu(1.0)
        gA2 = self.tv.L()  # Gamma is in here!
        self.tv.set_mu(old_mu)

        Lg = self.rr.L()

        return (-gM * gA2 + np.sqrt((gM * gA2) ** 2.0 \
             + gM * Lg * gA2 * eps)) \
             / (gM * Lg)

    def eps_opt(self, mu):
        """The optimal value of \epsilon given \mu.

        From the interface "Continuation".
        """
        gM = self.tv.l * self.tv.M()

        # Mu is set to 1.0, because it is in fact not here "anymore". It is
        # factored out in this solution.
        old_mu = self.tv.set_mu(1.0)
        gA2 = self.tv.L()  # Gamma is in here!
        self.tv.set_mu(old_mu)

        Lg = self.rr.L()

        return (2.0 * gM * gA2 * mu \
             + gM * Lg * mu ** 2.0) \
             / gA2

    def eps_max(self, mu):
        """The maximum value of \epsilon.

        From the interface "Continuation".
        """
        gM = self.tv.l * self.tv.M()

        return mu * gM

    def betahat(self, alphak, betak):
        """ Returns the beta that minimises the dual function. Used when we
        compute the gap.

        From the interface "DualFunction".
        """
        if self._Xty == None:
            self._Xty = np.dot(self.X.T, self.y)

        Ata_tv = self.tv.l * self.tv.Aa(alphak)
        Ata_l1 = self.l1.l * SmoothedL1.project([betak / consts.TOLERANCE])[0]
        v = (self._Xty - Ata_tv - Ata_l1)

        shape = self.X.shape

        if shape[0] > shape[1]:  # If n > p

            # Ridge solution
            if self._invXXkI == None:
                XtXkI = np.dot(self.X.T, self.X)
                index = np.arange(min(XtXkI.shape))
                XtXkI[index, index] += self.rr.k
                self._invXXkI = np.linalg.inv(XtXkI)

            beta_hat = np.dot(self._invXXkI, v)

        else:  # If p > n
            # Ridge solution using the Woodbury matrix identity:
            if self._XtinvXXtkI == None:
                XXtkI = np.dot(self.X, self.X.T)
                index = np.arange(min(XXtkI.shape))
                XXtkI[index, index] += self.rr.k
                invXXtkI = np.linalg.inv(XXtkI)
                self._XtinvXXtkI = np.dot(self.X.T, invXXtkI)

            beta_hat = (v - np.dot(self._XtinvXXtkI, np.dot(self.X, v))) \
                       / self.rr.k

        return beta_hat

    def gap(self, beta, beta_hat=None):
        """Compute the duality gap.

        From the interface "DualFunction".
        """
        alpha = self.tv.alpha(beta)

        P = self.rr.f(beta) \
          + self.l1.f(beta) \
          + self.tv.phi(alpha, beta)

        beta_hat = self.betahat(alpha, beta)

        D = self.rr.f(beta_hat) \
          + self.l1.f(beta_hat) \
          + self.tv.phi(alpha, beta_hat)

        return P - D

    def A(self):
        """Linear operator of the Nesterov function.

        From the interface "NesterovFunction".
        """
        return self.tv.A()

    def Aa(self, alpha):
        """Computes A^\T\alpha.

        From the interface "NesterovFunction".
        """
        return self.tv.Aa()

    def project(self, a):
        """ Projection onto the compact space of the Nesterov function.

        From the interface "NesterovFunction".
        """
        return self.tv.project(a)


class SmoothedL1TV(AtomicFunction, Regularisation, NesterovFunction,
                   Eigenvalues):

    def __init__(self, l, g, Atv=None, Al1=None, mu=0.0):

        self.l = float(l)
        self.g = float(g)

        self._p = Atv[0].shape[1]  # WARNING: Number of rows may differ from p.
        if Al1 == None:
            Al1 = sparse.eye(self._p, self._p)
        self._A = [l * Al1,
                   g * Atv[0],
                   g * Atv[1],
                   g * Atv[2]]

        self.mu = float(mu)

        # TODO: Is reset still necessary?
        self.reset()

    def reset(self):

        self._lambda_max = None

    def f(self, beta):
        """ Function value.
        """
        if self.l < consts.TOLERANCE and self.g < consts.TOLERANCE:
            return 0.0

        A = self.A()
        return maths.norm1(A[0].dot(beta)) + \
               np.sum(np.sqrt(A[1].dot(beta) ** 2.0 + \
                              A[2].dot(beta) ** 2.0 + \
                              A[3].dot(beta) ** 2.0))

    def phi(self, alpha, beta):
        """ Function value with known alpha.

        From the interface "NesterovFunction".
        """
        if self.l < consts.TOLERANCE and self.g < consts.TOLERANCE:
            return 0.0

        Aa = self.Aa(alpha)

        alpha_sqsum = 0.0
        for a in alpha:
            alpha_sqsum += np.sum(a ** 2.0)

        return np.dot(beta.T, Aa)[0, 0] - (self.mu / 2.0) * alpha_sqsum

    def lambda_max(self):
        """ Largest eigenvalue of the corresponding covariance matrix.

        From the interface "Eigenvalues".
        """
        # Note that we can save the state here since lmax(A) does not change.
        if len(self._A) == 4 \
            and self._A[2].nnz == 0 and self._A[3].nnz == 0:
#        if len(self._shape) == 3 \
#            and self._shape[0] == 1 and self._shape[1] == 1:
            # TODO: Instead of p, this should really be the number of non-zero
            # rows of A.
            p = self._A[1].shape[0]
            lmaxTV = 2.0 * (1.0 - math.cos(float(p - 1) * math.pi \
                                           / float(p)))
            self._lambda_max = lmaxTV * self.g ** 2.0 + self.l ** 2.0

        elif self._lambda_max == None:

            from parsimony.algorithms import FastSparseSVD

#            A = sparse.vstack(self.A())
#            v = algorithms.FastSparseSVD(A, max_iter=max_iter)
#            us = A.dot(v)
#            self._lambda_max = np.sum(us ** 2.0)

            A = sparse.vstack(self.A()[1:])
            # TODO: Add max_iter here!!
            v = FastSparseSVD()(A)  # , max_iter=max_iter)
            us = A.dot(v)
            self._lambda_max = np.sum(us ** 2.0) + self.l ** 2.0

        return self._lambda_max

#    """ Linear operator of the Nesterov function.
#
#    From the interface "NesterovFunction".
#    """
#    def A(self):
#
#        return self._A

#    """ Computes A^\T\alpha.
#
#    From the interface "NesterovFunction".
#    """
#    def Aa(self, alpha):
#
#        A = self.A()
#        Aa = A[0].T.dot(alpha[0])
#        for i in xrange(1, len(A)):
#            Aa += A[i].T.dot(alpha[i])
#
#        return Aa

    def alpha(self, beta):
        """ Dual variable of the Nesterov function.

        From the interface "NesterovFunction". Overloaded since we need to do
        more than the default.
        """
        A = self.A()

        a = [0] * len(A)
        a[0] = (1.0 / self.mu) * A[0].dot(beta)
        a[1] = (1.0 / self.mu) * A[1].dot(beta)
        a[2] = (1.0 / self.mu) * A[2].dot(beta)
        a[3] = (1.0 / self.mu) * A[3].dot(beta)
        # Remember: lambda and gamma are already in the A matrices.

        return self.project(a)

    def project(self, a):
        """ Projection onto the compact space of the Nesterov function.

        From the interface "NesterovFunction".
        """
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

        anorm_tv_i = anorm_tv[i_tv] ** 0.5  # Square root taken here. Faster.
        ax[i_tv] = np.divide(ax[i_tv], anorm_tv_i)
        ay[i_tv] = np.divide(ay[i_tv], anorm_tv_i)
        az[i_tv] = np.divide(az[i_tv], anorm_tv_i)

        return [al1, ax, ay, az]

    def estimate_mu(self, beta):
        """Computes a "good" value of \mu with respect to the given \beta.

        From the interface "NesterovFunction".
        """
        raise NotImplementedError("We do not use this here!")

    def M(self):
        """ The maximum value of the regularisation of the dual variable. We
        have

            M = max_{\alpha \in K} 0.5*|\alpha|²_2.

        From the interface "NesterovFunction".
        """
        A = self.A()

        return (A[0].shape[0] / 2.0) \
             + (A[1].shape[0] / 2.0)


class RR_SmoothedL1TV(CompositeFunction, LipschitzContinuousGradient,
                        GradientMap, DualFunction, NesterovFunction):

    def __init__(self, X, y, k, l, g, Atv=None, Al1=None, mu=0.0):

        self.X = X
        self.y = y

        self.g = RidgeRegression(X, y, k)
        self.h = SmoothedL1TV(l, g, Atv=Atv, Al1=Al1, mu=mu)

        self.mu = mu

        self.reset()

    def reset(self):

        self.g.reset()
        self.h.reset()

        self._Xy = None
        self._XtinvXXtkI = None

    def set_params(self, **kwargs):

        mu = kwargs.pop("mu", self.get_mu())
        self.set_mu(mu)

        super(RR_SmoothedL1TV, self).set_params(**kwargs)

    def get_mu(self):
        """Returns the regularisation constant for the smoothing.

        From the interface "NesterovFunction".
        """
        return self.h.get_mu()

    def set_mu(self, mu):
        """Sets the regularisation constant for the smoothing.

        From the interface "NesterovFunction".

        Arguments:
        =========
        mu: The regularisation constant for the smoothing to use from now on.

        Returns:
        =======
        old_mu: The old regularisation constant for the smoothing that was
                overwritten and is no longer used.
        """
        return self.h.set_mu(mu)

    def f(self, beta):
        """ Function value.
        """
        return self.g.f(beta) \
             + self.h.f(beta)

    def phi(self, alpha, beta):
        """ Function value.
        """
        return self.g.f(beta) \
             + self.h.phi(alpha, beta)

    def L(self):
        """Lipschitz constant of the gradient.

        From the interface "LipschitzContinuousGradient".
        """
        b = self.g.lambda_min()
        # TODO: Use max_iter here!!
        a = self.h.lambda_max()  # max_iter=max_iter)

        return a / b

    def V(self, u, beta, L):
        """The gradient map associated to the function.

        From the interface "GradientMap".
        """
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

    def betahat(self, alpha, beta=None):
        """ Returns the beta that minimises the dual function.

        From the interface "DualFunction".
        """
        # TODO: Kernelise this function! See how I did in LRL2_L1_TV._beta_hat.

        A = self.h.A()
        grad = A[0].T.dot(alpha[0])
        grad += A[1].T.dot(alpha[1])
        grad += A[2].T.dot(alpha[2])
        grad += A[3].T.dot(alpha[3])

#        XXkI = np.dot(X.T, X) + self.g.k * np.eye(X.shape[1])

        if self._Xy == None:
            self._Xy = np.dot(self.X.T, self.y)

        Xty_grad = (self._Xy - grad) / self.g.k

#        t = time()
#        XXkI = np.dot(X.T, X)
#        index = np.arange(min(XXkI.shape))
#        XXkI[index, index] += self.g.k
#        invXXkI = np.linalg.inv(XXkI)
#        print "t:", time() - t
#        beta = np.dot(invXXkI, Xty_grad)

        if self._XtinvXXtkI == None:
            XXtkI = np.dot(self.X, self.X.T)
            index = np.arange(min(XXtkI.shape))
            XXtkI[index, index] += self.g.k
            invXXtkI = np.linalg.inv(XXtkI)
            self._XtinvXXtkI = np.dot(self.X.T, invXXtkI)

        beta = (Xty_grad - np.dot(self._XtinvXXtkI, np.dot(self.X, Xty_grad)))

        return beta

    def gap(self, beta, beta_hat):
        """Compute the duality gap.

        From the interface "DualFunction".
        """
        # TODO: Add this function!
        raise NotImplementedError("We cannot currently do this!")

    def estimate_mu(self, beta):
        """Computes a "good" value of \mu with respect to the given \beta.

        From the interface "NesterovFunction".
        """
        raise NotImplementedError("We do not use this here!")

    def M(self):
        """ The maximum value of the regularisation of the dual variable. We
        have

            M = max_{\alpha \in K} 0.5*|\alpha|²_2.

        From the interface "NesterovFunction".
        """
        return self.h.M()

    def project(self, a):
        """ Projection onto the compact space of the Nesterov function.

        From the interface "NesterovFunction".
        """
        return self.h.project(a)