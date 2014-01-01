# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.functions` module contains several functions used
throughout the package. These represent mathematical functions and should thus
have properties used by the corresponding algorithms.

Loss functions should be stateless. Loss functions may be shared and copied and
should therefore not hold anything that cannot be recomputed the next time it
is called.

Created on Mon Apr 22 10:54:29 2013

@author:  Tommy Löfstedt, Vincent Guillemot, Edouard Duchesnay and \
Fouad Hadj Selem
@email:   tommy.loefstedt@cea.fr, edouard.duchesnay@cea.fr
@license: BSD 3-Clause
"""
import abc
import math
import numbers

import numpy as np
import scipy.sparse as sparse

import parsimony.utils.maths as maths
import parsimony.utils.consts as consts

__all__ = ['RidgeRegression', 'RidgeLogisticRegression', 'L1', 'SmoothedL1', 'TotalVariation',
           'SmoothedL1TV',
           'RR_L1_TV', 'RLR_L1_TV',
           'RR_SmoothedL1TV',
           'AnonymousFunction']


class Function(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def f(self, *args, **kwargs):
        """Function value.
        """
        raise NotImplementedError('Abstract method "f" must be '
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


class MultiblockFunction(CompositeFunction):
    """ This is a function that is the combination (i.e. sum) of other
    multiblock, composite or atomic functions. The difference from
    CompositeFunction is that this function assumes that relevant functions
    accept an index, i, that is the block we are working with.
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
        raise NotImplementedError('Abstract method "feasible" must be '
                                  'specialised!')


class ProximalOperator(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def prox(self, beta, factor=1.0):
        """The proximal operator corresponding to the function.
        """
        raise NotImplementedError('Abstract method "prox" must be '
                                  'specialised!')


class MultiblockProximalOperator(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def prox(self, beta, index, factor=1.0):
        """The proximal operator corresponding to the function with the index.
        """
        raise NotImplementedError('Abstract method "prox" must be '
                                  'specialised!')


class NesterovFunction(object):
    # TODO: We need a superclass for NesterovFunction wrappers.
    """
    Parameters
    ----------
    l : The Lagrange multiplier, or regularisation constant, of the function.

    c : Float. The limit of the constraint. The function is feasible if
            sqrt(x'Mx) <= c. The default value is c=0, i.e. the default is a
            regularisation formulation.

    A : The linear operator for the Nesterov formulation. May not be None!

    mu: The regularisation constant for the smoothing
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, l, c=0.0, A=None, mu=consts.TOLERANCE):

        self.l = float(l)
        self.c = float(c)
        self._A = A
        self.mu = float(mu)

    def fmu(self, beta, mu=None):
        """Returns the smoothed function value.

        Parameters
        ----------
        beta : A weight vector

        mu : The regularisation constant for the smoothing
        """
        if mu is None:
            mu = self.get_mu()

        alpha = self.alpha(beta)
        alpha_sqsum = 0.0
        for a in alpha:
            alpha_sqsum += np.sum(a ** 2.0)

        Aa = self.Aa(alpha)

        return self.l * ((np.dot(beta.T, Aa)[0, 0]
                          - (mu / 2.0) * alpha_sqsum) - self.c)

    def grad(self, beta):
        """ Gradient of the function at beta.

        Parameters
        ----------
        beta : The point at which to evaluate the gradient.
        """
        if self.l < consts.TOLERANCE:
            return 0.0

        alpha = self.alpha(beta)

        return self.l * self.Aa(alpha)

    def get_mu(self):
        """Return the regularisation constant for the smoothing.
        """
        return self.mu

    def set_mu(self, mu):
        """Set the regularisation constant for the smoothing.

        Parameters
        ----------
        mu: The regularisation constant for the smoothing to use from now on.

        Returns
        -------
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
        raise NotImplementedError('Abstract method "phi" must be '
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
        """ Compute A^\T\alpha.
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
        raise NotImplementedError('Abstract method "project" must be '
                                  'specialised!')

    @abc.abstractmethod
    def M(self):
        """ The maximum value of the regularisation of the dual variable. We
        have

            M = max_{\alpha \in K} 0.5*|\alpha|²_2.
        """
        raise NotImplementedError('Abstract method "M" must be '
                                  'specialised!')

    @abc.abstractmethod
    def estimate_mu(self, beta):
        """ Compute a "good" value of \mu with respect to the given \beta.
        """
        raise NotImplementedError('Abstract method "mu" must be '
                                  'specialised!')


class Continuation(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def mu_opt(self, eps):
        """The optimal value of \mu given \epsilon.
        """
        raise NotImplementedError('Abstract method "mu_opt" must be '
                                  'specialised!')

    @abc.abstractmethod
    def eps_opt(self, mu):
        """The optimal value of \epsilon given \mu.
        """
        raise NotImplementedError('Abstract method "eps_opt" must be '
                                  'specialised!')

    @abc.abstractmethod
    def eps_max(self, mu):
        """The maximum value of \epsilon.
        """
        raise NotImplementedError('Abstract method "eps_max" must be '
                                  'specialised!')


class Gradient(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def grad(self, beta):
        """Gradient of the function.

        Parameters
        ----------
        beta : The point at which to evaluate the gradient.
        """
        raise NotImplementedError('Abstract method "grad" must be '
                                  'specialised!')


class MultiblockGradient(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def grad(self, beta, index):
        """Gradient of the function.

        Parameters
        ----------
        beta : The point at which to evaluate the gradient.
        """
        raise NotImplementedError('Abstract method "grad" must be '
                                  'specialised!')


class Hessian(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def hessian(self, beta, vector=None):
        """The Hessian of the function.

        Arguments:
        ---------
        beta : The point at which to evaluate the Hessian.

        vector : If not None, it is multiplied with the Hessian from the right.
        """
        raise NotImplementedError('Abstract method "hessian" must be '
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
        raise NotImplementedError('Abstract method "hessian_inverse" must be '
                                  'specialised!')


class LipschitzContinuousGradient(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def L(self):
        """Lipschitz constant of the gradient.
        """
        raise NotImplementedError('Abstract method "L" must be '
                                  'specialised!')


class GradientStep(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def step(self, beta=None, index=0):
        """The step size to use in gradient descent.

        Arguments
        ---------
        beta : A weight vector. Optional, since some functions may determine
                the step without knowing beta.
        """
        raise NotImplementedError('Abstract method "step" must be '
                                  'specialised!')


class GradientMap(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def V(self, alpha, beta, L):
        """The gradient map associated to the function.
        """
        raise NotImplementedError('Abstract method "V" must be '
                                  'specialised!')


class DualFunction(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def gap(self, beta, beta_hat):
        """Compute the duality gap.
        """
        raise NotImplementedError('Abstract method "gap" must be '
                                  'specialised!')

    @abc.abstractmethod
    def betahat(self, alpha, beta=None):
        """Return the beta that minimises the dual function.
        """
        raise NotImplementedError('Abstract method "betahat" must be '
                                  'specialised!')


class Eigenvalues(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def lambda_max(self):
        """Largest eigenvalue of the corresponding covariance matrix.
        """
        raise NotImplementedError('Abstract method "lambda_max" must be '
                                  'specialised!')

    def lambda_min(self):
        """Smallest eigenvalue of the corresponding covariance matrix.
        """
        raise NotImplementedError('Abstract method "lambda_min" is not '
                                  'implemented!')


class AnonymousFunction(AtomicFunction):

    def __init__(self, f):

        self._f = f

    def f(self, x):

        return self._f(x)


class RidgeRegression(CompositeFunction, Gradient, LipschitzContinuousGradient,
                      Eigenvalues):
    """ The Ridge Regression function

    Parameters
    ----------
    X : Regressor

    y : Regressand

    k : Float. Ridge Regression weight coefficient
    """

    def __init__(self, X, y, k):

        self.X = X
        self.y = y
        self.k = float(k)

        self.reset()

    def reset(self):
        """Reset the value of _lambda_max and _lambda_min
        """

        self._lambda_max = None
        self._lambda_min = None

    def f(self, beta):
        """Function value of Ridge regression.

        Parameters
        ----------
        beta : Regression coefficient vector
        """
        return (1.0 / 2.0) * np.sum((np.dot(self.X, beta) - self.y) ** 2.0) \
             + (self.k / 2.0) * np.sum(beta ** 2.0)

    def grad(self, beta):
        """Gradient of the function at beta.

        From the interface "Gradient".

        Parameters
        ----------
        beta : The point at which to evaluate the gradient.
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
        if self._lambda_max is None:
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
        if self._lambda_min is None:
            s = np.linalg.svd(self.X, full_matrices=False, compute_uv=False)

            self._lambda_max = np.max(s) ** 2.0

            if len(s) < self.X.shape[1]:
                self._lambda_min = 0.0
            else:
                self._lambda_min = np.min(s) ** 2.0

        return self._lambda_min + self.k


class RidgeLogisticRegression(AtomicFunction, Gradient,
                         LipschitzContinuousGradient):#, Eigenvalues):
    """ The Logistic Regression function
    Ridge (re-weighted) log-likelihood (cross-entropy):
    * f(beta) = -log L(beta) + k/2 ||beta||^2_2
              = -Sum_i wi[yi log(pi) + (1−yi) log(1−pi)] + k/2||beta||^2_2
              = -Sum_i wi[yi xi'beta − log(1 + e(xi'beta) )] + k/2||beta||^2_2
    
    * grad f(beta) = -sum_i[ xi (yi - pi)] + k beta
    
    pi = p(y=1|xi,beta) = 1 / (1 + exp(-xi'beta))
    wi: sample i weight
    
    [Hastie 2009, p.: 102, 119 and 161, Bishop 2006 p.: 206]
    Parameters
    ----------
    X : Regressor

    y : Regressand

    weights : A vector with weights for each sample.
    """
    def __init__(self, X, y, k=0, weights=None):
        self.X = X
        self.y = y
        self.k = float(k)
        if weights is None:
            # TODO: Make the weights sparse.
            #weights = np.eye(self.X.shape[0])
            weights = np.ones(y.shape).reshape(y.shape)
        self.weights = weights

        self.reset()

    def reset(self):
        """Reset the value of _lambda_max and _lambda_min
        """
        self.lipschitz = None
#        self._lambda_max = None
#        self._lambda_min = None

    def f(self, beta):
        """Function value of Logistic regression at beta.

        Parameters
        ----------
        beta : Regression coefficient vector
        """
        # TODO check the correctness of the re-weighted loglike
        Xbeta = np.dot(self.X, beta)
        loglike = np.sum(self.weights * 
            ((self.y * Xbeta) - np.log(1 + np.exp(Xbeta))))
        return -loglike + (self.k / 2.0) * np.sum(beta ** 2.0)
#        n = self.X.shape[0]
#        s = 0
#        for i in xrange(n):
#            s = s + self.W[i, i] * (self.y[i, 0] * Xbeta[i, 0] \
#                                    - np.log(1 + np.exp(Xbeta[i, 0])))
#        return -s  + (self.k / 2.0) * np.sum(beta ** 2.0) ## TOCHECK


    def grad(self, beta):
        """Gradient of the function at beta.

        From the interface "Gradient".

        Parameters
        ----------
        beta : The point at which to evaluate the gradient.
        """
        Xbeta = np.dot(self.X, beta)
        pi = 1.0 / (1.0 + np.exp(-Xbeta))
        return -np.dot(self.X.T, self.weights * (self.y - pi)) + self.k * beta  
#        return -np.dot(self.X.T,
#                       np.dot(self.W, (self.y - pi))) \
#                       + self.k * beta

    def L(self):
        """Lipschitz constant of the gradient.

        From the interface "LipschitzContinuousGradient".
        max eigen value of (1/4 Xt W X)
        """
        if self.lipschitz == None:
            # pi(x) * (1 - pi(x)) <= 0.25 = 0.5 * 0.5
            PWX = 0.5 * np.sqrt(self.weights) * self.X  # TODO CHECK WITH FOUAD
            # PW = 0.5 * np.eye(self.X.shape[0]) ## miss np.sqrt(self.W)
            #PW = 0.5 * np.sqrt(self.W)
            #PWX = np.dot(PW, self.X)
            # TODO: Use FastSVD for speedup!
            s = np.linalg.svd(PWX, full_matrices=False, compute_uv=False)
            self.lipschitz = np.max(s) ** 2.0 + self.k  ## TOCHECK
        return self.lipschitz
#        return self.lambda_max()

#    def lambda_max(self):
#        """Largest eigenvalue of the corresponding covariance matrix.
#
#        From the interface "Eigenvalues".
#        """
#        if self._lambda_max is None:
#            s = np.linalg.svd(self.X, full_matrices=False, compute_uv=False)
#
#            self._lambda_max = np.max(s) ** 2.0
#
#            if len(s) < self.X.shape[1]:
#                self._lambda_min = 0.0
#            else:
#                self._lambda_min = np.min(s) ** 2.0
#
#        return self._lambda_max + self.k


class QuadraticConstraint(AtomicFunction, Gradient, Constraint):
    """The proximal operator of the quadratic function

        f(x) = l * (sqrt(x'Mx) - c),

    where M is a given positive definite matrix. The constrained version has
    the form

        sqrt(x'Mx) <= c.

    Parameters:
    ----------
    l : Float. The Lagrange multiplier, or regularisation constant, of the
            function.

    c : Float. The limit of the constraint. The function is feasible if
            sqrt(x'Mx) <= c. The default value is c=0, i.e. the default is a
            regularisation formulation.

    M : Array. The given positive definite matrix
    """
    def __init__(self, l=1.0, c=0.0, M=None):

        self.l = float(l)
        self.c = float(c)
        self.M = M

    def f(self, beta):
        """Function value.
        """
        return self.l * (np.sqrt(np.dot(beta.T, np.dot(self.M, beta)))
                         - self.c)

    def grad(self, beta):
        """Gradient of the function.

        From the interface "Gradient".
        """
        return self.l * np.dot(self.M, beta)

    def feasible(self, beta):
        """Feasibility of the constraint.

        From the interface "Constraint".
        """
        return np.sqrt(np.dot(beta.T, np.dot(self.M, beta))) <= self.c


class RGCCAConstraint(QuadraticConstraint):
    """The proximal operator of the quadratic function

        f(x) = l * (sqrt(x'(\tau * I + ((1 - \tau) / n) * X'X)x) - c),

    where \tau is a given regularisation constant. The constrained version has
    the form

        sqrt(x'(\tau * I + ((1 - \tau) / n) * X'X)x) <= c.

    Parameters:
    ----------
    l : Float. The Lagrange multiplier, or regularisation constant, of the
            function.

    c : Float. The limit of the constraint. The function is feasible if
            sqrt(x'(\tau * I + ((1 - \tau) / n) * X'X)x) <= c. The default
            value is c=0, i.e. the default is a regularisation formulation.

    tau : Float. Given regularisation constant

    unbiased : Boolean.
    """
    def __init__(self, l=1.0, c=0.0, tau=1.0, X=None, unbiased=True):

        self.l = float(l)
        self.c = float(c)
        self.tau = max(0.0, min(float(tau), 1.0))
        self.X = X
        self.unbiased = unbiased

    def f(self, beta):
        """Function value.
        """
        xtMx = self._compute_value(beta)
        return self.l * (np.sqrt(xtMx) - self.c)

    def grad(self, beta):
        """Gradient of the function.

        From the interface "Gradient".
        """
        if self.unbiased:
            n = self.X.shape[0] - 1.0
        else:
            n = self.X.shape[0]

        XtXbeta = np.dot(self.X.T, np.dot(self.X, beta))
        grad = self.tau * beta \
             + ((1.0 - self.tau) / float(n)) * XtXbeta

        return grad

    def feasible(self, beta):
        """Feasibility of the constraint.

        From the interface "Constraint".
        """
        xtMx = self._compute_value(beta)
        return np.sqrt(xtMx) <= self.c

    def _compute_value(self, beta):

        if self.unbiased:
            n = self.X.shape[0] - 1.0
        else:
            n = self.X.shape[0]
        Xbeta = np.dot(self.X, beta)
        val = self.tau * np.dot(beta.T, beta) \
            + ((1.0 - self.tau) / float(n)) * np.dot(Xbeta.T, Xbeta)
        return val


class L1(AtomicFunction, Constraint, ProximalOperator):
    """The proximal operator of the L1 function

        f(\beta) = l * (||\beta||_1 - c),

    where ||\beta||_1 is the L1 loss function. The constrained version has the
    form

        ||\beta||_1 <= c.

    Parameters:
    ----------
    l : The Lagrange multiplier, or regularisation constant, of the function.

    c : The limit of the constraint. The function is feasible if
            ||\beta||_1 <= c. The default value is c=0, i.e. the default is a
            regularisation formulation.
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
        return maths.norm1(beta) <= self.c


class SmoothedL1(AtomicFunction, Constraint, NesterovFunction, Gradient,
                 LipschitzContinuousGradient):
    """The proximal operator of the smoothed L1 function

        f(\beta) = l * (L1mu(\beta) - c),

    where L1mu(\beta) is the smoothed L1 function. The constrained version has
    the form

        ||\beta||_1 <= c.

    Parameters
    ----------
    l : The Lagrange multiplier, or regularisation constant, of the function.

    c : The limit of the constraint. The function is feasible if
            ||\beta||_1 <= c. The default value is c=0, i.e. the default is a
            regularisation formulation.

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

    def feasible(self, beta):
        """Feasibility of the constraint.

        From the interface "Constraint".
        """
        return maths.norm1(beta) <= self.c

    def phi(self, alpha, beta):
        """ Function value with known alpha.

        From the interface "NesterovFunction".
        """
        if self.l < consts.TOLERANCE:
            return 0.0

        return self.l * ((np.dot(alpha[0].T, beta)[0, 0]
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

        TV(\beta) <= c.

    Parameters
    ----------
    l : The Lagrange multiplier, or regularisation constant, of the function.

    c : The limit of the constraint. The function is feasible if
            TV(\beta) <= c. The default value is c=0, i.e. the default is a
            regularisation formulation.

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
        return self.l * (np.sum(np.sqrt(A[0].dot(beta) ** 2.0 +
                                        A[1].dot(beta) ** 2.0 +
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

        return self.l * ((np.dot(beta.T, Aa)[0, 0]
                          - (self.mu / 2.0) * alpha_sqsum) - self.c)

    def feasible(self, beta):
        """Feasibility of the constraint.

        From the interface "Constraint".
        """
        A = self.A()
        val = np.sum(np.sqrt(A[0].dot(beta) ** 2.0 +
                             A[1].dot(beta) ** 2.0 +
                             A[2].dot(beta) ** 2.0))
        return val <= self.c

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
            self._lambda_max = 2.0 * (1.0 - math.cos(float(self._p - 1)
                                                     * math.pi
                                                     / float(self._p)))

        elif self._lambda_max is None:

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


class RR_L1_TV(CompositeFunction, Gradient, LipschitzContinuousGradient,
               ProximalOperator, NesterovFunction, Continuation,
               DualFunction):
    """Combination (sum) of RidgeRegression, L1 and TotalVariation

    Parameters
    ----------
    X : Ridge Regression parameter.

    y : Ridge Regression parameter.

    k : Ridge Regression parameter.

    l : L1 parameter.
            The Lagrange multiplier, or regularisation constant, of the
            function.

    g : Total Variation parameter
            The Lagrange multiplier, or regularisation constant, of the
            function.

    A : Total Variation parameter.
            The linear operator for the Nesterov formulation. May not be None!

    mu : Total Variation parameter.
            The regularisation constant for the smoothing.
    """

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

        Parameters
        ----------
        mu: The regularisation constant for the smoothing to use from now on.

        Returns
        -------
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

        return (-gM * gA2 + np.sqrt((gM * gA2) ** 2.0
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

        return (2.0 * gM * gA2 * mu
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
        if self._Xty is None:
            self._Xty = np.dot(self.X.T, self.y)

        Ata_tv = self.tv.l * self.tv.Aa(alphak)
        Ata_l1 = self.l1.l * SmoothedL1.project([betak / consts.TOLERANCE])[0]
        v = (self._Xty - Ata_tv - Ata_l1)

        shape = self.X.shape

        if shape[0] > shape[1]:  # If n > p

            # Ridge solution
            if self._invXXkI is None:
                XtXkI = np.dot(self.X.T, self.X)
                index = np.arange(min(XtXkI.shape))
                XtXkI[index, index] += self.rr.k
                self._invXXkI = np.linalg.inv(XtXkI)

            beta_hat = np.dot(self._invXXkI, v)

        else:  # If p > n
            # Ridge solution using the Woodbury matrix identity:
            if self._XtinvXXtkI is None:
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


class RLR_L1_TV(RR_L1_TV):
    """Combination (sum) of RidgeLogisticRegression, L1 and TotalVariation

    Parameters
    ----------
    X : Ridge Regression parameter.

    y : Ridge Regression parameter.

    k : Ridge Regression parameter.

    l : L1 parameter.
            The Lagrange multiplier, or regularisation constant, of the
            function.

    g : Total Variation parameter
            The Lagrange multiplier, or regularisation constant, of the
            function.

    A : Total Variation parameter.
            The linear operator for the Nesterov formulation. May not be None!

    mu : Total Variation parameter.
            The regularisation constant for the smoothing.
    """

    def __init__(self, X, y, k, l, g, A=None, mu=0.0):

        self.X = X
        self.y = y

        self.rr = RidgeLogisticRegression(X, y, k)
        self.l1 = L1(l)
        self.tv = TotalVariation(g, A=A, mu=0.0)

        self.reset()


class SmoothedL1TV(AtomicFunction, Regularisation, NesterovFunction,
                   Eigenvalues):
    """
    Parameters
    ----------
    l : L1 parameter.
            The Lagrange multiplier, or regularisation constant, of the
            function.

    g : Total Variation parameter
            The Lagrange multiplier, or regularisation constant, of the
            function.

    Atv : The linear operator for the total variation Nesterov function

    Al1 : Matrix allocation for regression

    mu: The regularisation constant for the smoothing
    """
    def __init__(self, l, g, Atv=None, Al1=None, mu=0.0):

        self.l = float(l)
        self.g = float(g)

        self._p = Atv[0].shape[1]  # WARNING: Number of rows may differ from p.
        if Al1 is None:
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
               np.sum(np.sqrt(A[1].dot(beta) ** 2.0 +
                              A[2].dot(beta) ** 2.0 +
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
            lmaxTV = 2.0 * (1.0 - math.cos(float(p - 1) * math.pi
                                           / float(p)))
            self._lambda_max = lmaxTV * self.g ** 2.0 + self.l ** 2.0

        elif self._lambda_max is None:

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
    """
    Parameters
    ----------
    X : Ridge Regression parameter.

    y : Ridge Regression parameter.

    k : Ridge Regression parameter.

    l : L1 parameter.
            The Lagrange multiplier, or regularisation constant, of the
            function.

    g : Total Variation parameter.
            The Lagrange multiplier, or regularisation constant, of the
            function.

    Atv : The linear operator for the total variation Nesterov function.

    Al1 : Matrix allocation for regression.

    mu: The regularisation constant for the smoothing.
    """
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

        Parameters
        ----------
        mu: The regularisation constant for the smoothing to use from now on.

        Returns
        -------
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

        if self._Xy is None:
            self._Xy = np.dot(self.X.T, self.y)

        Xty_grad = (self._Xy - grad) / self.g.k

#        t = time()
#        XXkI = np.dot(X.T, X)
#        index = np.arange(min(XXkI.shape))
#        XXkI[index, index] += self.g.k
#        invXXkI = np.linalg.inv(XXkI)
#        print "t:", time() - t
#        beta = np.dot(invXXkI, Xty_grad)

        if self._XtinvXXtkI is None:
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


class LatentVariableCovariance(MultiblockFunction, MultiblockGradient):

    def __init__(self, X, unbiased=True):

        self.X = X
        if unbiased:
            self.n = X[0].shape[0] - 1.0
        else:
            self.n = X[0].shape[0]

#        self.reset()
#
#    def reset(self):
#        pass

    def f(self, w):
        """Function value.

        From the interface "Function".
        """
        wX = np.dot(self.X[0], w[0]).T
        Yc = np.dot(self.X[1], w[1])
        return np.dot(wX, Yc) / float(self.n)

    def grad(self, w, index):
        """Gradient of the function.

        From the interface "MultiblockGradient".
        """
        index = int(index)
        return np.dot(self.X[index].T,
                      np.dot(self.X[1 - index], w[1 - index])) \
             / float(self.n)


class GeneralisedMultiblock(MultiblockFunction, MultiblockGradient,
                            MultiblockProximalOperator, GradientStep,
#                            LipschitzContinuousGradient,
#                            NesterovFunction, Continuation, DualFunction
                            ):

    def __init__(self, X, functions):

        self.X = X
        self.functions = functions

        self.reset()

    def reset(self):

        for i in xrange(len(self.functions)):
            for j in xrange(len(self.functions[i])):
                if i == j:
                    for k in xrange(len(self.functions[i][j])):
                        self.functions[i][j][k].reset()
                else:
                    self.functions[i][j].reset()

    def f(self, w):
        """Function value.
        """
        val = 0.0
        for i in xrange(len(self.functions)):
            fi = self.functions[i]
            for j in xrange(len(fi)):
                fij = fi[j]
                if i == j and isinstance(fij, (list, tuple)):
                    for k in xrange(len(fij)):
#                        print "Diag: ", i
                        val += fij[k].f(w[i])
                else:
#                    print "f(w[%d], w[%d])" % (i, j)
                    val += fij.f([w[i], w[j]])

        # TODO: Check instead if it is a numpy array.
        if not isinstance(val, numbers.Number):
            return val[0, 0]
        else:
            return val

    def grad(self, w, index):
        """Gradient of the differentiable part of the function.

        From the interface "MultiblockGradient".
        """
        grad = 0.0
        fi = self.functions[index]
        for j in xrange(len(fi)):
            fij = fi[j]
            if index != j:
                if isinstance(fij, Gradient):
                    grad += fij.grad(w[index])
                elif isinstance(fij, MultiblockGradient):
                    grad += fij.grad([w[index], w[j]], 0)

        for i in xrange(len(self.functions)):
            fij = self.functions[i][index]
            if i != index:
                if isinstance(fij, Gradient):
                    # We shouldn't do anything here, right? This means e.g.
                    # that this (block i) is the y of a logistic regression.
                    pass
#                    grad += fij.grad(w)
                elif isinstance(fij, MultiblockGradient):
                    grad += fij.grad([w[i], w[index]], 1)

        fii = self.functions[index][index]
        for k in xrange(len(fii)):
            if isinstance(fii[k], Gradient):
                grad += fii[k].grad(w[index])

        return grad

    def prox(self, w, index, factor=1.0):
        """The proximal operator corresponding to the function with the index.

        From the interface "MultiblockProximalOperator".
        """
        # Find a proximal operator.
        fii = self.functions[index][index]
        for k in xrange(len(fii)):
            if isinstance(fii[k], ProximalOperator):
                w[index] = fii[k].prox(w[index], factor)
                break
        # If no proximal operator was found, we will just return the same
        # vectors again. The proximal operator of the zero function returns the
        # vector itself.

        return w

    def step(self, w, index):
        return 0.01  # TODO: Fix this!! Add backtracking?
