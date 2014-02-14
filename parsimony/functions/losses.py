# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.functions.loss` module contains the loss functions used
throughout the package. These represent mathematical functions and should thus
have properties used by the corresponding algorithms. These properties are
defined in :mod:`parsimony.functions.interfaces`.

Loss functions should be stateless. Loss functions may be shared and copied
and should therefore not hold anything that cannot be recomputed the next time
it is called.

Created on Mon Apr 22 10:54:29 2013

@author:  Tommy Löfstedt, Vincent Guillemot, Edouard Duchesnay and
          Fouad Hadj-Selem
@email:   lofstedt.tommy@gmail.com, edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
import numpy as np

import interfaces
import parsimony.utils as utils

__all__ = ["RidgeRegression", "RidgeLogisticRegression", "AnonymousFunction"]


class RidgeRegression(interfaces.CompositeFunction,
                      interfaces.Gradient,
                      interfaces.LipschitzContinuousGradient,
                      interfaces.Eigenvalues,
                      interfaces.StronglyConvex):
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

    @utils.deprecated("StronglyConvex.parameter")
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

    def parameter(self):
        """Returns the strongly convex parameter for the function.

        From the interface "StronglyConvex".
        """
        if self._lambda_min is None:
            s = np.linalg.svd(self.X, full_matrices=False, compute_uv=False)

            self._lambda_max = np.max(s) ** 2.0

            if len(s) < self.X.shape[1]:
                self._lambda_min = 0.0
            else:
                self._lambda_min = np.min(s) ** 2.0

        return self._lambda_min + self.k


class RidgeLogisticRegression(interfaces.CompositeFunction,
                              interfaces.Gradient,
                              interfaces.LipschitzContinuousGradient):
    """ The Logistic Regression function.

    Ridge (re-weighted) log-likelihood (cross-entropy):
    * f(beta) = -Sum wi (yi log(pi) + (1 − yi) log(1 − pi)) + k/2 ||beta||^2_2
              = -Sum wi (yi xi' beta − log(1 + e(xi' beta))) + k/2 ||beta||^2_2

    * grad f(beta) = -Sum wi[ xi (yi - pi)] + k beta

    pi = p(y=1|xi, beta) = 1 / (1 + exp(-xi' beta))
    wi: sample i weight
    [Hastie 2009, p.: 102, 119 and 161, Bishop 2006 p.: 206]

    Parameters
    ----------
    X : Numpy array (n-by-p). The regressor matrix.

    y : Numpy array (n-by-1). The regressand vector.

    weights: Numpy array (n-by-1). The sample's weights.
    """
    def __init__(self, X, y, k=0, weights=None):
        self.X = X
        self.y = y
        self.k = float(k)
        if weights is None:
            # TODO: Make the weights sparse.
            #weights = np.eye(self.X.shape[0])
            weights = np.ones(y.shape).reshape(y.shape)
        # TODO: Allow the weight vector to be a list.
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
            PWX = 0.5 * np.sqrt(self.weights) * self.X  # TODO: CHECK WITH FOUAD
            # PW = 0.5 * np.eye(self.X.shape[0]) ## miss np.sqrt(self.W)
            #PW = 0.5 * np.sqrt(self.W)
            #PWX = np.dot(PW, self.X)
            # TODO: Use FastSVD for speedup!
            s = np.linalg.svd(PWX, full_matrices=False, compute_uv=False)
            self.lipschitz = np.max(s) ** 2.0 + self.k  # TODO: CHECK
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


class AnonymousFunction(interfaces.AtomicFunction):

    def __init__(self, f):

        self._f = f

    def f(self, x):

        return self._f(x)