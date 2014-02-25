# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.functions.losses` module contains the loss functions used
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

try:
    from . import interfaces  # Only works when imported as a package.
except ValueError:
    import parsimony.functions.interfaces as interfaces  # Run as a script
import parsimony.utils as utils

__all__ = ["LinearRegression", "RidgeRegression", "RidgeLogisticRegression"]


class LinearRegression(interfaces.CompositeFunction,
                       interfaces.Gradient,
                       interfaces.LipschitzContinuousGradient,
                       interfaces.StepSize):
    """The Linear regression loss function.
    """
    def __init__(self, X, y, mean=True):
        """
        Parameters
        ----------
        X : Numpy array (n-by-p). The regressor matrix.

        y : Numpy array (n-by-1). The regressand vector.

        k : Non-negative float. The ridge parameter.

        mean : Boolean. Whether to compute the squared loss or the mean
                squared loss. Default is True, the mean squared loss.
        """
        self.X = X
        self.y = y
        self.mean = bool(mean)

        self.reset()

    def reset(self):
        """Reset the value of _lambda_max and _lambda_min.

        From the interface "Function".
        """
        self._lambda_max = None

    def f(self, beta):
        """Function value.

        From the interface "Function".

        Parameters
        ----------
        beta : Numpy array. Regression coefficient vector. The point at which
                to evaluate the function.
        """
        if self.mean:
            d = 2.0 * float(self.X.shape[0])
        else:
            d = 2.0

        f = (1.0 / d) * np.sum((np.dot(self.X, beta) - self.y) ** 2.0)

        return f

    def grad(self, beta):
        """Gradient of the function at beta.

        From the interface "Gradient".

        Parameters
        ----------
        beta : The point at which to evaluate the gradient.

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.functions.losses import LinearRegression
        >>>
        >>> np.random.seed(42)
        >>> X = np.random.rand(100, 150)
        >>> y = np.random.rand(100, 1)
        >>> lr = LinearRegression(X=X, y=y)
        >>> beta = np.random.rand(150, 1)
        >>> np.linalg.norm(lr.grad(beta) - lr.approx_grad(beta, eps=1e-4))
        1.2935592057892195e-08
        """
        grad = np.dot(self.X.T, np.dot(self.X, beta) - self.y)

        if self.mean:
            grad /= float(self.X.shape[0])

        return grad

    def L(self):
        """Lipschitz constant of the gradient.

        From the interface "LipschitzContinuousGradient".
        """
        if self._lambda_max is None:

            from parsimony.algorithms.implicit import FastSVD

            # Rough limits for when FastSVD is faster than np.linalg.svd.
            n, p = self.X.shape
            if (max(n, p) > 500 and max(n, p) <= 1000 \
                    and float(max(n, p)) / min(n, p) <= 1.3) \
               or (max(n, p) > 1000 and max(n, p) <= 5000 \
                    and float(max(n, p)) / min(n, p) <= 5.0) \
               or (max(n, p) > 5000 and max(n, p) <= 10000 \
                       and float(max(n, p)) / min(n, p) <= 15.0) \
               or (max(n, p) > 10000 and max(n, p) <= 20000 \
                       and float(max(n, p)) / min(n, p) <= 200.0) \
               or max(n, p) > 10000:

                v = FastSVD().run(self.X, max_iter=1000)
                us = np.dot(self.X, v)
                self._lambda_max = np.sum(us ** 2.0)

            else:
                s = np.linalg.svd(self.X,
                                  full_matrices=False, compute_uv=False)
                self._lambda_max = np.max(s) ** 2.0

        return self._lambda_max

    def step(self, beta, index=0):
        """The step size to use in descent methods.

        Parameters
        ----------
        beta : Numpy array. The point at which to determine the step size.
        """
        return 1.0 / self.L()


class RidgeRegression(interfaces.CompositeFunction,
                      interfaces.Gradient,
                      interfaces.LipschitzContinuousGradient,
                      interfaces.Eigenvalues,
                      interfaces.StronglyConvex,
                      interfaces.StepSize):
    """ The Ridge Regression function

    Parameters
    ----------
    X : Numpy array (n-by-p). The regressor matrix.

    y : Numpy array (n-by-1). The regressand vector.

    k : Non-negative float. The ridge parameter.
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
        beta : Numpy array. The point at which to evaluate the gradient.

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.functions.losses import RidgeRegression
        >>>
        >>> np.random.seed(42)
        >>> X = np.random.rand(100, 150)
        >>> y = np.random.rand(100, 1)
        >>> rr = RidgeRegression(X=X, y=y, k=3.14159265)
        >>> beta = np.random.rand(150, 1)
        >>> np.linalg.norm(rr.grad(beta) - rr.approx_grad(beta, eps=1e-4))
        1.3403176569860683e-06
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
        return self.parameter()

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

    def step(self, beta, index=0):
        """The step size to use in descent methods.

        Parameters
        ----------
        beta : Numpy array. The point at which to determine the step size.
        """
        return 1.0 / self.L()


class RidgeLogisticRegression(interfaces.CompositeFunction,
                              interfaces.Gradient,
                              interfaces.LipschitzContinuousGradient):
    """The Logistic Regression loss function.

    Ridge (re-weighted) log-likelihood (cross-entropy):
    * f(beta) = -Sum wi (yi log(pi) + (1 − yi) log(1 − pi)) + k/2 ||beta||^2_2
              = -Sum wi (yi xi' beta − log(1 + e(xi' beta))) + k/2 ||beta||^2_2

    * grad f(beta) = -Sum wi[ xi (yi - pi)] + k beta

    pi = p(y=1|xi, beta) = 1 / (1 + exp(-xi' beta))
    wi: sample i weight
    [Hastie 2009, p.: 102, 119 and 161, Bishop 2006 p.: 206]
    """
    def __init__(self, X, y, k=0.0, weights=None):
        """
        Parameters
        ----------
        X : Numpy array (n-by-p). The regressor matrix.

        y : Numpy array (n-by-1). The regressand vector.

        k : Non-negative float. The ridge parameter.

        weights: Numpy array (n-by-1). The sample's weights.
        """
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
        self._lipschitz = None
#        self._lambda_max = None
#        self._lambda_min = None

    def f(self, beta):
        """Function value of Logistic regression at beta.

        Parameters
        ----------
        beta : Numpy array. Regression coefficient vector. The point at which
                to evaluate the function.
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
        beta : Numpy array. The point at which to evaluate the gradient.

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.functions.losses import RidgeLogisticRegression
        >>>
        >>> np.random.seed(42)
        >>> X = np.random.rand(100, 150)
        >>> y = np.random.rand(100, 1)
        >>> y[y < 0.5] = 0.0
        >>> y[y >= 0.5] = 1.0
        >>> rr = RidgeLogisticRegression(X=X, y=y, k=2.71828182)
        >>> beta = np.random.rand(150, 1)
        >>> np.linalg.norm(rr.grad(beta) - rr.approx_grad(beta, eps=1e-4))
        3.5290185882784444e-08
        """
        Xbeta = np.dot(self.X, beta)
        pi = 1.0 / (1.0 + np.exp(-Xbeta))
        return -np.dot(self.X.T, self.weights * (self.y - pi)) + self.k * beta
#        return -np.dot(self.X.T,
#                       np.dot(self.W, (self.y - pi))) \
#                       + self.k * beta

    def L(self):
        """Lipschitz constant of the gradient.

        Returns the maximum eigenvalue of (1 / 4) * X'WX.

        From the interface "LipschitzContinuousGradient".
        """
        if self._lipschitz == None:
            # pi(x) * (1 - pi(x)) <= 0.25 = 0.5 * 0.5
            PWX = 0.5 * np.sqrt(self.weights) * self.X  # TODO: CHECK WITH FOUAD
            # PW = 0.5 * np.eye(self.X.shape[0]) ## miss np.sqrt(self.W)
            #PW = 0.5 * np.sqrt(self.W)
            #PWX = np.dot(PW, self.X)
            # TODO: Use FastSVD for speedup!
            s = np.linalg.svd(PWX, full_matrices=False, compute_uv=False)
            self._lipschitz = np.max(s) ** 2.0 + self.k  # TODO: CHECK
        return self._lipschitz


#class AnonymousFunction(interfaces.AtomicFunction):
#
#    def __init__(self, f):
#
#        self._f = f
#
#    def f(self, x):
#
#        return self._f(x)


if __name__ == "__main__":
    import doctest
    doctest.testmod()