# -*- coding: utf-8 -*-
"""
The :mod:`strukturerad.algorithms` module includes several algorithms for
minimising various functions.

#These algorithms all have in common that they maximise a criteria on the form
#
#    f(w_1, ..., w_n) = \sum_{i,j=1}^n c_{i,j} g(cov(X_iw_i, X_jw_j)),
#
#with possibly very different constraints put on the weights w_i or on the
#scores t_i = X_iw_i (e.g. unit 2-norm of weights, unit variance of scores,
#L1/LASSO constraint on the weights etc.).
#
#This includes models such as PCA (f(p) = cov(Xp, Xp)),
#PLS-R (f(w, c) = cov(Xw, Yc)), PLS-PM (the criteria above), RGCCA (the
#criteria above), etc.

Note:
----
Do not let algorithms be stateful. I.e. do not keep references to objects with
state in the algorithm objects. This could be e.g. output from the algorithm or
other things not related to the actual running of the algorithm (such as number
of iterations). It should be possible to copy and share algorithms between
models, and thus they should not depend on any state.

Created on Fri Feb  8 17:24:11 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: TBD
"""

import abc
import start_vectors
import utils
from utils import math
from utils import warning

import numpy as np
import scipy.sparse as sparse

__all__ = ['BaseAlgorithm',

           'NIPALSBaseAlgorithm',
           'SparseSVD', 'FastSVD',

           'ProximalGradientAlgorithm',
           'ISTA', 'FISTA', 'MonotoneFISTA',

           'ExcessiveGapAlgorithm',
           'ExcessiveGapRidgeRegression',

           'BisectionMethod', 'TernarySearch', 'GoldenSectionSearch']


class BaseAlgorithm(object):
    """Baseclass for all algorithms.

    Parameters:
    ----------
    max_iter   : The maximum number of iteration before the algorithm is
                 forced to stop. The default number of iterations is 500.

    tolerance  : The level below which we treat numbers as zero. This is
                 used as stop criterion in the algorithm. Smaller value
                 will give more acurate results, but will take longer
                 time to compute. The default tolerance is 5E-07.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, max_iter=utils.MAX_ITER, tolerance=utils.TOLERANCE,
                 **kwargs):

        super(BaseAlgorithm, self).__init__()

        self.set_max_iter(max_iter)
        self.set_tolerance(tolerance)

    def get_max_iter(self):

        return self.max_iter

    def set_max_iter(self, max_iter):

        self.max_iter = max_iter

    def get_tolerance(self):

        return self.tolerance

    def set_tolerance(self, tolerance):

        self.tolerance = tolerance

    @abc.abstractmethod
    def run(self, *X, **kwargs):

        raise NotImplementedError('Abstract method "run" must be specialised!')


class NIPALSBaseAlgorithm(BaseAlgorithm):

    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):

        super(NIPALSBaseAlgorithm, self).__init__(**kwargs)

    @abc.abstractmethod
    def run(self, *X, **kwargs):

        raise NotImplementedError('Abstract method "run" must be specialised!')


class SparseSVD(NIPALSBaseAlgorithm):
    """A kernel SVD implementation for sparse CSR matrices.

    This is usually faster than np.linalg.svd when density < 20% and when
    M << N or N << M (at least one order of magnitude). When M == N >= 10000 it
    is faster when the density < 1% and always faster regardless of density
    when M == N < 10000.

    These are ballpark estimates that may differ on your computer.
    """
    def __init__(self, max_iter=None, **kwargs):

        if max_iter == None:

            max_iter = 50

        super(SparseSVD, self).__init__(max_iter=max_iter, **kwargs)

    def run(self, X, **kwargs):
        """ Performs SVD of sparse matrices. This is faster than applying the
        general SVD.

        TODO: Do not save number of iterations in the class return this instead

        Arguments:
        ---------
        X : The matrix to decompose
        """
        M, N = X.shape
        p = start_vectors.RandomStartVector().get_vector(X)
        Xt = X.T
        if M < N:
            K = X.dot(Xt)
            t = X.dot(p)
            self.iterations = 0
            for it in xrange(self.max_iter):
                t_ = t
                t = K.dot(t_)
                t /= np.sqrt(np.sum(t_ ** 2.0))

                self.iterations += 1

                diff = t_ - t
                if (np.sum(diff ** 2.0)) < utils.TOLERANCE:
#                    print "broke at", self.iterations
                    break

            p = Xt.dot(t)
            normp = np.sqrt(np.sum(p ** 2.0))
            # Is the solution significantly different from zero (TOLERANCE)?
            if normp > utils.TOLERANCE:
                p /= normp
            else:
                p = np.ones(p.shape) / np.sqrt(p.shape[0])

        else:
            K = Xt.dot(X)
            self.iterations = 0
            for it in xrange(self.max_iter):
                p_ = p
                p = K.dot(p_)
                normp = np.sqrt(np.sum(p ** 2.0))
                if normp > utils.TOLERANCE:
                    p /= normp
                else:
                    p = np.ones(p.shape) / np.sqrt(p.shape[0])

                self.iterations += 1

                diff = p_ - p
                if (np.sum(diff ** 2.0)) < utils.TOLERANCE:
#                    print "broke at", self.iterations
                    break

#        t = X.dot(p)

#        sigma = numpy.sqrt(numpy.sum(t ** 2.0))
#        t /= sigma

        return p


class FastSVD(NIPALSBaseAlgorithm):
    """A kernel SVD implementation.

    This is always faster than np.linalg.svd. Particularly, this is a lot
    faster than np.linalg.svd when M << N or M >> N, for an M-by-N matrix.

    TODO: Do not save number of iterations in the class return this instead
    """
    def __init__(self, max_iter=None, start_vector=None, **kwargs):

        if max_iter == None:

            max_iter = 100

        super(FastSVD, self).__init__(max_iter=max_iter, **kwargs)

    def run(self, X, **kwargs):
        """ Performs SVD of given matrix. This is faster than applying
        np.linalg.svd.

        Arguments:
        ---------
        X         : The matrix to decompose

        Returns:
        -------
        v : The right singular vector.
        """
        M, N = X.shape
        if M < 80 and N < 80:  # Very arbitrary threshold for my computer ;-)
            _, _, V = np.linalg.svd(X, full_matrices=True)
            v = V[[0], :].T
        elif M < N:
            Xt = X.T
            K = np.dot(X, Xt)
            t = start_vectors.RandomStartVector().get_vector(Xt)
            self.iterations = 0
            for it in xrange(self.max_iter):
                t_ = t
                t = K.dot(t_)
                t /= np.sqrt(np.sum(t_ ** 2.0))

                self.iterations += 1

                diff = t_ - t
                if np.sqrt(np.sum(diff ** 2.0)) < utils.TOLERANCE:
#                    print "broke at", self.iterations
                    break

            v = np.dot(Xt, t)
            v /= np.sqrt(np.sum(v ** 2.0))

        else:
            Xt = X.T
            K = np.dot(Xt, X)
            v = start_vectors.RandomStartVector().get_vector(X)
            self.iterations = 0
            for it in xrange(self.max_iter):
                v_ = v
                v = np.dot(K, v_)
                v /= np.sqrt(np.sum(v ** 2.0))

                self.iterations += 1

                diff = v_ - v
                if np.sqrt(np.sum(diff ** 2.0)) < utils.TOLERANCE:
#                    print "broke at", self.iterations
                    break

        return v


class ProximalGradientAlgorithm(BaseAlgorithm):
    """ Proximal gradient method.

    Optimises a function decomposed as f(x) = g(x) + h(x), where f is convex
    and differentiable and h is convex.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):

        super(ProximalGradientAlgorithm, self).__init__(**kwargs)

    @abc.abstractmethod
    def run(self, beta, g, h, **kwargs):

        raise NotImplementedError('Abstract method "run" must be specialised!')


class ISTA(ProximalGradientAlgorithm):
    """ The Iterative Shrinkage-Thresholding Algorithm.
    """
    def __init__(self, *args, **kwargs):

        super(ISTA, self).__init__(*args, **kwargs)

    def run(self, beta, g, h, smooth=False, extended_output=False, **kwargs):

        tscale = 0.99
        t = tscale / g.Lipschitz()

        beta_old = beta
        beta_new = beta_old

        if extended_output or not smooth:
            f_new = g.f(beta_new, smooth=smooth) + h.f(beta_new)
            f_old = f_new
            f = []

        iterations = 0
        while True:
            converged = True

            beta_new = h.prox(beta_old - t * g.grad(beta_old), t)

            if math.norm1(beta_old - beta_new) > self.tolerance * t:
                converged = False

#            if extended_output or not smooth:
            f_new = g.f(beta_new, smooth=smooth) + h.f(beta_new)

#            if not smooth and f_new > f_old:  # Early stopping
#                converged = True
#                warning('Early stopping criterion triggered. Mu too large?')
#            else:
            # Save updated values
            if extended_output or not smooth:
                f.append(f_new)
                f_old = f_new
            iterations += 1

            beta_old = beta_new

            if converged:
                break

            if iterations >= self.max_iter:
                warning('Maximum number of iterations reached before ' \
                        'convergence')
                break

        if extended_output:
            output = {'f': f, 'iterations': iterations}
            return beta_new, output
        else:
            return beta_new


class FISTA(ISTA):
    """ The fast ISTA algorithm for regression.
    """
    def __init__(self, *args, **kwargs):

        super(FISTA, self).__init__(*args, **kwargs)

    def run(self, beta, g, h, smooth=False, extended_output=False, **kwargs):

        tscale = 0.99
        t = tscale / g.Lipschitz()

        beta_old = beta
        beta_new = beta_old

        if extended_output:
#            f_new = g.f(beta_new, smooth=smooth) + h.f(beta_new)
            f = []

        iterations = 0
        while True:
            converged = True

            k = float(iterations)
            z = beta_new + ((k - 2.0) / (k + 1.0)) * (beta_new - beta_old)
            beta_old = beta_new
            beta_new = h.prox(z - t * g.grad(z), t)

            if math.norm1(z - beta_new) > self.tolerance * t:
                converged = False

            if extended_output:
                f_new = g.f(beta_new, smooth=smooth) + h.f(beta_new)
                f.append(f_new)

            iterations += 1

            if converged:
                break

            if iterations >= self.max_iter:
                warning('Maximum number of iterations reached before ' \
                        'convergence')
                break

        if extended_output:
            output = {'f': f, 'iterations': iterations}
            return beta_new, output
        else:
            return beta_new


class MonotoneFISTA(ISTA):
    """ A monotonised version of the fast ISTA algorithm for regression.
    """
    def __init__(self, **kwargs):

        super(MonotoneFISTA, self).__init__(**kwargs)

    def run(self, beta, g, h, ista_steps=2, smooth=False,
            extended_output=False, **kwargs):

        tscale = 0.99
        t = tscale / g.Lipschitz()

        beta_old = beta
        beta_new = beta_old

        if extended_output or not smooth:
            f_new = g.f(beta_new, smooth=smooth) + h.f(beta_new)
            f = []

        iterations = 0
        while True:
            converged = True

            k = float(iterations)
            z = beta_new + ((k - 2.0) / (k + 1.0)) * (beta_new - beta_old)
            beta_old = beta_new
            beta_new = h.prox(z - t * g.grad(z), t)

            f_old = f_new
            f_new = g.f(beta_new, smooth=smooth) + h.f(beta_new)

            stop_early = False
            if f_new > f_old:  # FISTA increased the value of f

                beta_new = beta_old  # Go one step back to old beta
                for it in xrange(ista_steps):
                    beta_old = beta_new
                    beta_new = h.prox(beta_old - t * g.grad(beta_old), t)

                    if extended_output or not smooth:
                        f_old = f_new
                        f_new = g.f(beta_new, smooth=smooth) + h.f(beta_new)

#                    if early_stopping and f_new > f_old:  # Early stopping
                    if not smooth and f_new > f_old:  # Early stopping
                        converged = True
                        stop_early = True
                        print "f_new = %f > %f = f_old" % (f_new, f_old)
                        warning('Early stopping criterion triggered. ' \
                                'Mu too large?')
                        break  # Do not save this suboptimal point

                    if extended_output or not smooth:
                        f.append(f_new)

                    iterations += 1

                # If not early stopping, check ISTA convergence criterion
                if not stop_early \
                    and math.norm1(beta_old - beta_new) > self.tolerance * t:

                    converged = False

            else:  # The value of f decreased in the FISTA iteration
                if extended_output or not smooth:
                    f.append(f_new)

                iterations += 1

                if math.norm1(z - beta_new) > self.tolerance * t:
                    converged = False

            if converged:
                break

            if iterations >= self.max_iter:
                warning('Maximum number of iterations reached before ' \
                        'convergence')
                break

        if extended_output:
            output = {'f': f, 'iterations': iterations}
            return beta_new, output
        else:
            return beta_new


class ExcessiveGapAlgorithm(BaseAlgorithm):
    """ Baseclass for algorithms implementing the excessive gap method.

    Optimises a function decomposed as f(x) = g(x) + h(x), where g is
    strongly convex and twice differentiable and h is convex and Nesterov.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):

        super(ExcessiveGapAlgorithm, self).__init__(**kwargs)

    @abc.abstractmethod
    def run(self, *X, **kwargs):

        raise NotImplementedError('Abstract method "run" must be ' \
                                  'specialised!')


class ExcessiveGapRidgeRegression(ExcessiveGapAlgorithm):
    """ The excessive gap method for ridge regression.
    """
    # TODO: Do not save number of iterations and other variables in the
    # algorithm object
    def __init__(self, **kwargs):

        super(ExcessiveGapRidgeRegression, self).__init__(**kwargs)

    def run(self, X, y, g, h, **kwargs):

        self.l = g.l
        self.X = X
        self.y = y
        self.Xty = np.dot(self.X.T, self.y)

        # Used by Ridge regression
#        self.invXXI = np.linalg.inv(np.dot(self.X.T, self.X) \
#                                    + self.l * np.eye(self.X.shape[1]))

        # Used by Woodbury
        invXXtlI = np.linalg.inv(np.dot(self.X, self.X.T) \
                                    + self.l * np.eye(self.X.shape[0]))
        self.XtinvXXtlI = np.dot(self.X.T, invXXtlI)
        self.Aa = np.zeros((self.X.shape[1], 1))

        # TODO: Do not keep these references!
        self.A = h.A()
        self.At = h.At()
        A = sparse.vstack(self.A)
        v = SparseSVD(max_iter=100).run(A)
        u = A.dot(v)
        L = np.sum(u ** 2.0)
        del A
        L = L / g.lambda_min()  # Lipschitz constant
        print "Lipschitz constant:", L

        # Use the woodbury formulas to compute beta_hat
        _beta_hat_0 = self._beta_hat_0_2

        # Values for k=0
        mu = L / 1.0
        h.set_mu(mu)
        zero = [0] * len(self.A)
        for i in xrange(len(self.A)):
            zero[i] = np.zeros((self.A[i].shape[0], 1))
        beta_new = _beta_hat_0(zero)
        alpha = self._V(zero, beta_new, L)
        tau = 2.0 / 3.0
        alpha_hat = self._alpha_hat_muk(beta_new, h)
        u = [0] * len(alpha_hat)
        for i in xrange(len(alpha_hat)):
            u[i] = (1.0 - tau) * alpha[i] + tau * alpha_hat[i]

        f_new = g.f(beta_new) + h.f(beta_new)
        self.f = [f_new]
        self.iterations = 1
        while True:
            converged = True

            # Current iteration (compute for k+1)
            mu = (1.0 - tau) * mu
            h.set_mu(mu)
            beta_old = beta_new
            beta_new = (1.0 - tau) * beta_old + tau * _beta_hat_0(u)
            alpha = self._V(u, beta_old, L)

            if math.norm1(beta_new - beta_old) > self.tolerance:
                converged = False

            f_new = g.f(beta_new) + h.f(beta_new)
            self.f.append(f_new)
            self.iterations += 1

            if converged:
                break

            if self.iterations >= self.max_iter:
                warning('Maximum number of iterations reached before ' \
                        'convergence')
                break

            # Prepare for next iteration (next iteration's k)
            tau = 2.0 / (float(self.iterations) + 3.0)
            alpha_hat = self._alpha_hat_muk(beta_new, h)
            for i in xrange(len(alpha_hat)):
                u[i] = (1.0 - tau) * alpha[i] + tau * alpha_hat[i]

#        print "EGM Smooth: ", f_new
#        print "EGM True:   ", (self.g.f(beta_new, smooth=False) + self.h.f(beta_new, smooth=False))
#        print

        return beta_new

    def _V(self, u, beta, L):

        u_new = [0] * len(u)
        if L > utils.TOLERANCE:
            for i in xrange(len(u)):
                u_new[i] = u[i] + self.A[i].dot(beta) / L
        else:
            for i in xrange(len(u)):
                u_new[i] = np.ones(u[i].shape) * 9999999.9  # Large number <tm>
        return list(self.h.projection(*u_new))

    def _alpha_hat_muk(self, beta, h):

        return h.alpha(beta)

    def _beta_hat_0_1(self, alpha):
        """ Straight-forward naive Ridge regression.
        """
        self.Aa *= 0
        for i in xrange(len(alpha)):
            self.Aa += self.At[i].dot(alpha[i])
        v = self.Xty - self.Aa  # / 2.0

        return np.dot(self.invXXI, v)

    def _beta_hat_0_2(self, alpha):
        """ Ridge regression using the Woodbury formula.
        """
        self.Aa *= 0
        for i in xrange(len(alpha)):
            self.Aa += self.At[i].dot(alpha[i])
#        wk = (self.Xty - self.Aa / 2.0) / self.l
        wk = (self.Xty - self.Aa) / self.l

        return wk - np.dot(self.XtinvXXtlI, np.dot(self.X, wk))


class BisectionMethod(BaseAlgorithm):
    """Finds the root of a function f: R^n -> R lying on the line between
    x_0 and x_1.

    I.e. computes an x such that f(x) = 0.

    If no root exist on the line between x_0 and x_1, the result is
    undefined (but will be close to either x_0 or x_1).
    """
    def __init__(self, max_iter=100, tolerance=utils.TOLERANCE, **kwargs):
        """
        Parameters:
        ----------
        max_iter : The number of iteration before the algorithm is forced to
                stop. The default number of iterations is 100.

        tolerance : The level below which we treat numbers as zero. This is
                used as stopping criterion in the algorithm. Smaller value will
                give more acurate results, but will take longer time to
                compute. The default tolerance is utils.TOLERANCE.
        """
        super(BisectionMethod, self).__init__(max_iter=max_iter,
                                              tolerance=tolerance)

    def run(self, function, x_0, x_1):
        """
        Parameters:
        ----------
        function : The loss function for which the roots are found.

        x_0 : A variable for which f(x_0) < 0.

        x_1 : A variable for which f(x_1) > 0.
        """
        self.f = []
        self.iterations = 1
        while True:
            self.x = (x_1 + x_0) / 2.0
            f = function.f(self.x)
            if f < 0:
                x_0 = self.x
            if f > 0:
                x_1 = self.x

#            print "x:", self.x, "f:", f

            if np.sqrt(np.sum((x_1 - x_0) ** 2.0)) < self.tolerance:
                break

#            if abs(f) < self.tolerance:
#                break

            if self.iterations >= self.max_iter:
                break

            self.iterations += 1
            self.f.append(f)

        self.x = (x_1 + x_0) / 2.0
        self.f.append(function.f(self.x))

        return self.x


class TernarySearch(BaseAlgorithm):
    """Finds the minimum of a unimodal function f: R -> R using the Ternary
    search method.

    Implementation from: https://en.wikipedia.org/wiki/Ternary_search
    """
    def __init__(self, max_iter=100, tolerance=utils.TOLERANCE, **kwargs):
        """
        Parameters:
        ----------
        max_iter : The number of iteration before the algorithm is forced to
                stop. The default number of iterations is 100.

        tolerance : The level below which we treat numbers as zero. This is
                used as stopping criterion in the algorithm. A smaller value
                will give more acurate results, but will take a longer time to
                compute. The default tolerance is utils.TOLERANCE.
        """
        super(TernarySearch, self).__init__(max_iter=max_iter,
                                            tolerance=tolerance)

    def run(self, function, x_0, x_1):
        """
        Parameters:
        ----------
        function : The loss function to minimise.

        x_0 : A variable for which f(x_0) < f(x), where x is the optimum. Note
              that we must have x_0 < x_1.

        x_1 : A variable for which f(x_1) > f(x), where x is the optimum. Note
              that we must have x_0 < x_1.
        """

        x, it = self.ternary_search(x_0, x_1, 1, function)

        self.x = x
        self.iterations = it

        return self.x

    def ternary_search(self, left, right, it, function):

        # Left and right are the current bounds; the maximum is between them
        if (right - left) < self.tolerance:
            return (left + right) / 2.0, it

        if it >= self.max_iter:
            return (left + right) / 2.0, it

        leftThird = (2.0 * left + right) / 3.0
        rightThird = (left + 2.0 * right) / 3.0

        if function.f(leftThird) > function.f(rightThird):
            return self.ternary_search(leftThird, right, it + 1, function)
        else:
            return self.ternary_search(left, rightThird, it + 1, function)


class GoldenSectionSearch(BaseAlgorithm):
    """Finds the minimum of a unimodal function f: R -> R using the Golden
    section search method.

    Implementation from: https://en.wikipedia.org/wiki/Golden_section_search
    """
    def __init__(self, max_iter=100, tolerance=utils.TOLERANCE, **kwargs):
        """
        Parameters:
        ----------
        max_iter : The number of iteration before the algorithm is forced to
                stop. The default number of iterations is 100.

        tolerance : The level below which we treat numbers as zero. This is
                used as stopping criterion in the algorithm. A smaller value
                will give more acurate results, but will take a longer time to
                compute. The default tolerance is utils.TOLERANCE.
        """
        super(GoldenSectionSearch, self).__init__(max_iter=max_iter,
                                                  tolerance=tolerance)

        self.phi = (1.0 + np.sqrt(5)) / 2.0
        self.resphi = 2.0 - self.phi

    def run(self, function, x_0, x_1):
        """
        Parameters:
        ----------
        function : The loss function to minimise.

        x_0 : A variable for which f(x_0) < f(x), where x is the optimum. Note
              that we must have x_0 < x_1.

        x_1 : A variable for which f(x_1) > f(x), where x is the optimum. Note
              that we must have x_0 < x_1.
        """

        x, it = self.golden_section_search(x_0, (x_0 + x_1) / 2.0, x_1,
                                           self.tolerance, 1, function)
#                                           np.sqrt(self.tolerance), 1)

        self.x = x
        self.iterations = it

        return self.x

    def golden_section_search(self, a, b, c, tau, it, function):

        if c - b > b - a:
            x = b + self.resphi * (c - b)
        else:
            x = b - self.resphi * (b - a)

        if abs(c - a) < tau * (abs(b) + abs(x)):
#            print "First condition"
            return (c + a) / 2.0, it

        if abs(c - a) < tau ** 2.0:
#            print "Second condition"
            return (c + a) / 2.0, it

        if it >= self.max_iter:
            return (c + a) / 2.0, it

        if function.f(x) < function.f(b):
            if c - b > b - a:
                return self.golden_section_search(b, x, c, tau, it + 1, function)
            else:
                return self.golden_section_search(a, x, b, tau, it + 1, function)
        else:
            if c - b > b - a:
                return self.golden_section_search(a, b, x, tau, it + 1, function)
            else:
                return self.golden_section_search(x, b, c, tau, it + 1, function)