# -*- coding: utf-8 -*-
"""
The :mod:`multiblock.algorithms` module includes several projection
based latent variable algorithms.

These algorithms all have in common that they maximise a criteria on the form

    f(w_1, ..., w_n) = \sum_{i,j=1}^n c_{i,j} g(cov(X_iw_i, X_jw_j)),

with possibly very different constraints put on the weights w_i or on the
scores t_i = X_iw_i (e.g. unit 2-norm of weights, unit variance of scores,
L1/LASSO constraint on the weights etc.).

This includes methods such as PCA (f(p) = cov(Xp, Xp)),
PLS-R (f(w, c) = cov(Xw, Yc)), PLS-PM (the criteria above), RGCCA (the
criteria above), etc.

Created on Fri Feb  8 17:24:11 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD Style
"""

import abc
import warnings
import prox_ops
import multiblock.start_vectors as start_vectors
import schemes
import modes
import error_functions

from multiblock.utils import MAX_ITER, TOLERANCE, make_list, dot, zeros, sqrt
from multiblock.utils import norm, norm1

import numpy
from numpy import ones, eye
from numpy.linalg import pinv

__all__ = ['BaseAlgorithm', 'NIPALSBaseAlgorithm', 'NIPALSAlgorithm',
           'RGCCAAlgorithm', 'ISTARegression', 'FISTARegression',
           'MonotoneFISTARegression']


class BaseAlgorithm(object):
    """Baseclass for all multiblock (and single-block) algorithms.

    All algorithms, even the single-block algorithms, must return a list of
    weights.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, prox_op=None, max_iter=MAX_ITER, tolerance=TOLERANCE,
                 start_vector=None, **kwargs):
        """
        max_iter   : The number of iteration before the algorithm is forced
                     to stop. The default number of iterations is 500.

        tolerance  : The level below which we treat numbers as zero. This is
                     used as stop criterion in the algorithm. Smaller value
                     will give more acurate results, but will take longer
                     time to compute. The default tolerance is 5E-07.
        """
        super(BaseAlgorithm, self).__init__()

        if prox_op == None:
            prox_op = prox_ops.ProxOp()
        if start_vector == None:
            start_vector = start_vectors.OnesStartVector()

        self.prox_op = prox_op
        self.start_vector = start_vector
        self.max_iter = max_iter
        self.tolerance = tolerance

    def set_max_iter(self, max_iter):
        self.max_iter = max_iter

    def set_tolerance(self, tolerance):
        self.tolerance = tolerance

    def get_prox_op(self):
        return self.prox_op

    def set_prox_op(self, prox_op):
        if not isinstance(prox_op, prox_ops.ProxOp):
            raise ValueError('The proximal operator must be an instance of ' \
                             '"ProxOp"')
        self.prox_op = prox_op

    def set_start_vector(self, start_vector):
        if not isinstance(start_vector, start_vectors.BaseStartVector):
            raise ValueError('The start vector must be an instance of ' \
                             '"BaseStartVector"')
        self.start_vector = start_vector

    @abc.abstractmethod
    def run(self, *X, **kwargs):
        raise NotImplementedError('Abstract method "run" must be specialised!')


class NIPALSBaseAlgorithm(BaseAlgorithm):

    __metaclass__ = abc.ABCMeta

    def __init__(self, adj_matrix=None, scheme=None, not_normed=[],
                 **kwargs):
        """
        Parameters:
        ----------
        adj_matrix : Adjacency matrix that is a numpy array of shape [n, n].
                     This matrix defines the path structure of the model. If an
                     element in position adj_matrix[i,j] is 1, then block i and
                     j are connected, and 0 otherwise. If this parameter is
                     omitted, all matrices are assumed to be connected.

        scheme     : The inner weighting scheme to use in the algorithm. The
                     scheme may be an instance of Horst, Centroid or Factorial.
        """

        super(NIPALSBaseAlgorithm, self).__init__(**kwargs)

        if scheme == None:
            scheme = schemes.Horst()

        self.adj_matrix = adj_matrix
        self.scheme = scheme
        self.not_normed = not_normed

    def set_adjacency_matrix(self, adj_matrix):
        try:
            adj_matrix = numpy.asarray(adj_matrix)
        except Exception:
            raise ValueError('The adjacency matrix must be a numpy array')
        if not adj_matrix.shape[0] == adj_matrix.shape[1]:
            raise ValueError('The adjacency matrix must be square')

        self.adj_matrix = adj_matrix

    def set_scheme(self, scheme):
        if isinstance(scheme, (tuple, list)):
            for s in scheme:
                if not isinstance(s, schemes.WeightingScheme):
                    raise ValueError('Argument "scheme" must be a list or ' \
                                     'tuple of WeightingSchemes')
        else:
            if not isinstance(scheme, schemes.WeightingScheme):
                raise ValueError('Argument "scheme" must be an instance of ' \
                                 'WeightingScheme')

        self.scheme = scheme

    @abc.abstractmethod
    def run(self, *X, **kwargs):
        raise NotImplementedError('Abstract method "run" must be specialised!')


class NIPALSAlgorithm(NIPALSBaseAlgorithm):

    def __init__(self, mode=None, **kwargs):
        """
        Parameters:
        ----------
        mode       : A tuple or list with n instances of Mode with the mode to
                     use for a matrix.
        """

        super(NIPALSAlgorithm, self).__init__(**kwargs)

        if mode == None:
            mode = modes.NewA()

        self.mode = mode

    def run(self, *X, **kwargs):
        """Inner loop of the NIPALS algorithm.

        Performs the NIPALS algorithm on the supplied tuple or list of numpy
        arrays in X. This method applies for 1, 2 or more blocks.

        One block could result in e.g. PCA; two blocks could result in e.g.
        SVD or CCA; and multiblock (n > 2) could be for instance GCCA,
        PLS-PM or MAXDIFF.

        This function uses Wold's procedure (based on Gauss-Siedel iteration)
        for fast convergence.

        Parameters
        ----------
        X          : A tuple or list with n numpy arrays of shape [M, N_i],
                     i=1,...,n. These matrices are the training set.

        Returns
        -------
        w          : A list with n numpy arrays of weights of shape [N_i, 1].
        """
        n = len(X)

        if self.adj_matrix == None:
            if n > 1:
                self.adj_matrix = ones((n, n)) - eye(n, n)
            else:
                self.adj_matrix = ones((1, 1))

        self.mode = make_list(self.mode, n, modes.NewA())
        self.scheme = make_list(self.scheme, n, schemes.Horst())

        w = []
        for Xi in X:
            w.append(self.start_vector.get_vector(Xi))

        # Main NIPALS loop
        self.iterations = 0
        while True:
            self.converged = True
            for i in xrange(n):
                Xi = X[i]
                ti = dot(Xi, w[i])
                ui = zeros(ti.shape)
                for j in xrange(n):
                    Xj = X[j]
                    wj = w[j]
                    tj = dot(Xj, wj)

                    # Determine scheme weights
                    eij = self.scheme[i].compute(ti, tj)

                    # Internal estimation using connected matrices' scores
                    if self.adj_matrix[i, j] != 0 or \
                            self.adj_matrix[j, i] != 0:
                        ui += eij * tj

                # Outer estimation
                wi = self.mode[i].estimation(Xi, ui)

                # Apply proximal operator
                wi = self.prox_op.prox(wi, i)

                # Apply normalisation depending on the mode
                if not i in self.not_normed:
                    wi = self.mode[i].normalise(wi, Xi)

                # Check convergence for each weight vector. They all have to
                # leave converged = True in order for the algorithm to stop.
                diff = wi - w[i]
                if dot(diff.T, diff) > self.tolerance:
                    self.converged = False

                # Save updated weight vector
                w[i] = wi

            if self.converged:
                break

            if self.iterations >= self.max_iter:
                warnings.warn('Maximum number of iterations reached before ' \
                              'convergence')
                break

            self.iterations += 1

        return w


class RGCCAAlgorithm(NIPALSBaseAlgorithm):

    def __init__(self, tau, **kwargs):
        """
        Parameters:
        ----------
        tau        : A tuple or list with n shrinkage constants tau[i]. If
                     tau is a single real, all matrices will use this value.
        """

        super(RGCCAAlgorithm, self).__init__(**kwargs)

        self.tau = tau

    def run(self, *X, **kwargs):
        """Inner loop of the RGCCA algorithm.

        Performs the RGCCA algorithm on the supplied tuple or list of numpy
        arrays in X. This method applies for 1, 2 or more blocks.

        One block would result in e.g. PCA; two blocks would result in e.g.
        SVD or CCA; and multiblock (n > 2) would be for instance SUMCOR,
        SSQCOR or SUMCOV.

        Parameters
        ----------
        X          : A tuple or list with n numpy arrays of shape [M, N_i],
                     i=1,...,n. These are the training set.

        Returns
        -------
        a          : A list with n numpy arrays of weights of shape [N_i, 1].
        """
        n = len(X)

        if self.adj_matrix == None:
            self.adj_matrix = ones((n, n)) - eye(n, n)

        # TODO: Add Schäfer and Strimmer's method here if tau == None!
        self.tau = make_list(self.tau, n, 1)  # Default 1, maximum covariance
        self.scheme = make_list(self.scheme, n, schemes.Horst())
#        self.not_normed = make_list(self.not_normed, n, False)

        invIXX = []
        a = []
        for i in range(n):
            Xi = X[i]
            XX = dot(Xi.T, Xi)
            I = eye(XX.shape[0])

            a_ = self.start_vector.get_vector(Xi)
            invIXX.append(pinv(self.tau[i] * I + \
                    ((1.0 - self.tau[i]) / (Xi.shape[0] - 0.0)) * XX))
            invIXXa = dot(invIXX[i], a_)
            ainvIXXa = dot(a_.T, invIXXa)
            a_ = invIXXa / sqrt(ainvIXXa)

            a.append(a_)

        # Main RGCCA loop
        self.iterations = 0
        while True:

            self.converged = True
            for i in xrange(n):
                Xi = X[i]
                Xai = dot(Xi, a[i])
                zi = zeros(Xai.shape)
                for j in xrange(n):
                    Xaj = dot(X[j], a[j])

                    # Determine scheme weights
                    eij = self.scheme[i].compute(Xai, Xaj)

                    # Internal estimation using connected matrices' scores
                    if self.adj_matrix[i, j] != 0 or \
                            self.adj_matrix[j, i] != 0:
                        zi += eij * Xaj

                # Outer estimation for block i
                Xz = dot(Xi.T, zi)
                ai = dot(invIXX[i], Xz)

                # Apply the proximal operator
                ai = self.prox_op.prox(ai, i)

                # Apply normalisation
                if not i in self.not_normed:
                    ai = ai / sqrt(dot(Xz.T, ai))

                # Check convergence for each weight vector. They all have to
                # leave converged = True in order for the algorithm to stop.
                diff = ai - a[i]
                if dot(diff.T, diff) > self.tolerance:
                    self.converged = False

                # Save updated weight vector
                a[i] = ai

            if self.converged:
                break

            if self.iterations >= self.max_iter:
                warnings.warn('Maximum number of iterations reached before ' \
                              'convergence')
                break

            self.iterations += 1

        return a


class ProximalGradientMethod(BaseAlgorithm):
    """ Proximal gradient method.

    Optimises a function decomposed as f(x) = g(x) + h(x), where f is convex
    and differentiable and h is convex.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        """
        """
        super(ProximalGradientMethod, self).__init__(**kwargs)

    @abc.abstractmethod
    def run(self, *X, **kwargs):
        raise NotImplementedError('Abstract method "run" must be specialised!')


class ISTARegression(ProximalGradientMethod):
    """ The ISTA algorithm for regression settings.
    """

    def __init__(self, **kwargs):

        super(ISTARegression, self).__init__(**kwargs)

    def run(self, X, y, g=None, h=None, t=None, tscale=0.95, **kwargs):

        if g == None:
            g = error_functions.MeanSquareRegressionError(X, y)
        if h == None:
            h = error_functions.ZeroErrorFunction()

        if not isinstance(g, error_functions.DifferentiableErrorFunction):
            raise ValueError('The functions in g must be ' \
                             'DifferentiableErrorFunctions')
        if not isinstance(g, error_functions.ConvexErrorFunction):
            raise ValueError('The functions in g must be ' \
                             'ConvexErrorFunction')
        if not isinstance(h, error_functions.ConvexErrorFunction):
            raise ValueError('The functions in h must be ConvexErrorFunction')

        if t == None:
            D, _ = numpy.linalg.eig(numpy.dot(X.T, X))
            t = tscale / numpy.max(D.real)

        beta = self.start_vector.get_vector(X)
        f_old = g.f(beta) + h.f(beta)
        print "f_old: ", f_old
#        print "f before:", self.f
        self.f = [f_old]

        self.iterations = 0
        while True:
            self.converged = True

            beta_ = h.prox(beta - t * g.grad(beta), t)

            if norm1(beta - beta_) > self.tolerance * t:
                self.converged = False

            # Save updated weight vector
            beta = beta_

            f_new = g.f(beta) + h.f(beta)
            self.f.append(f_new)
#            if abs(f_old - f_new) / f_old > self.tolerance:
#                self.converged = False
            f_old = f_new

            self.iterations += 1

            if self.converged:
                break

            if self.iterations >= self.max_iter:
                warnings.warn('Maximum number of iterations reached before ' \
                              'convergence')
                break

        return beta


class FISTARegression(ISTARegression):
    """ The fast ISTA algorithm for regression.
    """

    def __init__(self, **kwargs):

        super(FISTARegression, self).__init__(**kwargs)

    def run(self, X, y, g=None, h=None, t=None, tscale=0.95, **kwargs):

        if g == None:
            g = error_functions.MeanSquareRegressionError(X, y)
        if h == None:
            h = error_functions.ZeroErrorFunction()

        if not isinstance(g, error_functions.DifferentiableErrorFunction):
            raise ValueError('The functions in g must be ' \
                             'DifferentiableErrorFunctions')
        if not isinstance(g, error_functions.ConvexErrorFunction):
            raise ValueError('The functions in g must be ' \
                             'ConvexErrorFunction')
        if not isinstance(h, error_functions.ConvexErrorFunction):
            raise ValueError('The functions in h must be ConvexErrorFunction')

        if t == None:
            D, _ = numpy.linalg.eig(numpy.dot(X.T, X))
            t = tscale / numpy.max(D.real)

        beta = self.start_vector.get_vector(X)
        beta_ = beta
        f_old = g.f(beta) + h.f(beta)
        self.f = [f_old]

        self.iterations = 0
        while True:
            self.converged = True

            k = self.iterations + 1
            z = beta_ - ((k - 2) / (k + 1)) * (beta_ - beta)
            beta = beta_
            beta_ = h.prox(z - t * g.grad(z), t)

            if norm1(z - beta_) > self.tolerance * t:
                self.converged = False

            f_new = g.f(beta_) + h.f(beta_)
            self.f.append(f_new)
#            if abs(f_old - f_new) / f_old > self.tolerance:
#                self.converged = False
            f_old = f_new

            self.iterations += 1

            if self.converged:
                break

            if self.iterations >= self.max_iter:
                warnings.warn('Maximum number of iterations reached before ' \
                              'convergence')
                break

        return beta_


class MonotoneFISTARegression(ISTARegression):
    """ The fast ISTA algorithm for regression.
    """

    def __init__(self, **kwargs):

        super(MonotoneFISTARegression, self).__init__(**kwargs)

    def run(self, X, y, t=0.95, g=None, h=None, **kwargs):

        if g == None:
            g = error_functions.MeanSquareRegressionError(X, y)
        if h == None:
            h = error_functions.ZeroErrorFunction()

        if not isinstance(g, error_functions.DifferentiableErrorFunction):
            raise ValueError('The functions in g must be ' \
                             'DifferentiableErrorFunctions')
        if not isinstance(g, error_functions.ConvexErrorFunction):
            raise ValueError('The functions in g must be ' \
                             'ConvexErrorFunction')
        if not isinstance(h, error_functions.ConvexErrorFunction):
            raise ValueError('The functions in h must be ConvexErrorFunction')

        beta = self.start_vector.get_vector(X)
        beta_ = beta
        f_old = g.f(beta) + h.f(beta)
        self.f = [f_old]

        self.iterations = 0
        while True:
            self.converged = True

            k = self.iterations + 1
            z = beta_ - ((k - 2) / (k + 1)) * (beta_ - beta)
            beta = beta_
            beta_ = h.prox(z - t * g.grad(z), t)

            f_new = g.f(beta_) + h.f(beta_)
            if f_new > f_old:
                print "Two ISTA steps instead of FISTA!!"
                for it in xrange(2):
                    beta = beta_
                    beta_ = h.prox(beta - t * g.grad(beta), t)

                    f_old = f_new
                    f_new = g.f(beta_) + h.f(beta_)
                    self.f.append(f_new)
                    self.iterations += 1

                    assert(f_new < f_old)

                if norm1(beta - beta_) > max(1e-8, self.tolerance * t):
                    self.converged = False
            else:
                self.f.append(f_new)
                self.iterations += 1

                if norm1(z - beta_) > max(1e-8, self.tolerance * t):
                    self.converged = False

#            if abs(f_old - f_new) / f_old > self.tolerance:
#                self.converged = False
            f_old = f_new

            if self.converged:
                break

            if self.iterations >= self.max_iter:
                warnings.warn('Maximum number of iterations reached before ' \
                              'convergence')
                break

        return beta