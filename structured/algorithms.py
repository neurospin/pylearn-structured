# -*- coding: utf-8 -*-
"""
The :mod:`multiblock.algorithms` module includes several projection
based latent variable algorithms.

These algorithms all have in common that they maximise a criteria on the form

    f(w_1, ..., w_n) = \sum_{i,j=1}^n c_{i,j} g(cov(X_iw_i, X_jw_j)),

with possibly very different constraints put on the weights w_i or on the
scores t_i = X_iw_i (e.g. unit 2-norm of weights, unit variance of scores,
L1/LASSO constraint on the weights etc.).

This includes models such as PCA (f(p) = cov(Xp, Xp)),
PLS-R (f(w, c) = cov(Xw, Yc)), PLS-PM (the criteria above), RGCCA (the
criteria above), etc.

Try to avoid making the algorithms stateful. I.e. do not keep references to
objects with state in the algorithm objects. It should be possible to copy and
share algorithms between models, and thus they should not depend on any state.

Created on Fri Feb  8 17:24:11 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: TBD
"""

import abc
import prox_ops
import start_vectors
import schemes
import modes
import loss_functions
import algorithms

from utils import MAX_ITER, TOLERANCE, make_list, zeros, sqrt
from utils import norm, norm1, warning

import numpy as np
from numpy import ones, eye
from numpy.linalg import pinv
import scipy.sparse as sparse
import copy

__all__ = ['BaseAlgorithm', 'SparseSVD', 'FastSVD', 'NIPALSBaseAlgorithm',
           'NIPALSAlgorithm', 'RGCCAAlgorithm', 'ISTARegression',
           'FISTARegression', 'MonotoneFISTARegression',
           'ExcessiveGapRidgeRegression',

           'BisectionMethod', 'TernarySearch', 'GoldenSectionSearch']


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
            start_vector = start_vectors.RandomStartVector()

        self.prox_op = prox_op
        self.start_vector = start_vector
        self.max_iter = max_iter
        self.tolerance = tolerance

    def _get_max_iter(self):
        return self.max_iter

    def _set_max_iter(self, max_iter):
        self.max_iter = max_iter

    def _get_tolerance(self):
        return self.tolerance

    def _set_tolerance(self, tolerance):
        self.tolerance = tolerance

    def _get_prox_op(self):
        return self.prox_op

    def _set_prox_op(self, prox_op):
        if not isinstance(prox_op, prox_ops.ProxOp):
            raise ValueError('The proximal operator must be an instance of ' \
                             '"ProxOp"')
        self.prox_op = prox_op

    def _get_start_vector(self):
        return self.start_vector

    def _set_start_vector(self, start_vector):
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

    def _get_adjacency_matrix(self):
        return self.adj_matrix

    def _set_adjacency_matrix(self, adj_matrix):
        try:
            adj_matrix = np.asarray(adj_matrix)
        except Exception:
            raise ValueError('The adjacency matrix must be a numpy array')
        if not adj_matrix.shape[0] == adj_matrix.shape[1]:
            raise ValueError('The adjacency matrix must be square')

        self.adj_matrix = adj_matrix

    def _set_scheme(self, scheme):
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


class SparseSVD(NIPALSBaseAlgorithm):
    """A kernel SVD implementation for sparse CSR matrices.

    This is usually faster than NIPALSAlgorithm when density < 20% and when
    M << N or N << M (at least one order of magnitude). When M == N >= 10000 it
    is faster when the density < 1% and always faster regardless of density
    when M == N < 10000.

    These are ballpark estimates that may differ on your computer.
    """

    def __init__(self, max_iter=None, start_vector=None, **kwargs):
        if max_iter == None:
            max_iter = 10
        if start_vector == None:
            start_vector = start_vectors.RandomStartVector()
        super(SparseSVD, self).__init__(max_iter=max_iter,
                                        start_vector=start_vector, **kwargs)

    def run(self, X, **kwargs):
        """ Performs SVD of sparse matrices. This is faster than applying the
        general SVD.

        Arguments:
        X : The matrix to decompose
        """
        M, N = X.shape
        p = self.start_vector.get_vector(X)
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
                if (np.sum(diff ** 2.0)) < TOLERANCE:
#                    print "broke at", self.iterations
                    break

            p = Xt.dot(t)
            normp = np.sqrt(np.sum(p ** 2.0))
            # Is the solution significantly different from zero (TOLERANCE)?
            if normp > TOLERANCE:
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
                if normp > TOLERANCE:
                    p /= normp
                else:
                    p = np.ones(p.shape) / np.sqrt(p.shape[0])

                self.iterations += 1

                diff = p_ - p
                if (np.sum(diff ** 2.0)) < TOLERANCE:
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
    """

    def __init__(self, max_iter=None, start_vector=None, **kwargs):
        if max_iter == None:
            max_iter = 100
        if start_vector == None:
            start_vector = start_vectors.RandomStartVector()
        super(FastSVD, self).__init__(max_iter=max_iter,
                                        start_vector=start_vector, **kwargs)

    def run(self, X, **kwargs):
        """ Performs SVD of given matrix. This is faster than applying
        np.linalg.svd.

        Arguments:
        X         : The matrix to decompose

        Returns:
        v : The right singular vector.
        """
        M, N = X.shape
        if M < 80 and N < 80:
            _, _, V = np.linalg.svd(X, full_matrices=True)
            v = V[[0], :].T
        elif M < N:
            Xt = X.T
            K = np.dot(X, Xt)
            t = self.start_vector.get_vector(Xt)
            self.iterations = 0
            for it in xrange(self.max_iter):
                t_ = t
                t = K.dot(t_)
                t /= np.sqrt(np.sum(t_ ** 2.0))

                self.iterations += 1

                diff = t_ - t
                if np.sqrt(np.sum(diff ** 2.0)) < TOLERANCE:
#                    print "broke at", self.iterations
                    break

            v = np.dot(Xt, t)
            v /= np.sqrt(np.sum(v ** 2.0))

        else:
            Xt = X.T
            K = np.dot(Xt, X)
            v = self.start_vector.get_vector(X)
            self.iterations = 0
            for it in xrange(self.max_iter):
                v_ = v
                v = np.dot(K, v_)
                v /= np.sqrt(np.sum(v ** 2.0))

                self.iterations += 1

                diff = v_ - v
                if np.sqrt(np.sum(diff ** 2.0)) < TOLERANCE:
#                    print "broke at", self.iterations
                    break

        return v


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
                ti = np.dot(Xi, w[i])
                ui = zeros(ti.shape)
                for j in xrange(n):
                    Xj = X[j]
                    wj = w[j]
                    tj = np.dot(Xj, wj)

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
                if np.dot(diff.T, diff) > self.tolerance:
                    self.converged = False

                # Save updated weight vector
                w[i] = wi

            self.iterations += 1

            if self.converged:
                break

            if self.iterations >= self.max_iter:
                warning('Maximum number of iterations reached before ' \
                        'convergence')
                break

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

        bias       : Whether or not to use a biased covariance estimation.
                     Default is False.

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

        bias = kwargs.pop('bias', False)
        ddof = 0.0 if bias else 1.0

        invIXX = []
        a = []
        for i in range(n):
            Xi = X[i]
            XX = np.dot(Xi.T, Xi)
            I = eye(XX.shape[0])

            a_ = self.start_vector.get_vector(Xi)
            invIXX.append(pinv(self.tau[i] * I + \
                    ((1.0 - self.tau[i]) / (Xi.shape[0] - ddof)) * XX))
            invIXXa = np.dot(invIXX[i], a_)
            ainvIXXa = np.dot(a_.T, invIXXa)
            a_ = invIXXa / sqrt(ainvIXXa)

            a.append(a_)

        # Main RGCCA loop
        self.iterations = 0
        while True:

            self.converged = True
            for i in xrange(n):
                Xi = X[i]
                Xai = np.dot(Xi, a[i])
                zi = zeros(Xai.shape)
                for j in xrange(n):
                    Xaj = np.dot(X[j], a[j])

                    # Determine scheme weights
                    eij = self.scheme[i].compute(Xai, Xaj)

                    # Internal estimation using connected matrices' scores
                    if self.adj_matrix[i, j] != 0 or \
                            self.adj_matrix[j, i] != 0:
                        zi += eij * Xaj

                # Outer estimation for block i
                Xz = np.dot(Xi.T, zi)
                ai = np.dot(invIXX[i], Xz)

                # Apply the proximal operator
                ai = self.prox_op.prox(ai, i)

                # Apply normalisation
                if not i in self.not_normed:
                    ai = ai / sqrt(np.dot(Xz.T, ai))

                # Check convergence for each weight vector. They all have to
                # leave converged = True in order for the algorithm to stop.
                diff = ai - a[i]
                if np.dot(diff.T, diff) > self.tolerance:
                    self.converged = False

                # Save updated weight vector
                a[i] = ai

            self.iterations += 1

            if self.converged:
                break

            if self.iterations >= self.max_iter:
                warning('Maximum number of iterations reached before ' \
                        'convergence')
                break

        return a


class ProximalGradientMethod(BaseAlgorithm):
    """ Proximal gradient method.

    Optimises a function decomposed as f(x) = g(x) + h(x), where f is convex
    and differentiable and h is convex.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, g=None, h=None, **kwargs):

        super(ProximalGradientMethod, self).__init__(**kwargs)

        if h == None:
            h = loss_functions.ZeroErrorFunction()

        self.g = g
        self.h = h

    @abc.abstractmethod
    def run(self, *X, **kwargs):
        raise NotImplementedError('Abstract method "run" must be specialised!')


class ISTARegression(ProximalGradientMethod):
    """ The ISTA algorithm for regression settings.
    """
    def __init__(self, g=None, h=None, **kwargs):

        super(ISTARegression, self).__init__(g, h, **kwargs)

    def run(self, X, y, t=None, tscale=0.95, smooth=False, **kwargs):

        if t == None:
            t = tscale / self.g.Lipschitz()
        else:
            t *= tscale

        beta_old = self.start_vector.get_vector(X)
        beta_new = beta_old
        f_new = self.g.f(beta_new, smooth=smooth) + self.h.f(beta_new)
        f_old = f_new
        self.f = [f_new]

        self.iterations = 1
        while True:
            self.converged = True

            beta_new = self.h.prox(beta_old - t * self.g.grad(beta_old), t)

            if norm1(beta_old - beta_new) > self.tolerance * t:
                self.converged = False

            f_new = self.g.f(beta_new, smooth=smooth) + self.h.f(beta_new)

            if not smooth and f_new > f_old:  # Early stopping
                self.converged = True
                warning('Early stopping criterion triggered. Mu too large?')
            else:
                # Save updated values
                f_old = f_new
                self.f.append(f_new)
                self.iterations += 1

            beta_old = beta_new

            if self.converged:
                break

            if self.iterations >= self.max_iter:
                warning('Maximum number of iterations reached before ' \
                        'convergence')
                break

        return beta_new


class FISTARegression(ISTARegression):
    """ The fast ISTA algorithm for regression.
    """
    def __init__(self, g=None, h=None, **kwargs):

        super(FISTARegression, self).__init__(g, h, **kwargs)

    def run(self, X, y, t=None, tscale=0.95, smooth=False, eval_f=True,
            **kwargs):

        if t == None:
            t = tscale / self.g.Lipschitz()
        else:
            t *= tscale

        beta_old = self.start_vector.get_vector(X)
        beta_new = beta_old

        if eval_f:
            f_new = self.g.f(beta_new, smooth=smooth) + self.h.f(beta_new)
            self.f = [f_new]

        self.iterations = 1
        while True:
            self.converged = True

            k = float(self.iterations)
            z = beta_new + ((k - 2.0) / (k + 1.0)) * (beta_new - beta_old)
            beta_old = beta_new
            beta_new = self.h.prox(z - t * self.g.grad(z), t)

            if norm1(z - beta_new) > self.tolerance * t:
                self.converged = False

            if eval_f:
                f_new = self.g.f(beta_new, smooth=smooth) + self.h.f(beta_new)

                self.f.append(f_new)

            self.iterations += 1

            if self.converged:
                break

            if self.iterations >= self.max_iter:
                warning('Maximum number of iterations reached before ' \
                        'convergence')
                break

        return beta_new


class MonotoneFISTARegression(ISTARegression):
    """ A monotonised version of the fast ISTA algorithm for regression.
    """
    def __init__(self, **kwargs):

        super(MonotoneFISTARegression, self).__init__(**kwargs)

    def run(self, X, y, t=None, tscale=0.95, ista_steps=2, smooth=False,
            early_stopping=True, **kwargs):

        if t == None:
            t = tscale / self.g.Lipschitz()
        else:
            t *= tscale

        beta_old = self.start_vector.get_vector(X)
        beta_new = beta_old

        f_new = self.g.f(beta_new, smooth=smooth) + self.h.f(beta_new)
        f_old = f_new
        self.f = [f_new]

        self.iterations = 1
        while True:
            self.converged = True

            k = float(self.iterations)
            z = beta_new + ((k - 2.0) / (k + 1.0)) * (beta_new - beta_old)
            beta_old = beta_new
            beta_new = self.h.prox(z - t * self.g.grad(z), t)

            f_old = f_new
            f_new = self.g.f(beta_new, smooth=smooth) + self.h.f(beta_new)

            stop_early = False
            if f_new > f_old:  # FISTA increased the value of f

                beta_new = beta_old  # Go one step back to old beta
                for it in xrange(ista_steps):
                    beta_old = beta_new
                    beta_new = self.h.prox(beta_old \
                                - t * self.g.grad(beta_old), t)

                    f_old = f_new
                    f_new = self.g.f(beta_new, smooth=smooth) \
                                + self.h.f(beta_new)

                    if early_stopping and f_new > f_old:  # Early stopping
                        self.converged = True
                        stop_early = True
                        warning('Early stopping criterion triggered. ' \
                                'Mu too large?')
                        break  # Do not save this suboptimal point

                    self.f.append(f_new)
                    self.iterations += 1

                # If not early stopping, check ISTA convergence criterion
                if not stop_early \
                        and norm1(beta_old - beta_new) > self.tolerance * t:
                    self.converged = False

            else:  # The value of f decreased in the FISTA iteration
                self.f.append(f_new)
                self.iterations += 1

                if norm1(z - beta_new) > self.tolerance * t:
                    self.converged = False

            if self.converged:
                break

            if self.iterations >= self.max_iter:
                warning('Maximum number of iterations reached before ' \
                        'convergence')
                break

        return beta_new


class ExcessiveGapMethod(BaseAlgorithm):
    """ Baseclass for algorithms implementing the excessive gap technique.

    Optimises a function decomposed as f(x) = g(x) + h(x), where g is
    strongly convex and twice differentiable and h is convex and Nesterov.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):

        super(ExcessiveGapMethod, self).__init__(**kwargs)

    @abc.abstractmethod
    def run(self, *X, **kwargs):
        raise NotImplementedError('Abstract method "run" must be ' \
                                  'specialised!')


class ExcessiveGapRidgeRegression(ExcessiveGapMethod):
    """ The excessive gap method for ridge regression.
    """

    def __init__(self, **kwargs):

        super(ExcessiveGapRidgeRegression, self).__init__(**kwargs)

    def run(self, X, y, **kwargs):

        self.l = self.g.l
        self.X = X
        self.y = y
        self.Xty = np.dot(self.X.T, self.y)

        # Used by Ridge regression
#        self.invXXI = np.linalg.inv(np.dot(self.X.T, self.X) \
#                                    + self.l * np.eye(self.X.shape[1]))

        # USed by Woodbury
        invXXtlI = np.linalg.inv(np.dot(self.X, self.X.T) \
                                    + self.l * np.eye(self.X.shape[0]))
        self.XtinvXXtlI = np.dot(self.X.T, invXXtlI)
        self.Aa = np.zeros((self.X.shape[1], 1))

        self.A = self.h.A()
        self.At = self.h.At()
        A = sparse.vstack(self.A)
        v = algorithms.SparseSVD(max_iter=100).run(A)
        u = A.dot(v)
        L = np.sum(u ** 2.0)
        del A
        L = L / self.g.lambda_min()  # Lipschitz constant
        print "Lipschitz constant:", L

        _beta_hat_0 = self._beta_hat_0_2

        # Values for k=0
        mu = L / 1.0
        self.h.set_mu(mu)
        zero = [0] * len(self.A)
        for i in xrange(len(self.A)):
            zero[i] = np.zeros((self.A[i].shape[0], 1))
        beta_new = _beta_hat_0(zero)
        alpha = self._V(zero, beta_new, L)
        tau = 2.0 / 3.0
        alpha_hat = self._alpha_hat_muk(beta_new)
        u = [0] * len(alpha_hat)
        for i in xrange(len(alpha_hat)):
            u[i] = (1.0 - tau) * alpha[i] + tau * alpha_hat[i]

        f_new = self.g.f(beta_new) + self.h.f(beta_new)
        self.f = [f_new]
        self.iterations = 1
        while True:
            self.converged = True

            # Current iteration (compute for k+1)
            mu = (1.0 - tau) * mu
            self.h.set_mu(mu)
            beta_old = beta_new
            beta_new = (1.0 - tau) * beta_old + tau * _beta_hat_0(u)
            alpha = self._V(u, beta_old, L)

            if norm1(beta_new - beta_old) > self.tolerance:
                self.converged = False

            f_new = self.g.f(beta_new) + self.h.f(beta_new)
            self.f.append(f_new)
            self.iterations += 1

            if self.converged:
                break

            if self.iterations >= self.max_iter:
                warning('Maximum number of iterations reached before ' \
                        'convergence')
                break

            # Prepare for next iteration (next iteration's k)
            tau = 2.0 / (float(self.iterations) + 3.0)
            alpha_hat = self._alpha_hat_muk(beta_new)
            for i in xrange(len(alpha_hat)):
                u[i] = (1.0 - tau) * alpha[i] + tau * alpha_hat[i]

#        print "EGM Smooth: ", f_new
#        print "EGM True:   ", (self.g.f(beta_new, smooth=False) + self.h.f(beta_new, smooth=False))
#        print

        return beta_new

    def _V(self, u, beta, L):

        u_new = [0] * len(u)
        if L > TOLERANCE:
            for i in xrange(len(u)):
                u_new[i] = u[i] + self.A[i].dot(beta) / L
        else:
            for i in xrange(len(u)):
                u_new[i] = np.ones(u[i].shape) * 1000000.0  # Large number <tm>
        return list(self.h.projection(*u_new))

    def _alpha_hat_muk(self, beta):

        return self.h.alpha(beta)

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
    def __init__(self, max_iter=100, tolerance=TOLERANCE, **kwargs):
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
    def __init__(self, max_iter=100, tolerance=TOLERANCE, **kwargs):
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
    def __init__(self, max_iter=100, tolerance=TOLERANCE, **kwargs):
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