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
import prox_op
import start_vector
import scheme
import mode
from multiblock.utils import MAX_ITER, TOLERANCE, make_list, dot, zeros, sqrt
from numpy import eye
from numpy.linalg import pinv

__all__ = ['BaseAlgorithm', 'NIPALSBaseAlgorithm', 'NIPALSAlgorithm',
           'RGCCAAlgorithm']


class BaseAlgorithm(object):
    """Baseclass for all multiblock (and single-block) algorithms.

    All algorithms, even the single-block algorithms, must return a list of
    weights.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, prox_op=prox_op.ProxOp(),
                 max_iter=MAX_ITER, tolerance=TOLERANCE,
                 **kwargs):
        """
        max_iter   : The number of iteration before the algorithm is forced
                     to stop. The default number of iterations is 500.

        tolerance  : The level below which we treat numbers as zero. This is
                     used as stop criterion in the algorithm. Smaller value
                     will give more acurate results, but will take longer
                     time to compute. The default tolerance is 5E-07.
        """

        self.prox_op = prox_op
        self.max_iter = max_iter
        self.tolerance = tolerance

    @abc.abstractmethod
    def run(self, *X, **kwargs):
        raise NotImplementedError('Abstract method "run" must be specialised!')


class NIPALSBaseAlgorithm(BaseAlgorithm):

    __metaclass__ = abc.ABCMeta

    def __init__(self, adj_matrix,
                 scheme=scheme.Horst(),
                 start_vector=start_vector.LargestStartVector(),
                 not_normed=False,
                 **kwargs):
        """
        Parameters:
        ----------
        adj_matrix : Adjacency matrix that is a numpy array of shape [n, n].
                     If an element in position adj_matrix[i,j] is 1, then block
                     i and j are connected, and 0 otherwise.
        scheme     : The inner weighting scheme to use in the algorithm. The
                     scheme may be an instance of Horst, Centroid or Factorial.
        """

        BaseAlgorithm.__init__(self, **kwargs)

        self.adj_matrix = adj_matrix
        self.scheme = scheme
        self.start_vector = start_vector
        self.not_normed = not_normed

    @abc.abstractmethod
    def run(self, *X, **kwargs):
        raise NotImplementedError('Abstract method "run" must be specialised!')


class NIPALSAlgorithm(NIPALSBaseAlgorithm):

    def __init__(self,
                 mode=mode.NewA(),
                 **kwargs):
        """
        Parameters:
        ----------
        mode       : A tuple or list with n instances of Mode with the mode to
                     use for a matrix.
        """

        NIPALSBaseAlgorithm.__init__(self, **kwargs)

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
        self.mode = make_list(self.mode, n, mode.NewA())
        self.scheme = make_list(self.scheme, n, scheme.Horst())
        self.not_normed = make_list(self.not_normed, n, False)

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
                    eij = self.scheme.compute(ti, tj)

                    # Internal estimation using connected matrices' scores
                    if self.adj_matrix[i, j] != 0 or \
                            self.adj_matrix[j, i] != 0:
                        ui += eij * tj

                # Outer estimation
                wi = self.mode.estimation(Xi, ui)

                # Apply proximal operator
                wi = self.prox_op.prox(wi, i)

                # Apply normalisation depending on the mode
                if not i in self.not_normed:
                    wi = self.mode.normalise(wi, Xi)

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

        NIPALSBaseAlgorithm.__init__(self, **kwargs)

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
        w          : A list with n numpy arrays of weights of shape [N_i, 1].
        """

        n = len(X)
        # TODO: Add Schäfer and Strimmer's method here if tau == None!
        self.tau = make_list(self.tau, n, 1)  # Default is New Mode A
        self.scheme = make_list(self.scheme, n, scheme.Horst())
        self.not_normed = make_list(self.not_normed, n, False)

        invIXX = []
        w = []
        for i in range(n):
            Xi = X[i]
            XX = dot(Xi.T, Xi)
            I = eye(XX.shape[0])

            w_ = self.start_vector.get_vector(Xi)
            invIXX.append(pinv(self.tau[i] * I + \
                    ((1 - self.tau[i]) / Xi.shape[0]) * XX))
            invIXXw = dot(invIXX[i], w_)
            winvIXXw = dot(w_.T, invIXXw)
            w_ = invIXXw / sqrt(winvIXXw)

            w.append(w_)

        # Main RGCCA loop
        self.iterations = 0
        while True:

            self.converged = True
            for i in xrange(n):
                Xi = X[i]
                ti = dot(Xi, w[i])
                ui = zeros(ti.shape)
                for j in xrange(n):
                    tj = dot(X[j], w[j])

                    # Determine scheme weights
                    eij = self.scheme.compute(ti, tj)

                    # Internal estimation using connected matrices' scores
                    if self.adj_matrix[i, j] != 0 or \
                            self.adj_matrix[j, i] != 0:
                        ui += eij * tj

                # Outer estimation for block i
                wi = dot(Xi.T, ui)
                wi = dot(invIXX[i], wi)

                # Apply proximal operator
                wi = self.prox_op.prox(wi, i)

                # Apply normalisation
                if not i in self.not_normed:
                    wi = wi / sqrt(dot(wi.T, dot(invIXX[i], wi)))

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


class ProximalGradientAlgorithm(BaseAlgorithm):
    """ Proximal gradient method.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, f, fgrad, **kwargs):
        """
        """

        self.f = f
        self.fgrad = fgrad

    @abc.abstractmethod
    def run(self, *X, **kwargs):
        raise NotImplementedError('Abstract method "run" must be specialised!')


class ISTA(ProximalGradientAlgorithm):
    """ ISTA algorithm.
    """

    def __init__(self, epsilon, **kwargs):
        ProximalGradientAlgorithm.__init__(self, **kwargs)

    def run(self, *X, **kwargs):

        for k in xrange(max_iter):
            beta = prox_op.prox(beta_old - t * fgrad(beta_old, X[0], X[1], l))
