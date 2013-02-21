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

#import abc
import warnings
import numpy as np
import prox_op
from multiblock.utils import *


#__all__= ['NIPALS_PCA', 'NIPALS', 'RGCCA',
#          'NEWA', 'A', 'B', 'HORST', 'CENTROID', 'FACTORIAL']
__all__= ['NIPALS', 'RGCCA',
          'NEWA', 'A', 'B', 'HORST', 'CENTROID', 'FACTORIAL']


# NIPALS mode
NEWA       = "NewA"
A          = "A"
B          = "B"


# Inner weighting schemes
HORST      = "Horst"
CENTROID   = "Centroid"
FACTORIAL  = "Factorial"



def _start_vector(X, random = True, ones = False, largest = False):
    if largest: # Using row with largest sum of squares
        idx = np.argmax(np.sum(X**2, axis=1))
        w = X[[idx],:].T
    elif ones:
        w = np.ones((X.shape[1],1))
    else: # random start vector
        w = np.random.rand(X.shape[1],1)

    w /= norm(w)
    return w



#class BaseAlgorithm(object):
#    """Baseclass for all multiblock (and single-block) algorithms.
#    
#    All algorithmsm, even the single-block algorithms, must return a list
#    of weights.
#    """
#
#    __metaclass__ = abc.ABCMeta
#
#    def __init__(prox_op = prox_op.ProxOp(),
#                 max_iter = MAX_ITER, tolerance = TOLERANCE, **kwargs):
#
#        self.prox_op   = prox_op
#        self.max_iter  = max_iter
#        self.tolerance = tolerance
#
#    @abc.abstractmethod
#    def run(self, X, *args, **kwargs):
#        raise NotImplementedError('Abstract method "run" must be specialised!')



#class NIPALS_PCA(BaseAlgorithm):
#
#    def __init__(**kwargs):
#        BaseAlgorithm.__init__(self, **kwargs)
#
#def NIPALS_PCA(X, prox_op = prox_op.ProxOp(),
#               max_iter = MAX_ITER, tolerance = TOLERANCE, **kwargs):
#
#    if isinstance(X, (tuple, list)):
#        X = X[0]
#
#    (r, c) = X.shape
#
#    print tolerance
#
#    if r < c:
#        p = _start_vector(X, largest = True)
#        p /= norm(p)
#
#        iterations = 0
#        while True:
#            t  = dot(X, p)
#            p_ = dot(X.T, t) / dot(t.T, t)
#
#            p_ = prox_op.prox(p_)
#
#            p_ = p_ / norm(p_)
#
#            diff = p - p_
#            p = p_
#            if dot(diff.T, diff) < tolerance:
#                break
#    
#            iterations += 1
#            if iterations >= max_iter:
#                warnings.warn('Maximum number of iterations reached '
#                              'before convergence')
#                break
#    else:
##        if r < c: # Dual case
#    #        warnings.warn("No sparsity for the dual case!")
#    #        XX = dot(X, X.T)
##        else: # Primal case
#        XX = dot(X.T, X)
#
#        p = _start_vector(X, largest = True)
##        if r < c:
##            p = dot(X, p)
##            p /= np.sqrt(dot(p.T, dot(XX, p)))
#
#        iterations = 0
#        while True:
#
#            p_ = dot(XX, p)
#
##            if r < c:
##    #            p_ /= norm(dot(X.T, p_))
##                p_ /= np.sqrt(dot(p_.T, dot(XX, p_)))
##            else:
#            p_ = prox_op.prox(p_)
#            p_ = p_ / norm(p_)
#
#            diff = p - p_
#            p = p_
#            if dot(diff.T, diff) < tolerance:
#                break
#
#            iterations += 1
#            if iterations >= max_iter:
#                warnings.warn('Maximum number of iterations reached '
#                              'before convergence')
#                break
##    if r < c:
##        p = dot(X.T, p)
#    return [p]


#class NIPALS(BaseAlgorithm):
#
#    def __init__(C, mode, scheme, not_normed = [], **kwargs):
#        BaseAlgorithm.__init__(self, **kwargs)
#
#        self.C          = C
#        self.mode       = mode
#        self.scheme     = scheme
#        self.not_normed = not_normed
#
#
def NIPALS(X, adj_matrix, mode, scheme, not_normed = [], prox_op = prox_op.ProxOp(),
           max_iter = MAX_ITER, tolerance = TOLERANCE, **kwargs):
    """Inner loop of the NIPALS algorithm.

    Performs the NIPALS algorithm on the supplied tuple or list of numpy
    arrays in X. This method applies for 1, 2 or more blocks.

    One block would result in e.g. PCA; two blocks would result in e.g.
    SVD or CCA; and multiblock (n > 2) would be for instance GCCA,
    PLS-PM or MAXDIFF.

    This function uses Wold's procedure (based on Gauss-Siedel iteration)
    for fast convergence.

    Parameters
    ----------
    X          : A tuple or list with n numpy arrays of shape [M, N_i],
                 i=1,...,n. These are the training set.

    adj_matrix          : Adjacency matrix that is a numpy array of shape [n, n].
                 If an element in position adj_matrix[i,j] is 1, then block i and
                 j are connected, and 0 otherwise. If adj_matrix is None, then all
                 matrices are assumed to be connected.

    mode       : A tuple or list with n elements with the mode to use for
                 a matrix. The mode is represented by a string: "A", "B"
                 or "NewA". If mode = None, then "NewA" is used.

    scheme     : The inner weighting scheme to use in the algorithm. The
                 scheme may be "Horst", "Centroid" or "Factorial". If
                 scheme = None, then Horst's schme is used (the inner
                 weighting scheme is the identity).

    max_iter   : The number of iteration before the algorithm is forced
                 to stop. The default number of iterations is 500.

    tolerance  : The level below which we treat numbers as zero. This is
                 used as stop criterion in the algorithm. Smaller value
                 will give more acurate results, but will take longer
                 time to compute. The default tolerance is 5E-07.

    not_normed : In some algorithms, e.g. PLS regression, the weights or
                 loadings of some matrices (e.g. Y) are not normalised.
                 This tuple or list contains the indices in X of those
                 matrices that should not be normalised. Thus, for PLS
                 regression, this argument would be not_normed = (2,). If
                 not_normed = None, then all matrices are subject to
                 normalisation of either weights or scores depending on
                 the modes used.

    Returns
    -------
    W          : A list with n numpy arrays of weights of shape [N_i, 1].
    """

    n = len(X)

    mode           = make_list(mode, n, NEWA)
    scheme         = make_list(scheme, n, HORST)
#    soft_threshold = make_list(soft_threshold, n, 0)

    W = []
    for Xi in X:
        w = _start_vector(Xi, largest = True)
        W.append(w)

    # Main NIPALS loop
    iterations = 0
    while True:
        converged = True
        for i in range(n):
            Xi = X[i]
            ti = dot(Xi, W[i])
            ui = np.zeros(ti.shape)
            for j in range(n):
                Xj = X[j]
                wj = W[j]
                tj = dot(Xj, wj)

                # Determine weighting scheme and compute weight
                if scheme[i] == HORST:
                    eij = 1
                elif scheme[i] == CENTROID:
                    eij = sign(corr(ti, tj))
                elif scheme[i] == FACTORIAL:
                    eij = corr(ti, tj)

                # Internal estimation using connected matrices' score vectors
                if adj_matrix[i,j] != 0 or adj_matrix[j,i] != 0:
                    ui += eij*tj

            # External estimation
            if mode[i] == NEWA or mode[i] == A:
                # TODO: Ok with division here?
                wi = dot(Xi.T, ui) / dot(ui.T, ui)
            elif mode[i] == B:
                wi = dot(np.pinv(Xi), ui) # TODO: Precompute to speed up!

            # Apply proximal operator
#            if soft_threshold[i] > 0:
            wi  = prox_op.prox(wi, i)
#                wi = _soft_threshold(wi, soft_threshold[i], copy = False)

            # Normalise weight vectors according to their weighting scheme
            if mode[i] == NEWA and not i in not_normed:
                # Normalise weight vector wi to unit variance
                wi /= norm(wi)
            elif (mode[i] == A or mode[i] == B) and not i in not_normed:
                # Normalise score vector ti to unit variance
                wi /= norm(dot(Xi, wi))
                wi *= np.sqrt(wi.shape[0])

            # Check convergence for each weight vector. They all have to leave
            # converged = True in order for the algorithm to stop.
            diff = wi - W[i]
            if dot(diff.T, diff) > tolerance:
                converged = False

            # Save updated weight vector
            W[i] = wi

        if converged:
            break

        if iterations >= max_iter:
            warnings.warn('Maximum number of iterations reached '
                          'before convergence')
            break

        iterations += 1

    return W



#class RGCCA(BaseAlgorithm):
#
#    def __init__(C, tau, scheme, not_normed = [], **kwargs):
#        BaseAlgorithm.__init__(self, **kwargs)
#
#        self.C          = C
#        self.tau        = tau
#        self.scheme     = scheme
#        self.not_normed = not_normed
#
#
def RGCCA(X, adj_matrix, tau, scheme, not_normed = [],
          max_iter = MAX_ITER, tolerance = TOLERANCE, **kwargs):
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

    adj_matrix          : Adjacency matrix that is a numpy array of shape [n, n].
                 If an element in position adj_matrix[i,j] is 1, then block i and
                 j are connected, and 0 otherwise. If adj_matrix is None, then all
                 matrices are assumed to be connected such that adj_matrix has
                 ones everywhere except for on the diagonal.

    tau        : A tuple or list with n shrinkage constants tau[i]. If
                 tau is a single real, all matrices will use this value.

    scheme     : The inner weighting scheme to use in the algorithm. The
                 scheme may be "Horst", "Centroid" or "Factorial". If
                 scheme = None, then Horst's scheme is used (where the
                 inner weighting scheme is the identity).

    max_iter   : The number of iteration before the algorithm is forced
                 to stop. The default number of iterations is 500.

    tolerance  : The level below which we treat numbers as zero. This is
                 used as stop criterion in the algorithm. Smaller value
                 will give more acurate results, but will take longer
                 time to compute. The default tolerance is 5E-07.

    not_normed : In some algorithms, e.g. PLS regression, the weights or
                 loadings of some matrices (e.g. Y) are not normalised.
                 This tuple or list contains the indices in X of those
                 matrices that should not be normalised. Thus, for PLS
                 regression, this argument would be not_normed = (2,). If
                 not_normed = None, then all matrices are subject to
                 normalisation of either weights or scores depending on
                 the modes used.

    Returns
    -------
    W          : A list with n numpy arrays of weights of shape [N_i, 1].
    """

    n = len(X)

    invIXX = []
    W      = []
    for i in range(n):
        Xi = X[i]
        XX = dot(Xi.T, Xi)
        I  = np.eye(XX.shape[0])

        w  = _start_vector(Xi, largest = True)

        invIXX.append(np.linalg.pinv(tau[i]*I + ((1-tau[i])/w.shape[0])*XX))
        invIXXw  = dot(invIXX[i], w)
        winvIXXw = dot(w.T, invIXXw)
        w        = invIXXw/np.sqrt(winvIXXw)

        W.append(w)

    # Main RGCCA loop
    iterations = 0
    while True:

        converged = True
        for i in range(n):
            Xi = X[i]
            ti = dot(Xi, W[i])
            ui = np.zeros(ti.shape)
            for j in range(n):
                tj = dot(X[j], W[j])

                # Determine weighting scheme and compute weight
                if scheme == HORST:
                    eij = 1
                elif scheme == CENTROID:
                    eij = sign(cov(ti, tj))
                elif scheme == FACTORIAL:
                    eij = cov(ti, tj)

                # Internal estimation using connected matrices' score vectors
                if adj_matrix[i,j] != 0 or adj_matrix[j,i] != 0:
                    ui += eij*tj

            # Outer estimation for block i
            wi = dot(Xi.T, ui)
            invIXXw  = dot(invIXX[i], wi)
            # Should we normalise?
            if not i in not_normed:
                winvIXXw = dot(wi.T, invIXXw)
                wi        = invIXXw / np.sqrt(winvIXXw)
            else:
                wi        = invIXXw

            # Check convergence for each weight vector. They all have to leave
            # converged = True in order for the algorithm to stop.
            diff = wi - W[i]
            if dot(diff.T, diff) > tolerance:
                converged = False

            # Save updated weight vector
            W[i] = wi

        if converged:
            break

        if iterations >= max_iter:
            warnings.warn('Maximum number of iterations reached '
                          'before convergence')
            break

        iterations += 1

    return W
