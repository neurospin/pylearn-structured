# -*- coding: utf-8 -*-
"""
The :mod:`multiblock.methods` module includes several different projection
based latent variable methods that all are computed using the NIPALS algorithm.
"""

# Author: Tommy Löfstedt <tommy.loefstedt@cea.fr>
# License: BSD Style.

__all__ = ['PCA', 'SVD', 'EIGSym', 'PLSR', 'PLSC', 'O2PLS',
           'center', 'scale', 'direct']

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils import check_arrays

import abc
import warnings
import numpy as np
from multiblock.utils import *
import prox_op
import algorithms
from algorithms import NEWA, A, B
from algorithms import HORST, CENTROID, FACTORIAL


## Settings
#_MAXITER    = 500
#_TOLERANCE  = 5e-7


#def _soft_threshold(w, l, copy = True):
#    worig = w.copy()
#    lorig = l
#
#    warn = False
#    while True:
#        sign = np.sign(worig)
#        if copy:
#            w = np.absolute(worig) - l
#        else:
#            np.absolute(worig, w)
#            w -= l
#            w[w < 0] = 0
#        w = np.multiply(sign,w)
#
#        if np.linalg.norm(w) > _TOLERANCE:
#            break
#        else:
#            warn = True
#            # TODO: Can this be improved?
#            l *= 0.9 # Reduce by 10 % until at least one variable is significant
#
#    if warn:
#        warnings.warn('Soft threshold was too large (all variables purged).'\
#                ' Threshold reset to %f (was %f)' % (l, lorig))
#    return w


def center(X, return_means = False, copy = True):
    """ Centers the numpy array(s) in X

    Arguments
    ---------
    X            : The matrices to center
    return_means : Whether or not the computed means are to be computed as well
    copy         : Whether or not to return a copy, or center in-place

    Returns
    -------
        Centered X, means
    """

    is_list = True
    if not isinstance(X, (tuple, list)):
        X       = [X]
        is_list = False

    means = []
    for i in xrange(len(X)):
        X[i] = np.asarray(X[i])
        mean = X[i].mean(axis = 0)
        if copy:
            X[i] = X[i] - mean
        else:
            X[i] -= mean
        means.append(mean)

    if not is_list:
        X     = X[0]
        means = means[0]

    if return_means:
        return X, means
    else:
        return X


def scale(X, centered = True, return_stds = False, copy = True):
    """ Scales the numpy arrays in arrays to standard deviation 1
    Returns
    -------
        Scaled arrays, stds
    """

    is_list = True
    if not isinstance(X, (tuple, list)):
        X       = [X]
        is_list = False

    stds = []
    for i in xrange(len(X)):
        if centered == True:
            ddof = 1
        else:
            ddof = 0
        std = X[i].std(axis = 0, ddof = ddof)
        std[std < TOLERANCE] = 1.0
        if copy:
            X[i] = X[i] / std
        else:
            X[i] /= std
        stds.append(std)

    if not is_list:
        X    = X[0]
        stds = stds[0]

    if return_stds:
        return X, stds
    else:
        return X


def direct(W, T = None, P = None, compare = False):
    if compare and T == None:
        raise ValueError("In order to compare you need to supply two arrays")

    for j in xrange(W.shape[1]):
        w = W[:,[j]]
        if compare:
            t = T[:,[j]]
            cov = dot(w.T, t)
            if P != None:
                p = P[:,[j]]
                cov2 = dot(w.T, p)
        else:
            cov = dot(w.T, np.ones(w.shape))
        if cov < 0:
            if not compare:
                w *= -1
                if T != None:
                    t = T[:,[j]]
                    t *= -1
                    T[:,j] = t.ravel()
                if P != None:
                    p = P[:,[j]]
                    p *= -1
                    P[:,j] = p.ravel()
            else:
                t = T[:,[j]]
                t *= -1
                T[:,j] = t.ravel()

            W[:,j] = w.ravel()

        if compare and P != None and cov2 < 0:
            p = P[:,[j]]
            p *= -1
            P[:,j] = p.ravel()

    if T != None and P != None:
        return W, T, P
    elif T != None and P == None:
        return W, T
    elif T == None and P != None:
        return W, P
    else:
        return W


class BasePLS(BaseEstimator, TransformerMixin):
    __metaclass__ = abc.ABCMeta

    def __init__(self, adj_matrix = None, num_comp = 2, tau = None,
                 center = True, scale = True, mode = None, scheme = None,
                 not_normed = None, copy = True, normalise_directions = False,
                 max_iter = MAX_ITER, tolerance = TOLERANCE,
                 prox_op = prox_op.ProxOp()):

        # Supplied by the user
        self.adj_matrix = adj_matrix
        self.num_comp   = num_comp
        self.tau        = tau
        self.center     = center
        self.scale      = scale
        self.mode       = mode
        self.scheme     = scheme
        self.not_normed = not_normed
        self.copy       = copy
        self.max_iter   = max_iter
        self.tolerance  = tolerance
        self.normal_dir = normalise_directions
        self.prox_op    = prox_op


    @abc.abstractmethod
    def _get_transform(self, index = 0):
        raise NotImplementedError('Abstract method "_get_transform" must be specialised!')


    def _algorithm(self, *args, **kwargs):
        return algorithms.NIPALS(*args, **kwargs)


    def _deflate(self, X, w, t, p, index = None):
        return X - dot(t, p.T) # Default is deflation with loadings p


    def _check_inputs(self, X):

        if not hasattr(self, "n"):
            self.n = len(X)
        if self.n < 1:
            raise ValueError('At least one matrix must be given')

        err_msg = 'The mode must be either "%s", "%s" or "%s"' \
                    % (NEWA, A, B)
        if self.mode == None:
            self.mode = [NEWA]*self.n # Default mode is New A
        if not isinstance(self.mode, (tuple, list)):
            if not self.mode in (NEWA, A, B):
                raise ValueError(err_msg)
            self.mode = [self.mode]*self.n # If only one mode is given, all matrices gets this mode
        for m in self.mode:
            if not m in (NEWA, A, B):
                raise ValueError(err_msg)

        err_msg = 'The scheme must be either "%s", "%s" or "%s"' \
                    % (HORST, CENTROID, FACTORIAL)
        if self.scheme == None:
            self.scheme = [HORST]*self.n # Default scheme is Horst
        if not isinstance(self.scheme, (tuple, list)):
            if not self.scheme in (HORST, CENTROID, FACTORIAL):
                raise ValueError(err_msg)
            self.scheme = [self.scheme]*self.n
        for s in self.scheme:
            if not s in (HORST, CENTROID, FACTORIAL):
                raise ValueError(err_msg)

        # TODO: Add Schäfer and Strimmer's method here if tau == None!
        err_msg = 'The shrinking factor tau must be of type float or a ' \
                  'list or tuple with floats'
        if self.tau == None:
            self.tau = [0.5]*self.n
        if not isinstance(self.tau, (tuple, list)):
            if not isinstance(self.tau, (float, int)):
                raise ValueError(err_msg)
            self.tau = [float(self.tau)]*self.n
        for t in self.tau:
            if not isinstance(t, (float,int)):
                raise ValueError(err_msg)

        err_msg = 'The argument center must be of type bool or a ' \
                  'list or tuple with bools'
        if self.center == None:
            self.center = [True]*self.n
        if not isinstance(self.center, (tuple, list)):
            if not isinstance(self.center, (bool,)):
                raise ValueError(err_msg)
            self.center = [self.center]*self.n
        for c in self.center:
            if not isinstance(c, (bool,)):
                raise ValueError(err_msg)

        err_msg = 'The argument scale must be of type bool or a ' \
                  'list or tuple with bools'
        if self.scale == None:
            self.scale = [True]*self.n
        if not isinstance(self.scale, (tuple, list)):
            if not isinstance(self.scale, (bool,)):
                raise ValueError(err_msg)
            self.scale = [self.scale]*self.n
        for s in self.scale:
            if not isinstance(s, (bool,)):
                raise ValueError(err_msg)

        if self.not_normed == None:
            self.not_normed = ()

        # Number of rows
        M = X[0].shape[0]
        minN = float('Inf')
        for i in xrange(self.n):
            if X[i].ndim == 1:
                X[i] = X[i].reshape((X[i].size, 1))
            if X[i].ndim != 2:
                raise ValueError('The matrices in X must be 1- or 2D arrays')

            if X[i].shape[0] != M:
                raise ValueError('Incompatible shapes: X[%d] has %d samples, '
                                 'while X[%d] has %d' % (0,M, i,X[i].shape[0]))

            minN = min(minN, X[i].shape[1])

        if isinstance(self.num_comp, (tuple, list)):
            comps = self.num_comp
        else:
            comps = [self.num_comp]
        for i in xrange(len(comps)):
            if comps[i] < 0:
                raise ValueError('Invalid number of components')
            if comps[i] > minN:
                warnings.warn('Too many components! No more than %d can be '
                              'computed' % (minN,))
                comps[i] = minN
        if not isinstance(self.num_comp, (tuple, list)):
            self.num_comp = comps[0]

        if self.adj_matrix == None and self.n == 1:
            self.adj_matrix = np.ones((1,1))
        elif self.adj_matrix == None and self.n > 1:
            self.adj_matrix = np.ones((self.n,self.n)) - np.eye(self.n)


    def _preprocess(self, X):
        self.means = []
        self.stds  = []
        for i in xrange(self.n):
            if self.center[i]:
                X[i], means = center(X[i], return_means = True, copy = self.copy)
            else:
                means = np.zeros((1, X[i].shape[1]))
            self.means.append(means)

            if self.scale[i]:
                X[i], stds = scale(X[i], centered=self.center[i], return_stds = True, copy = self.copy)
            else:
                stds = np.ones((1, X[i].shape[1]))
            self.stds.append(stds)

        return X


    def fit(self, *X, **kwargs):
        # Copy since this will contain the residual (deflated) matrices
        X = check_arrays(*X, dtype = np.float, copy = self.copy,
                         sparse_format = 'dense')
        # Number of matrices
        self.n = len(X)

        preprocess = kwargs.pop("preprocess", True)

        self._check_inputs(X)
        if preprocess:
            X = self._preprocess(X)

        # Results matrices
        self.W  = []
        self.T  = []
        self.P  = []
        self.Ws = []
        for i in xrange(self.n):
            M, N = X[i].shape
            w  = np.zeros((N, self.num_comp))
            t  = np.zeros((M, self.num_comp))
            p  = np.zeros((N, self.num_comp))
            ws = np.zeros((N, self.num_comp))
            self.W.append(w)
            self.T.append(t)
            self.P.append(p)
            self.Ws.append(ws)

        # Outer loop, over components
        for a in xrange(self.num_comp):
            # Inner loop, weight estimation
            w = self._algorithm(X = X,
                                tau = self.tau,
                                mode = self.mode,
                                scheme = self.scheme,
                                prox_op = self.prox_op,
                                max_iter = self.max_iter,
                                tolerance = self.tolerance,
                                not_normed = self.not_normed,
                                adj_matrix = self.adj_matrix)

#            self.adj_matrix = adj_matrix
#            self.num_comp   = num_comp
#            self.tau        = tau
#            self.center     = center
#            self.scale      = scale
#            self.mode       = mode
#            self.scheme     = scheme
#            self.not_normed = not_normed
#            self.copy       = copy
#            self.max_iter   = max_iter
#            self.tolerance  = tolerance
#            self.normal_dir = normalise_directions
#            self.prox_op    = prox_op

            # Compute scores and loadings
            for i in xrange(self.n):

                # If we should make all weights correlate with np.ones((N,1))
                if self.normal_dir:
                    w[i] = direct(w[i])

                # Score vector
                t  = dot(X[i], w[i]) / dot(w[i].T, w[i])

                # Test for null variance
                if dot(t.T, t) < self.tolerance:
                    warnings.warn('Scores of block X[%d] are too small at '
                                  'iteration %d' % (i, a))

                # Loading vector
                p = dot(X[i].T, t) / dot(t.T, t)

                self.W[i][:,a] = w[i].ravel()
                self.T[i][:,a] = t.ravel()
                self.P[i][:,a] = p.ravel()

                # Generic deflation method. Overload for specific deflation!
                X[i] = self._deflate(X[i], w[i], t, p, i)

        # Compute W*, the rotation from input space X to transformed space T
        # such that T = XW(P'W)^-1 = XW*
        for i in xrange(self.n):
            self.Ws[i] = dot(self.W[i], np.linalg.inv(dot(self.P[i].T,self.W[i])))

        return self


    def transform(self, *X, **kwargs):

        copy = kwargs.pop('copy', True)

        n = len(X)
        if n > self.n:
            raise ValueError('Model was trained for %d matrices', self.n)

        T  = []
        for i in xrange(n):
            X_ = np.asarray(X[i])
            # TODO: Use center() and scale() instead! (Everywhere!)
            # Center and scale
            if copy:
                X_ = (X_ - self.means[i]) / self.stds[i]
            else:
                X_ -= self.means[i]
                X_ /= self.stds[i]

            # Apply rotation
            t = dot(X_, self._get_transform(i))

            T.append(t)

        return T


    def fit_transform(self, *X, **fit_params):
        return self.fit(*X, **fit_params).transform(*X)


class PCA(BasePLS):

    def __init__(self, **kwargs):
        BasePLS.__init__(self, adj_matrix = np.ones((1,1)), **kwargs)

    def _get_transform(self, index = 0):
        return self.P

    def fit(self, *X, **kwargs):
        BasePLS.fit(self, X[0], **kwargs)
        self.T = self.T[0]
        self.P = self.W[0]
        del self.W

        return self

    def transform(self, *X, **kwargs):
        T = BasePLS.transform(self, X[0], **kwargs)
        return T[0]

    def fit_transform(self, *X, **fit_params):
        return self.fit(X[0], **fit_params).transform(X[0])


class SVD(PCA):
    """Performs the singular value decomposition.
    
    The decomposition generates matrices such that
    
        dot(U, dot(S, V)) == X
    """

    def __init__(self, **kwargs):
        center = kwargs.pop("center", False)
        scale  = kwargs.pop("scale",  False)
        PCA.__init__(self, center = center, scale = scale, **kwargs)

#    def _algorithm(self, *args, **kwargs):
#        return algorithms.NIPALS_PCA(*args, **kwargs)

    def _get_transform(self, index = 0):
        return self.V

    def fit(self, *X, **kwargs):
        PCA.fit(self, X[0], **kwargs)
        self.U = self.T
        # Move norms of U to the diagonal matrix S
        norms = np.sum(self.U**2, axis = 0)**(0.5)
        self.U /= norms
        self.S = np.diag(norms)
        self.V = self.P
        del self.T
        del self.P

        return self


class EIGSym(SVD):
    """Performs eigenvalue decomposition of a symmetric matrix.

    The decomposition generates matrices such that

        dot(V, dot(D, V.T)) == X
    """

    def __init__(self, **kwargs):
        SVD.__init__(self, **kwargs)

    def fit(self, *X, **kwargs):
        SVD.fit(self, X[0], **kwargs)
        self.D = self.S

        del self.U
        del self.S

        return self


class PLSR(BasePLS, RegressorMixin):

    def __init__(self, **kwargs):
#        proxop = kwargs.pop("prox_op", prox_op.ProxOp(normaliser = [norm, normI]))
        BasePLS.__init__(self, mode = NEWA, scheme = HORST, not_normed = [1],
                         **kwargs)

    def _get_transform(self, index = 0):
        if index < 0 or index > 1:
            raise ValueError("Index must be 0 or 1")
        if index == 0:
            return self.Ws
        else:
            return self.C

    def _deflate(self, X, w, t, p, index = None):
        if index == 0:
            return X - dot(t, p.T) # Deflate X using its loadings
        else:
            return X # Do not deflate Y

    def fit(self, X, Y = None, **kwargs):
        Y = kwargs.pop('y', Y)
        if Y == None:
            raise ValueError('Y is not supplied')
        BasePLS.fit(self, X, Y, **kwargs)
        self.C  = self.W[1]
        self.U  = self.T[1]
        self.Q  = self.P[1]
        self.W  = self.W[0]
        self.T  = self.T[0]
        self.P  = self.P[0]
        self.Ws = self.Ws[0]

        self.B = dot(self.Ws, self.C.T)

        return self

    def predict(self, X, copy = True):
        X = np.asarray(X)
        if copy:
            X = (X - self.means[0]) / self.stds[0]
        else:
            X -= self.means[0]
            X /= self.stds[0]

        if hasattr(self, 'B'):
            Ypred = dot(X, self.B)
        else:
            Ypred = dot(X, self.Bx)

        return (Ypred*self.stds[1]) + self.means[1]

    def transform(self, X, Y = None, **kwargs):
        Y = kwargs.pop('y', Y)
        if Y != None:
            T = BasePLS.transform(self, X, Y, **kwargs)
        else:
            T = BasePLS.transform(self, X, **kwargs)
            T = T[0]
        return T

    def fit_transform(self, X, Y = None, **kwargs):
        Y = kwargs.pop('y', Y)
        return self.fit(X, Y, **kwargs).transform(X, Y)



class PLSC(PLSR):
    """ PLS with canonical deflation (symmetric).

    Note that the model of each matrix is:
        X = T.P'
        Y = U.Q'
    The prediction is therefore performed as:
        Xhat = Y.C*.Dy.P' = U.(U'.U)^-1.U'.T.P' = Y.By
        Yhat = X.W*.Dx.Q' = T.(T'.T)^-1.T'.U.Q' = X.Bx,
    i.e. with least squares estimation of T and U, times P and Q, respectively.
    """

    def __init__(self, **kwargs):
        BasePLS.__init__(self, mode = NEWA, scheme = HORST, **kwargs)

    def _get_transform(self, index = 0):
        if index < 0 or index > 1:
            raise ValueError("Index must be 0 or 1")
        if index == 0:
            return self.Ws
        else: # index == 1
            return self.Cs

    def _deflate(self, X, w, t, p, index = None):
        return X - dot(t, p.T) # Deflate using their loadings

    def fit(self, X, Y = None, **kwargs):
        Y = kwargs.pop('y', Y)
        PLSR.fit(self, X, Y, **kwargs)
        self.Cs = dot(self.C, np.linalg.inv(dot(self.Q.T, self.C)))

        self.Dx = dot(np.linalg.pinv(self.T), self.U)
        self.Dy = dot(np.linalg.pinv(self.U), self.T)

#        self.Bx = self.B
        self.Bx = dot(self.Ws, dot(self.Dx, self.Q.T)) # Yhat = XW*DxQ' = XBx
        self.By = dot(self.Cs, dot(self.Dy, self.P.T)) # Xhat = YC*DyP' = YBy
#        self.Bx = dot(self.Ws, self.Q.T) # Yhat = XW*Q' = XBx
#        self.By = dot(self.Cs, self.P.T) # Xhat = XC*P' = YBy
        del self.B

        return self

    def predict(self, X, Y = None, copy = True, **kwargs):
        Y = kwargs.pop('y', Y)

        Ypred = PLSR.predict(self, X, copy = copy, **kwargs)

        if Y != None:
            Y = np.asarray(Y)
            if copy:
                Y = (Y - self.means[1]) / self.stds[1]
            else:
                Y -= self.means[1]
                Y /= self.stds[1]

            Xpred = (dot(Y, self.By)*self.stds[0]) + self.means[0]

            return Ypred, Xpred

        return Ypred



class O2PLS(PLSC):

    def __init__(self, num_comp = 0, **kwargs):
        PLSC.__init__(self, num_comp = num_comp[0], **kwargs)
        self.A  = num_comp[0]
        self.Ax = num_comp[1]
        self.Ay = num_comp[2]

        if len(self.prox_op.parameter) != 0:
            Xparam = self.prox_op.parameter[0]
            Yparam = self.prox_op.parameter[1]
            joint_param = [Xparam[0], Yparam[0]]
            unique_x_param = [Xparam[1]]
            unique_y_param = [Yparam[1]]

            self.unique_x_op = copy(self.prox_op)
            self.unique_x_op.parameter = unique_x_param
            self.unique_y_op = copy(self.prox_op)
            self.unique_y_op.parameter = unique_y_param
            self.prox_op.parameter = joint_param
        else:
            self.unique_x_op = prox_op.ProxOp()
            self.unique_y_op = prox_op.ProxOp()
            self.prox_op     = prox_op.ProxOp()

    def fit(self, X, Y = None, **kwargs):

        Y = kwargs.pop('y', Y)
        self.num_comp  = kwargs.pop("num_comp",  self.num_comp)
        self.max_iter  = kwargs.pop("max_iter",  self.max_iter)
        self.tolerance = kwargs.pop("tolerance", self.tolerance)

        # Copy since this will contain the residual (deflated) matrices
        X = check_arrays(X, Y, dtype = np.float, copy = self.copy,
                         sparse_format = 'dense')

        self._check_inputs(X)
        X = self._preprocess(X)

        Y = X[1]
        X = X[0]

#        A  = self.num_comp[0]
#        Ax = self.num_comp[1]
#        Ay = self.num_comp[2]

        # Results matrices
        M, N1 = X.shape
        M, N2 = Y.shape
        self.Wo = np.zeros((N1, self.Ax))
        self.To = np.zeros((M,  self.Ax))
        self.Po = np.zeros((N1, self.Ax))
        self.Co = np.zeros((N2, self.Ay))
        self.Uo = np.zeros((M,  self.Ay))
        self.Qo = np.zeros((N2, self.Ay))

        svd = SVD(num_comp = self.A, tolerance = self.tolerance,
                  max_iter = self.max_iter, prox_op = self.prox_op, **kwargs)
        svd.fit(dot(X.T, Y))
        W = svd.U
        C = svd.V

#        eigsym = EIGSym(num_comp = 1)
        eigsym = SVD(num_comp = 1, tolerance = self.tolerance,
                     max_iter = self.max_iter, prox_op = self.unique_x_op,
                     **kwargs)
        for a in xrange(self.Ax):
            T  = dot(X, W)
            E  = X - dot(T, W.T)
            TE = dot(T.T, E)
#            eigsym.fit(dot(TE.T, TE))
            eigsym.fit(TE)
            wo = eigsym.V
            s  = eigsym.S
            if s < self.tolerance:
                wo = np.zeros(wo.shape)
            to   = dot(X, wo)
            toto = dot(to.T, to)
            Xto  = dot(X.T, to)
            if toto > self.tolerance:
                po = Xto / toto
            else:
                po = np.zeros(Xto.shape)

            self.Wo[:,a] = wo.ravel()
            self.To[:,a] = to.ravel()
            self.Po[:,a] = po.ravel()

            X -= dot(to, po.T)

        eigsym = SVD(num_comp = 1, tolerance = self.tolerance,
                     max_iter = self.max_iter, prox_op = self.unique_y_op,
                     **kwargs)
        for a in xrange(self.Ay):
            U  = dot(Y, C)
            F  = Y - dot(U, C.T)
            UF = dot(U.T, F)
#            eigsym.fit(dot(UF.T, UF))
            eigsym.fit(UF)
            co = eigsym.V
            s  = eigsym.S
            if s < self.tolerance:
                co = np.zeros(co.shape)
            uo   = dot(Y, co)
            uouo = dot(uo.T, uo)
            Yuo  = dot(Y.T, uo)
            if uouo > self.tolerance:
                qo = Yuo / uouo
            else:
                qo = np.zeros(Yuo.shape)

            self.Co[:,a] = co.ravel()
            self.Uo[:,a] = uo.ravel()
            self.Qo[:,a] = qo.ravel()

            Y -= dot(uo, qo.T)

        PLSC.fit(self, X, Y, preprocess = False, **kwargs)

        return self

    def transform(self, X, Y = None, **kwargs):
        Y = kwargs.get('y', Y)
        if Y != None:
            To = dot(X, self.Wo)
            X = X - dot(To, self.Po.T)
            Uo = dot(Y, self.Co)
            Y = Y - dot(Uo, self.Qo.T)
            T = PLSC.transform(self, X, Y, **kwargs)
        else:
            To = dot(X, self.Wo)
            X = X - dot(To, self.Po.T)
            T = PLSC.transform(self, X, **kwargs)
            T = T[0]
        return T

    def predict(self, X = None, Y = None, copy = True, **kwargs):
        Y = kwargs.get('y', Y)

        if X == None and Y == None:
            raise ValueError("At least one of X and Y must be given")

        if X != None:
            X = np.asarray(X)
            if copy:
                X = (X - self.means[0]) / self.stds[0]
            else:
                X -= self.means[0]
                X /= self.stds[0]
            To = dot(X, self.Wo)
            if copy:
                X = X - dot(To, self.Po.T)
            else:
                X -= dot(To, self.Po.T)

#            self.Bx = dot(self.Ws, dot(self.Dx, self.Q.T))
            Ypred = (dot(X, self.Bx)*self.stds[1]) + self.means[1]

        if Y != None:
            Y = np.asarray(Y)
            if copy:
                Y = (Y - self.means[1]) / self.stds[1]
            else:
                Y -= self.means[1]
                Y /= self.stds[1]
            Uo = dot(Y, self.Co)
            if copy:
                Y = Y - dot(Uo, self.Qo.T)
            else:
                Y -= dot(Uo, self.Qo.T)

            Xpred = (dot(Y, self.By)*self.stds[0]) + self.means[0]

        if X != None and Y != None:
            return Ypred, Xpred
        elif X != None:
            return Ypred
        else: # Y != None
            return Xpred



#class CCA(BasePLS):
#
    #def __init__(self, **kwargs):
        #BasePLS.__init__(self, mode=(_B, _B), scheme=(_FACTORIAL,_FACTORIAL),
                         #tau = (0, 0), **kwargs)
#
#
    #def _get_transform(self, index = 0):
        #if index == 0:
            #return self.Ws
        #elif index == 1:
            #return self.Cs
        #else:
            #raise ValueError("Index must be 0 or 1")
#
#
    #def _algorithm(self, **kwargs):
        #return _RGCCA(**kwargs)
#
#
    #def fit(self, *X, **kwargs):
        #BasePLS.fit(self, *X)
        #self.C  = self.W[1]
        #self.U  = self.T[1]
        #self.Q  = self.P[1]
        #self.Cs = self.Ws[1]
        #self.W  = self.W[0]
        #self.T  = self.T[0]
        #self.P  = self.P[0]
        #self.Ws = self.Ws[0]
#
        #return self
#
#
    #def transform(self, X, Y = None, **kwargs):
        #if Y != None:
            #T = BasePLS.transform(self, X, Y, **kwargs)
        #else:
            #T = BasePLS.transform(self, X, **kwargs)
            #T = T[0]
        #return T


#class Enum(object):
#    def __init__(self, *sequential, **named):
#        enums = dict(zip(sequential, range(len(sequential))), **named)
#        for k, v in enums.items():
#            setattr(self, k, v)
#
#    def __setattr__(self, name, value): # Read-only
#        raise TypeError("Enum attributes are read-only.")
#
#    def __str__(self):
#        return "Enum: "+str(self.__dict__)