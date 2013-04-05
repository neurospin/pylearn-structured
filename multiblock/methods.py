# -*- coding: utf-8 -*-
"""
The :mod:`multiblock.methods` module includes several different multiblock
methods.
"""

# Author: Tommy Löfstedt <tommy.loefstedt@cea.fr>
# License: BSD Style.

__all__ = ['PCA', 'SVD', 'PLSR', 'PLSC', 'O2PLS']

from sklearn.utils import check_arrays

import abc
import warnings
import numpy as np
from numpy.linalg import inv, pinv
from multiblock.utils import dot, direct
#import preprocess as pp
import algorithms
import copy
import prox_ops
import schemes
import modes


class BaseMethod(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, algorithm, num_comp=2, norm_dir=False):

        super(BaseMethod, self).__init__()

        # Supplied by the user
        self.algorithm = algorithm
        self.num_comp = num_comp
        self.norm_dir = norm_dir

        self._check_inputs()

    def _check_inputs(self):

        if self.num_comp < 1:
            raise ValueError('At least one component must be computed')

        if not isinstance(self.algorithm, algorithms.BaseAlgorithm):
            raise ValueError('Argument "algorithm" must be a BaseAlgorithm ' \
                             'instance')

        if isinstance(self.num_comp, (tuple, list)):
            comps = self.num_comp
        else:
            comps = [self.num_comp]
        for i in xrange(len(comps)):
            if comps[i] < 0:
                raise ValueError('Invalid number of components')
        if not isinstance(self.num_comp, (tuple, list)):
            self.num_comp = comps[0]

        try:
            self.norm_dir = bool(self.norm_dir)
        except ValueError:
            raise ValueError('Argument norm_dir must be a bool')

    def _check_arrays(self, *X):

        if not hasattr(self, "n"):
            self.n = len(X)
        elif self.n != len(X):
            raise ValueError('Number of matrices differ from previous numbers')
        if self.n < 1:
            raise ValueError('At least one matrix must be given')

        # Copy since this will contain the residual (deflated) matrices
        X = check_arrays(*X, dtype=np.float, copy=True, sparse_format='dense')

        # Check number of rows
        M = X[0].shape[0]
        for i in xrange(self.n):
            if X[i].ndim == 1:  # If vector, make two-dimensional
                X[i] = X[i].reshape((X[i].size, 1))
            if X[i].ndim != 2:
                raise ValueError('The matrices in X must be 1- or 2D arrays')

            if X[i].shape[0] != M:
                raise ValueError('Incompatible shapes: X[%d] has %d samples, '
                                 'while X[%d] has %d'
                                 % (0, M, i, X[i].shape[0]))
        return X

#    def preprocess(self, *X):
#        X = list(X)
#        if self.preproc == None:
#            return X
#
#        for i in xrange(len(X)):
#            if self.preproc[i] != None:
#                X[i] = self.preproc[i].process(X[i])
#        return X
#
#    def postprocess(self, *X):
#        X = list(X)
#
#        if self.preproc == None:
#            return X
#
#        for i in xrange(len(X)):
#            if self.preproc[i] != None:
#                X[i] = self.preproc[i].revert(X[i])
#        return X

    def get_algorithm(self):
        return self.algorithm

    def set_algorithm(self, algorithm):
        if not isinstance(self.algorithm, algorithms.BaseAlgorithm):
            raise ValueError('The algorithm must be an instance of ' \
                             '"BaseAlgorithm"')
        self.algorithm = algorithm

    @abc.abstractmethod
    def _get_transform(self, index=0):
        raise NotImplementedError('Abstract method "_get_transform" must be '\
                                  'specialised!')

    def deflate(self, X, w, t, p, index=None):
        return X - dot(t, p.T)

    @abc.abstractmethod
    def fit(self, *X, **kwargs):
        raise NotImplementedError('Abstract method "fit" must be specialised!')


class PLSBaseMethod(BaseMethod):

    def __init__(self, algorithm=None, **kwargs):

        if algorithm == None:
            algorithm = algorithms.NIPALSAlgorithm()

        super(PLSBaseMethod, self).__init__(algorithm=algorithm, **kwargs)

#        # Supplied by the user
#        self.adj_matrix = adj_matrix

    def _check_inputs(self):

        super(PLSBaseMethod, self)._check_inputs()

#        if self.adj_matrix != None and not hasattr(self, "n"):
#            self.n = max(self.adj_matrix.shape)
#        if hasattr(self, "n"):
#            if self.adj_matrix == None and self.n == 1:
#                self.adj_matrix = np.ones((1, 1))
#            elif self.adj_matrix == None and self.n > 1:
#                self.adj_matrix = np.ones((self.n, self.n)) - np.eye(self.n)
#        elif self.adj_matrix == None:
#            raise ValueError('Argument "adj_matrix" must be given')

    def fit(self, *X, **kwargs):

        X = list(self._check_arrays(*X))
#        X = self.preprocess(*X)

        # Results matrices
        self.W = []
        self.T = []
        self.P = []
        self.Ws = []
        for i in xrange(self.n):
            M, N = X[i].shape
            w = np.zeros((N, self.num_comp))
            t = np.zeros((M, self.num_comp))
            p = np.zeros((N, self.num_comp))
            ws = np.zeros((N, self.num_comp))
            self.W.append(w)
            self.T.append(t)
            self.P.append(p)
            self.Ws.append(ws)

        # Outer loop, over components
        for a in xrange(self.num_comp):
            # Inner loop, weight estimation
            w = self.algorithm.run(*X)

            # Compute scores and loadings
            for i in xrange(self.n):

                # Score vector
                t = dot(X[i], w[i]) / dot(w[i].T, w[i])

                # Test for null variance
                if dot(t.T, t) < self.algorithm.tolerance:
                    warnings.warn('Scores of block X[%d] are too small at '
                                  'iteration %d' % (i, a))

                # Loading vector
                p = dot(X[i].T, t) / dot(t.T, t)

                # If we should make all weights correlate with np.ones((N,1))
                if self.norm_dir:
                    w[i], t, p = direct(w[i], t, p)

                self.W[i][:, a] = w[i].ravel()
                self.T[i][:, a] = t.ravel()
                self.P[i][:, a] = p.ravel()

                # Generic deflation method. Overload for specific deflation!
                X[i] = self.deflate(X[i], w[i], t, p, index=i)

        # Compute W*, the rotation from input space X to transformed space T
        # such that T = XW* = XW(P'W)^-1
        for i in xrange(self.n):
            self.Ws[i] = dot(self.W[i], inv(dot(self.P[i].T, self.W[i])))

        return self

    def _get_transform(self, index=0):
        """ Returns the linear transformation W that generates the score
        matrix T = XW* for matrix with index index.

        Arguments
        ---------
        X : The matrices to center

        Returns
        -------
        Centered X
        """
        return self.Ws[index]

    def transform(self, *X, **kwargs):

        X = self._check_arrays(*X)
#        X = self.preprocess(*X)

        T = []
        for i in xrange(self.n):
            # Apply rotation
            t = dot(X[i], self._get_transform(i))

            T.append(t)

        return T

    def fit_transform(self, *X, **kwargs):
        return self.fit(*X, **kwargs).transform(*X)


class PCA(PLSBaseMethod):

    def __init__(self, **kwargs):
#        prepro = kwargs.pop("preprocess", pp.PreprocessQueue([pp.Center(),
#                                                              pp.Scale()]))
        super(PCA, self).__init__(**kwargs)

    def _get_transform(self, index=0):
        return self.P

    def fit(self, *X, **kwargs):
#        PLSBaseMethod.fit(self, X[0], **kwargs)
        super(PCA, self).fit(X[0], **kwargs)
        self.T = self.T[0]
        self.P = self.W[0]
        del self.W
        del self.Ws

        return self

    def transform(self, *X, **kwargs):
#        T = PLSBaseMethod.transform(self, X[0], **kwargs)
        T = super(PCA, self).transform(X[0], **kwargs)
        return T[0]

    def fit_transform(self, *X, **fit_params):
        return self.fit(X[0], **fit_params).transform(X[0])


class SVD(PCA):
    """Performs the singular value decomposition.

    The decomposition generates matrices such that

        dot(U, dot(S, V.T)) == X
    """

    def __init__(self, **kwargs):
#        PCA.__init__(self, **kwargs)
        super(SVD, self).__init__(**kwargs)

    def _get_transform(self, index=0):
        return self.V

    def fit(self, *X, **kwargs):
#        PCA.fit(self, X[0], **kwargs)
        super(SVD, self).fit(X[0], **kwargs)
        self.U = self.T
        # Move norms of U to the diagonal matrix S
        norms = np.sum(self.U ** 2, axis=0) ** (0.5)
        self.U /= norms
        self.S = np.diag(norms)
        self.V = self.P
        del self.T
        del self.P

        return self


class PLSR(PLSBaseMethod):
    """Performs PLS regression between two matrices X and Y.
    """

    def __init__(self, algorithm=None, **kwargs):
#        prepro = kwargs.pop("preprocess", pp.PreprocessQueue([pp.Center(),
#                                                              pp.Scale()]))
#        PLSBaseMethod.__init__(self, algorithm=algorithm, **kwargs)
        if algorithm == None:
            algorithm = algorithms.NIPALSAlgorithm(not_normed=[1])

        super(PLSR, self).__init__(algorithm=algorithm, **kwargs)

    def _get_transform(self, index=0):
        if index == 0:
            return self.Ws
        else:
            return self.C

    def deflate(self, X, w, t, p, index=None):
        if index == 0:
            return X - dot(t, p.T)  # Deflate X using its loadings
        else:
            return X  # Do not deflate Y

    def fit(self, X, Y=None, **kwargs):
        Y = kwargs.pop('y', Y)
        if Y == None:
            raise ValueError('Y is not supplied')
#        PLSBaseMethod.fit(self, X, Y, **kwargs)
        super(PLSR, self).fit(X, Y, **kwargs)
        self.C = self.W[1]
        self.U = self.T[1]
        self.Q = self.P[1]
        self.W = self.W[0]
        self.T = self.T[0]
        self.P = self.P[0]
        self.Ws = self.Ws[0]

        self.B = dot(self.Ws, self.C.T)

        return self

    def predict(self, X):
        X = np.asarray(X)
#        if self.preproc != None and self.preproc[0] != None:
#            X = self.preproc[0].process(X)

        Ypred = dot(X, self.B)

#        if self.preproc != None and self.preproc[1] != None:
#            Ypred = self.preproc[1].revert(Ypred)

        return Ypred

    def transform(self, X, Y=None, **kwargs):
        Y = kwargs.pop('y', Y)
        if Y != None:
#            T = PLSBaseMethod.transform(self, X, Y, **kwargs)
            T = super(PLSR, self).transform(X, Y, **kwargs)
        else:
#            T = PLSBaseMethod.transform(self, X, **kwargs)
            T = super(PLSR, self).transform(X, **kwargs)
            T = T[0]
        return T

    def fit_transform(self, X, Y=None, **kwargs):
        Y = kwargs.pop('y', Y)
        return self.fit(X, Y, **kwargs).transform(X, Y)


class PLSC(PLSR):
    """ PLS with canonical deflation (symmetric).

    Note that the model of each matrix is:
        X = T.P'
        Y = U.Q'
    with the inner relation T = U + H, i.e. with T = U.Dy and U = T.Dx.

    The prediction is therefore performed as:
        Xhat = Y.C*.Dy.P' = U.(U'.U)^-1.U'.T.P' = Y.By
        Yhat = X.W*.Dx.Q' = T.(T'.T)^-1.T'.U.Q' = X.Bx,
    i.e. with least squares estimation of T and U, times P and Q,
    respectively.
    """

    def __init__(self, algorithm=None, **kwargs):
#        PLSR.__init__(self, algorithm=algorithm, **kwargs)
        if algorithm == None:
            algorithm = algorithms.NIPALSAlgorithm()
        super(PLSC, self).__init__(algorithm=algorithm, **kwargs)

    def _get_transform(self, index=0):
        if index == 0:
            return self.Ws
        else:  # index == 1
            return self.Cs

    def deflate(self, X, w, t, p, index=None):
        return X - dot(t, p.T)  # Deflate using their loadings

    def fit(self, X, Y=None, **kwargs):
        Y = kwargs.pop('y', Y)
#        PLSR.fit(self, X, Y, **kwargs)
        super(PLSC, self).fit(X, Y, **kwargs)

        self.Cs = dot(self.C, inv(dot(self.Q.T, self.C)))

        self.Dx = dot(pinv(self.T), self.U)
        self.Dy = dot(pinv(self.U), self.T)

        self.Bx = dot(self.Ws, dot(self.Dx, self.Q.T))  # Yhat = XW*DxQ' = XBx
        self.By = dot(self.Cs, dot(self.Dy, self.P.T))  # Xhat = YC*DyP' = YBy
#        self.Bx = dot(self.Ws, self.Q.T)               # Yhat = XW*Q' = XBx
#        self.By = dot(self.Cs, self.P.T)               # Xhat = XC*P' = YBy
        del self.B

        return self

    def predict(self, X, Y=None, **kwargs):
        Y = kwargs.pop('y', Y)

        X = np.asarray(X)
        Ypred = dot(X, self.Bx)

        if Y != None:
            Y = np.asarray(Y)
            Xpred = dot(Y, self.By)

            return Ypred, Xpred

        return Ypred


class O2PLS(PLSC):

    def __init__(self, num_comp=[2, 1, 1], **kwargs):
#      PLSC.__init__(self, num_comp=num_comp[0], algorithm=algorithm, **kwargs)
        super(O2PLS, self).__init__(num_comp=num_comp[0],
                                    **kwargs)
        self.A = num_comp[0]
        self.Ax = num_comp[1]
        self.Ay = num_comp[2]

        prox_op = self.algorithm.get_prox_op()
        if len(prox_op.parameter) != 0:
            Xparam = prox_op.parameter[0]
            Yparam = prox_op.parameter[1]
            joint_param = [Xparam[0], Yparam[0]]
            unique_x_param = [Xparam[1]]
            unique_y_param = [Yparam[1]]

            self.unique_x_op = copy.copy(prox_op)
            self.unique_x_op.parameter = unique_x_param
            self.unique_y_op = copy.copy(prox_op)
            self.unique_y_op.parameter = unique_y_param
            self.prox_op.parameter = joint_param
        else:
            self.unique_x_op = prox_ops.ProxOp()
            self.unique_y_op = prox_ops.ProxOp()
            self.prox_op = prox_ops.ProxOp()

    def fit(self, X, Y=None, **kwargs):

        Y = kwargs.pop('y', Y)
        if Y == None:
            raise ValueError("Both X and Y must be given!")
#        self.num_comp = kwargs.pop("num_comp", self.num_comp)

        self._check_inputs()
        X, Y = self._check_arrays(X, Y)

        # Results matrices
        M, N1 = X.shape
        M, N2 = Y.shape
        self.Wo = np.zeros((N1, self.Ax))
        self.To = np.zeros((M,  self.Ax))
        self.Po = np.zeros((N1, self.Ax))
        self.Co = np.zeros((N2, self.Ay))
        self.Uo = np.zeros((M,  self.Ay))
        self.Qo = np.zeros((N2, self.Ay))

        svd_alg = copy.deepcopy(self.algorithm)
        svd_alg.set_prox_op(self.prox_op)
        svd = SVD(num_comp=self.A, algorithm=svd_alg, **kwargs)
        svd.fit(dot(X.T, Y))
        W = svd.U
        C = svd.V

        self.algorithm.set_prox_op(self.unique_x_op)
        eigsym = SVD(num_comp=1, algorithm=self.algorithm, **kwargs)
        for a in xrange(self.Ax):
            T = dot(X, W)
            E = X - dot(T, W.T)
            TE = dot(T.T, E)
            eigsym.fit(TE)
            wo = eigsym.V
            s = eigsym.S
            if s < self.algorithm.tolerance:
                wo = np.zeros(wo.shape)
            to = dot(X, wo)
            toto = dot(to.T, to)
            Xto = dot(X.T, to)
            if toto > self.algorithm.tolerance:
                po = Xto / toto
            else:
                po = np.zeros(Xto.shape)

            self.Wo[:, a] = wo.ravel()
            self.To[:, a] = to.ravel()
            self.Po[:, a] = po.ravel()

            X = X - dot(to, po.T)

        self.algorithm.set_prox_op(self.unique_y_op)
        eigsym = SVD(num_comp=1, algorithm=self.algorithm, **kwargs)
        for a in xrange(self.Ay):
            U = dot(Y, C)
            F = Y - dot(U, C.T)
            UF = dot(U.T, F)
            eigsym.fit(UF)
            co = eigsym.V
            s = eigsym.S
            if s < self.algorithm.tolerance:
                co = np.zeros(co.shape)
            uo = dot(Y, co)
            uouo = dot(uo.T, uo)
            Yuo = dot(Y.T, uo)
            if uouo > self.algorithm.tolerance:
                qo = Yuo / uouo
            else:
                qo = np.zeros(Yuo.shape)

            self.Co[:, a] = co.ravel()
            self.Uo[:, a] = uo.ravel()
            self.Qo[:, a] = qo.ravel()

            Y = Y - dot(uo, qo.T)

        self.algorithm.set_prox_op(self.prox_op)
        self.algorithm.adj_matrix = None
        self.algorithm.scheme = schemes.Horst()
        self.algorithm.mode = modes.NewA()
#        PLSC.fit(self, X, Y, **kwargs)
        super(O2PLS, self).fit(X, Y, **kwargs)

        return self

    def transform(self, X, Y=None, **kwargs):
        Y = kwargs.get('y', Y)

        if Y != None:
            X, Y = self._check_arrays(X, Y)

            if self.Ax > 0:
#                X -= self.means[0]
#                X /= self.stds[0]
                for a in xrange(self.Ax):
                    to = dot(X, self.Wo[:, [a]])
                    X = X - dot(to, self.Po[:, [a]].T)
#                To = dot(X, self.Wo)
#                X = X - dot(To, self.Po.T)
#                X *= self.stds[0]
#                X += self.means[0]

            if self.Ay > 0:
#                Y -= self.means[1]
#                Y /= self.stds[1]
                for a in xrange(self.Ay):
                    uo = dot(Y, self.Co[:, [a]])
                    Y = Y - dot(uo, self.Qo[:, [a]].T)
#                Uo = dot(Y, self.Co)
#                Y = Y - dot(Uo, self.Qo.T)
#                Y *= self.stds[1]
#                Y += self.means[1]

#            T = PLSC.transform(self, X, Y, **kwargs)
            T = super(O2PLS, self).transform(X, Y, **kwargs)
        else:
            X = self._check_arrays(X)[0]

            if self.Ax > 0:
#                X -= self.means[0]
#                X /= self.stds[0]
                for a in xrange(self.Ax):
                    to = dot(X, self.Wo[:, [a]])
                    X = X - dot(to, self.Po[:, [a]].T)
#                To = dot(X, self.Wo)
#                X = X - dot(To, self.Po.T)
#                X *= self.stds[0]
#                X += self.means[0]

#            T = PLSC.transform(self, X, **kwargs)
            T = super(O2PLS, self).transform(X, **kwargs)

        return T

    def predict(self, X=None, Y=None, copy=True, **kwargs):
        Y = kwargs.get('y', Y)
        if X == None and Y == None:
            raise ValueError("At least one of X and Y must be given")

        if X != None:
            X = np.asarray(X)
#            if copy:
#                X = (X - self.means[0]) / self.stds[0]
#            else:
#                X -= self.means[0]
#                X /= self.stds[0]
            for a in xrange(self.Ax):
                to = dot(X, self.Wo[:, [a]])
                X = X - dot(to, self.Po[:, [a]].T)
#            To = dot(X, self.Wo)
#            if copy:
#                X = X - dot(To, self.Po.T)
#            else:
#                X -= dot(To, self.Po.T)

#            self.Bx = dot(self.Ws, dot(self.Dx, self.Q.T))
#            Ypred = (dot(X, self.Bx)*self.stds[1]) + self.means[1]
            Ypred = dot(X, self.Bx)

        if Y != None:
            Y = np.asarray(Y)
#            if copy:
#                Y = (Y - self.means[1]) / self.stds[1]
#            else:
#                Y -= self.means[1]
#                Y /= self.stds[1]
            for a in xrange(self.Ay):
                uo = dot(Y, self.Co[:, [a]])
                Y = Y - dot(uo, self.Qo[:, [a]].T)
#            Uo = dot(Y, self.Co)
#            if copy:
#                Y = Y - dot(Uo, self.Qo.T)
#            else:
#                Y -= dot(Uo, self.Qo.T)

#            Xpred = (dot(Y, self.By)*self.stds[0]) + self.means[0]
            Xpred = dot(Y, self.By)

        if X != None and Y != None:
            return Ypred, Xpred
        elif X != None:
            return Ypred
        else:  # Y != None
            return Xpred


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