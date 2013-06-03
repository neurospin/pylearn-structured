# -*- coding: utf-8 -*-
"""
The :mod:`multiblock.methods` module includes several different multiblock
methods.

@author:  Tommy Löfstedt <tommy.loefstedt@cea.fr>
@email:   tommy.loefstedt@cea.fr
@license: BSD Style.
"""

__all__ = ['PCA', 'SVD', 'PLSR', 'TuckerFactorAnalysis', 'PLSC', 'O2PLS',
           'RGCCA',
           'LinearRegression', 'LinearRegressionTV', 'RidgeRegressionTV',
           'LogisticRegression']

from sklearn.utils import check_arrays

import abc
import numpy as np
from numpy.linalg import pinv
from multiblock.utils import direct
import utils

import algorithms
import copy
import prox_ops
import schemes
import modes
import loss_functions
import start_vectors


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

    def get_max_iter(self):
        return self.get_algorithm()._get_max_iter()

    def set_max_iter(self, max_iter):
        self.get_algorithm()._set_max_iter(max_iter)

    def get_tolerance(self):
        self.get_algorithm()._get_tolerance()

    def set_tolerance(self, tolerance):
        self.get_algorithm()._set_tolerance(tolerance)

    def get_start_vector(self):
        return self.get_algorithm()._get_start_vector()

    def set_start_vector(self, start_vector):
        self.get_algorithm()._set_start_vector(start_vector)

    def get_algorithm(self):
        return self.algorithm

    def set_algorithm(self, algorithm):
        if not isinstance(self.algorithm, algorithms.BaseAlgorithm):
            raise ValueError('The algorithm must be an instance of ' \
                             '"BaseAlgorithm"')
        self.algorithm = algorithm

    def get_prox_op(self):
        return self.get_algorithm()._get_prox_op()

    def set_prox_op(self, prox_op):
        self.get_algorithm()._set_prox_op(prox_op)

    @abc.abstractmethod
    def get_transform(self, index=0):
        raise NotImplementedError('Abstract method "get_transform" must be '\
                                  'specialised!')

    def deflate(self, X, w, t, p, index=None):
        return X - np.dot(t, p.T)

    @abc.abstractmethod
    def fit(self, *X, **kwargs):
        raise NotImplementedError('Abstract method "fit" must be specialised!')


class PLSBaseMethod(BaseMethod):

    def __init__(self, algorithm=None, **kwargs):

        if algorithm == None:
            algorithm = algorithms.NIPALSAlgorithm()

        super(PLSBaseMethod, self).__init__(algorithm=algorithm, **kwargs)

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

    def set_scheme(self, scheme):
        self.get_algorithm()._set_scheme(scheme)

    def get_scheme(self):
        return self.get_algorithm()._get_scheme()

    def set_adjacency_matrix(self, adj_matrix):
        self.get_algorithm()._set_adjacency_matrix(adj_matrix)

    def get_adjacency_matrix(self):
        return self.get_algorithm()._set_adjacency_matrix()

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
                t = np.dot(X[i], w[i])  # / np.dot(w[i].T, w[i])

                # Sum of squares of t
                sst = np.sum(t ** 2.0)  # Faster than np.dot(t.T, t)

                # Loading vector
                p = np.dot(X[i].T, t)

                # Test for null variance
                if sst < self.algorithm.tolerance:
                    utils.warning('Scores of block X[%d] are too small at '
                                  'iteration %d' % (i, a))
                else:
                    # Loading vector
                    p /= sst

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
            self.Ws[i] = np.dot(self.W[i],
                                pinv(np.dot(self.P[i].T, self.W[i])))

        return self

    def get_transform(self, index=0):
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
            t = np.dot(X[i], self.get_transform(i))

            T.append(t)

        return T

    def fit_transform(self, *X, **kwargs):
        return self.fit(*X, **kwargs).transform(*X)


class PCA(PLSBaseMethod):

    def __init__(self, **kwargs):
#        prepro = kwargs.pop("preprocess", pp.PreprocessQueue([pp.Center(),
#                                                              pp.Scale()]))
        super(PCA, self).__init__(**kwargs)

    def get_transform(self, index=0):
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

        np.dot(U, np.dot(S, V.T)) == X
    """

    def __init__(self, **kwargs):
#        PCA.__init__(self, **kwargs)
        super(SVD, self).__init__(**kwargs)

    def get_transform(self, index=0):
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

    def get_transform(self, index=0):
        if index == 0:
            return self.Ws
        else:
            return self.C

    def deflate(self, X, w, t, p, index=None):
        if index == 0:
            return X - np.dot(t, p.T)  # Deflate X using its loadings
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

        self.B = np.dot(self.Ws, self.C.T)

        return self

    def predict(self, X):
        X = np.asarray(X)
#        if self.preproc != None and self.preproc[0] != None:
#            X = self.preproc[0].process(X)

        Ypred = np.dot(X, self.B)

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


class TuckerFactorAnalysis(PLSR):
    """ Tucker inner battery factor analysis, or PLS with symmetric deflation
    using the weights W and C.

    The model of each matrix is:
        X = T.W',
        Y = U.C'
    with the inner relation T = U + H, i.e. with T = U.Dy and U = T.Dx.

    Prediction is therefore performed as:
        Xhat = Y.C.Dy.W' = Y.By,
        Yhat = X.W.Dx.Q' = X.Bx.
    """

    def __init__(self, algorithm=None, **kwargs):
        if algorithm == None:
            algorithm = algorithms.NIPALSAlgorithm()

        super(TuckerFactorAnalysis, self).__init__(algorithm=algorithm,
                                                   **kwargs)

    def get_transform(self, index=0):
        if index == 0:
            return self.W
        else:  # index == 1
            return self.C

    def deflate(self, X, w, t, p, index=None):
        return X - np.dot(t, w.T)  # Deflate using their weights

    def fit(self, X, Y=None, **kwargs):
        Y = kwargs.pop('y', Y)

        super(TuckerFactorAnalysis, self).fit(X, Y, **kwargs)

        self.Dx = np.dot(pinv(self.T), self.U)
        self.Dy = np.dot(pinv(self.U), self.T)

        # Yhat = X.W.Dx.C' = X.Bx
        self.Bx = np.dot(self.W, np.dot(self.Dx, self.C.T))
        # Xhat = Y.C.Dy.W' = Y.By
        self.By = np.dot(self.C, np.dot(self.Dy, self.W.T))

        del self.B
        del self.Ws
        del self.P
        del self.Q

        return self

    def predict(self, X, Y=None, **kwargs):
        Y = kwargs.pop('y', Y)

        X = np.asarray(X)
        Ypred = np.dot(X, self.Bx)

        if Y != None:
            Y = np.asarray(Y)
            Xpred = np.dot(Y, self.By)

            return Ypred, Xpred

        return Ypred


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

    def get_transform(self, index=0):
        if index == 0:
            return self.Ws
        else:  # index == 1
            return self.Cs

    def deflate(self, X, w, t, p, index=None):
        return X - np.dot(t, p.T)  # Deflate using their loadings

    def fit(self, X, Y=None, **kwargs):
        Y = kwargs.pop('y', Y)
#        PLSR.fit(self, X, Y, **kwargs)
        super(PLSC, self).fit(X, Y, **kwargs)

        self.Cs = np.dot(self.C, pinv(np.dot(self.Q.T, self.C)))

        self.Dx = np.dot(pinv(self.T), self.U)
        self.Dy = np.dot(pinv(self.U), self.T)

        # Yhat = XW*DxQ' = XBx
        self.Bx = np.dot(self.Ws, np.dot(self.Dx, self.Q.T))
        # Xhat = YC*DyP' = YBy
        self.By = np.dot(self.Cs, np.dot(self.Dy, self.P.T))
#        self.Bx = np.dot(self.Ws, self.Q.T)               # Yhat = XW*Q' = XBx
#        self.By = np.dot(self.Cs, self.P.T)               # Xhat = XC*P' = YBy
        del self.B

        return self

    def predict(self, X, Y=None, **kwargs):
        Y = kwargs.pop('y', Y)

        X = np.asarray(X)
        Ypred = np.dot(X, self.Bx)

        if Y != None:
            Y = np.asarray(Y)
            Xpred = np.dot(Y, self.By)

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

    def fit(self, X, Y=None, **kwargs):

        Y = kwargs.pop('y', Y)
        if Y == None:
            raise ValueError("Both X and Y must be given!")
#        self.num_comp = kwargs.pop("num_comp", self.num_comp)

        self._check_inputs()
        X, Y = self._check_arrays(X, Y)

        prox_op = self.get_prox_op()
        if len(prox_op.parameter) != 0:
            Xparam = prox_op.parameter[0]
            Yparam = prox_op.parameter[1]
            joint_param = [Xparam[0], Yparam[0]]
            unique_x_param = [Xparam[1]]
            unique_y_param = [Yparam[1]]

            unique_x_op = copy.copy(prox_op)
            unique_x_op.parameter = unique_x_param
            unique_y_op = copy.copy(prox_op)
            unique_y_op.parameter = unique_y_param
            prox_op.parameter = joint_param
        else:
            unique_x_op = prox_ops.ProxOp()
            unique_y_op = prox_ops.ProxOp()
            prox_op = prox_ops.ProxOp()

        # Results matrices
        M, N1 = X.shape
        M, N2 = Y.shape
        self.Wo = np.zeros((N1, self.Ax))
        self.To = np.zeros((M,  self.Ax))
        self.Po = np.zeros((N1, self.Ax))
        self.Co = np.zeros((N2, self.Ay))
        self.Uo = np.zeros((M,  self.Ay))
        self.Qo = np.zeros((N2, self.Ay))

        alg = copy.deepcopy(self.algorithm)
#        svd = SVD(num_comp=self.A, algorithm=alg, **kwargs)
        tfa = TuckerFactorAnalysis(num_comp=self.A, algorithm=alg, **kwargs)
#        svd.set_prox_op(prox_op)
        tfa.set_prox_op(prox_op)
#        svd.fit(np.dot(X.T, Y))
        tfa.fit(X, Y)
#        W = svd.U
#        C = svd.V
        W = tfa.W
        C = tfa.C

        alg = copy.deepcopy(self.algorithm)
        eigsym = SVD(num_comp=1, algorithm=alg, **kwargs)
        eigsym.set_prox_op(unique_x_op)
        for a in xrange(self.Ax):
            T = np.dot(X, W)
            E = X - np.dot(T, W.T)
            TE = np.dot(T.T, E)
            eigsym.fit(TE)
            wo = eigsym.V
            s = eigsym.S
            if s < alg.tolerance:
                wo = np.zeros(wo.shape)
            to = np.dot(X, wo)
            toto = np.dot(to.T, to)
            Xto = np.dot(X.T, to)
            if toto > alg.tolerance:
                po = Xto / toto
            else:
                po = np.zeros(Xto.shape)

            self.Wo[:, a] = wo.ravel()
            self.To[:, a] = to.ravel()
            self.Po[:, a] = po.ravel()

            X = X - np.dot(to, po.T)

        alg = copy.deepcopy(self.algorithm)
        eigsym = SVD(num_comp=1, algorithm=alg, **kwargs)
        eigsym.set_prox_op(unique_y_op)
        for a in xrange(self.Ay):
            U = np.dot(Y, C)
            F = Y - np.dot(U, C.T)
            UF = np.dot(U.T, F)
            eigsym.fit(UF)
            co = eigsym.V
            s = eigsym.S
            if s < alg.tolerance:
                co = np.zeros(co.shape)
            uo = np.dot(Y, co)
            uouo = np.dot(uo.T, uo)
            Yuo = np.dot(Y.T, uo)
            if uouo > alg.tolerance:
                qo = Yuo / uouo
            else:
                qo = np.zeros(Yuo.shape)

            self.Co[:, a] = co.ravel()
            self.Uo[:, a] = uo.ravel()
            self.Qo[:, a] = qo.ravel()

            Y = Y - np.dot(uo, qo.T)

        alg_old = self.algorithm
        alg = copy.deepcopy(self.algorithm)
        alg._set_prox_op(prox_op)
        alg.adj_matrix = None
        alg.scheme = schemes.Horst()
        alg.mode = modes.NewA()
        self.algorithm = alg
        super(O2PLS, self).fit(X, Y, **kwargs)
        self.algorithm = alg_old

        return self

    def transform(self, X, Y=None, **kwargs):
        Y = kwargs.get('y', Y)

        if Y != None:
            X, Y = self._check_arrays(X, Y)

            if self.Ax > 0:
#                X -= self.means[0]
#                X /= self.stds[0]
                for a in xrange(self.Ax):
                    to = np.dot(X, self.Wo[:, [a]])
                    X = X - np.dot(to, self.Po[:, [a]].T)
#                To = np.dot(X, self.Wo)
#                X = X - np.dot(To, self.Po.T)
#                X *= self.stds[0]
#                X += self.means[0]

            if self.Ay > 0:
#                Y -= self.means[1]
#                Y /= self.stds[1]
                for a in xrange(self.Ay):
                    uo = np.dot(Y, self.Co[:, [a]])
                    Y = Y - np.dot(uo, self.Qo[:, [a]].T)
#                Uo = np.dot(Y, self.Co)
#                Y = Y - np.dot(Uo, self.Qo.T)
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
                    to = np.dot(X, self.Wo[:, [a]])
                    X = X - np.dot(to, self.Po[:, [a]].T)
#                To = np.dot(X, self.Wo)
#                X = X - np.dot(To, self.Po.T)
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
                to = np.dot(X, self.Wo[:, [a]])
                X = X - np.dot(to, self.Po[:, [a]].T)
#            To = np.dot(X, self.Wo)
#            if copy:
#                X = X - np.dot(To, self.Po.T)
#            else:
#                X -= np.dot(To, self.Po.T)

#            self.Bx = np.dot(self.Ws, np.dot(self.Dx, self.Q.T))
#            Ypred = (np.dot(X, self.Bx)*self.stds[1]) + self.means[1]
            Ypred = np.dot(X, self.Bx)

        if Y != None:
            Y = np.asarray(Y)
#            if copy:
#                Y = (Y - self.means[1]) / self.stds[1]
#            else:
#                Y -= self.means[1]
#                Y /= self.stds[1]
            for a in xrange(self.Ay):
                uo = np.dot(Y, self.Co[:, [a]])
                Y = Y - np.dot(uo, self.Qo[:, [a]].T)
#            Uo = np.dot(Y, self.Co)
#            if copy:
#                Y = Y - np.dot(Uo, self.Qo.T)
#            else:
#                Y -= np.dot(Uo, self.Qo.T)

#            Xpred = (np.dot(Y, self.By)*self.stds[0]) + self.means[0]
            Xpred = np.dot(Y, self.By)

        if X != None and Y != None:
            return Ypred, Xpred
        elif X != None:
            return Ypred
        else:  # Y != None
            return Xpred


class RGCCA(PLSBaseMethod):

    def __init__(self, num_comp=2, tau=None, **kwargs):

        super(RGCCA, self).__init__(num_comp=num_comp,
                                    algorithm=algorithms.RGCCAAlgorithm(tau),
                                    **kwargs)


class ContinuationRun(BaseMethod):

    def __init__(self, method, mus, algorithm=None, *args, **kwargs):
        if algorithm == None:
            algorithm = algorithms.ISTARegression()

        super(ContinuationRun, self).__init__(num_comp=1, algorithm=algorithm,
                                              *args, **kwargs)

        self.method = method
        self.mus = mus

    def get_transform(self, index=0):
        return self.beta

    def get_algorithm(self):
        return self.method.get_algorithm()

    def set_algorithm(self, algorithm):
        self.method.set_algorithm(algorithm)

    def fit(self, X, y, **kwargs):

        start_vector = self.method.get_start_vector()
        f = []
        for mu in self.mus:
            self.method.set_start_vector(start_vector)
            self.method.fit(X, y, mu=mu, early_stopping_mu=self.mus[-1],
                            **kwargs)

            utils.debug("continuation with mu = ", mu, \
                    ", es_mu = ", self.mus[-1], \
                    ", iterations =", self.method.get_algorithm().iterations)

            self.beta = self.method.get_transform()
            f = f + self.method.get_algorithm().f

            start_vector = start_vectors.IdentityStartVector(self.beta)

        self.method.get_algorithm().f = f
        self.method.get_algorithm().iterations = len(f)

        return self


class NesterovProximalGradientMethod(BaseMethod):

    __metaclass__ = abc.ABCMeta

    def __init__(self, algorithm=None, **kwargs):

        if algorithm == None:
            algorithm = algorithms.ISTARegression()

        super(NesterovProximalGradientMethod, self).__init__(num_comp=1,
                                                           algorithm=algorithm,
                                                           **kwargs)

    @abc.abstractmethod
    def fit(self, X, y, mu=None):

        raise NotImplementedError('Abstract method "fit" must be specialised!')

    def set_mu(self, mu):

        self.g.set_mu(mu)

    def get_mu(self):

        return self.g.get_mu()

    def get_g(self):

        return self.algorithm.g

    def set_g(self, g):

        self.algorithm.g = g

    def get_h(self):

        return self.algorithm.h

    def set_h(self, h):

        self.algorithm.h = h


class LinearRegression(NesterovProximalGradientMethod):

    def __init__(self, X=None, y=None, **kwargs):

        super(LinearRegression, self).__init__(**kwargs)

        if X != None and y != None:
            self.set_g(loss_functions.LinearRegressionError(X, y))

    def fit(self, X, y, **kwargs):

        self.set_g(loss_functions.LinearRegressionError(X, y))

        if self.get_g() == None:
            raise ValueError('The function g must be given either at ' \
                             'construction or when fitting')

        self.beta = self.algorithm.run(X, **kwargs)

        return self

    def get_transform(self, index=0):

        return self.beta

    def predict(self, X, **kwargs):

        yhat = np.dot(X, self.beta)

        return yhat


class LinearRegressionTV(NesterovProximalGradientMethod):

    def __init__(self, gamma, shape, mu=None, mask=None, **kwargs):

        super(LinearRegressionTV, self).__init__(**kwargs)

        self._tv = loss_functions.TotalVariation(gamma, shape, mu, mask)

    def fit(self, X, y, mu=None, **kwargs):

        self._reg = loss_functions.LinearRegressionError(X, y)
        self._combo = loss_functions.CombinedNesterovLossFunction(self._reg,
                                                                  self._tv)
        self.set_g(self._combo)

        if mu != None:
            mu_old = self._tv.get_mu()
            self._tv.set_mu(mu)

        self.beta = self.algorithm.run(X, **kwargs)

        if mu != None:
            self._tv.set_mu(mu_old)

        return self

    def get_transform(self, **kwargs):

        return self.beta

    def predict(self, X, **kwargs):

        yhat = np.dot(X, self.beta)

        return yhat


class LogisticRegression(NesterovProximalGradientMethod):

    def __init__(self, **kwargs):

        super(LogisticRegression, self).__init__(**kwargs)

    def fit(self, X, y, **kwargs):

        self.set_g(loss_functions.LogisticRegressionError(X, y))

        self.beta = self.algorithm.run(X, **kwargs)

        return self

    def get_transform(self, index=0):

        return self.beta

    def predict(self, X, **kwargs):

        yhat = np.dot(X, self.beta)

        return yhat


class ExcessiveGapMethod(BaseMethod):

    __metaclass__ = abc.ABCMeta

    def __init__(self, algorithm=None, **kwargs):

        if algorithm == None:
            algorithm = algorithms.ExcessiveGapRidgeRegression()

        super(ExcessiveGapMethod, self).__init__(num_comp=1,
                                                 algorithm=algorithm,
                                                 **kwargs)

    @abc.abstractmethod
    def fit(self, X, y, **kwargs):

        raise NotImplementedError('Abstract method "fit" must be specialised!')

    def get_g(self):

        return self.algorithm.g

    def set_g(self, g):

        self.algorithm.g = g

    def get_h(self):

        return self.algorithm.h

    def set_h(self, h):

        self.algorithm.h = h


class RidgeRegressionTV(ExcessiveGapMethod):

    def __init__(self, l, gamma, shape, mu=None, mask=None, **kwargs):

        super(RidgeRegressionTV, self).__init__(**kwargs)

        self.l = l
        self.set_h(loss_functions.TotalVariation(gamma, shape, mu, mask))

    def fit(self, X, y, **kwargs):

        self.set_g(loss_functions.RidgeRegression(X, y, self.l))

        self.beta = self.algorithm.run(X, y, **kwargs)

        return self

    def get_transform(self, index=0):

        return self.beta

    def predict(self, X, **kwargs):

        yhat = np.dot(X, self.beta)

        return yhat