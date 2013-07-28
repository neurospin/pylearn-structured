# -*- coding: utf-8 -*-
"""
The :mod:`multiblock.models` module includes several different multiblock
models.

@author:  Tommy Löfstedt <tommy.loefstedt@cea.fr>
@email:   tommy.loefstedt@cea.fr
@license: TBD
"""

__all__ = ['PCA', 'SVD', 'PLSR', 'TuckerFactorAnalysis', 'PLSC', 'O2PLS',
           'RGCCA',

           'ContinuationRun', 'Continuation',

           'LinearRegression',
           'Lasso', 'ElasticNet', 'LinearRegressionL1L2',
           'LinearRegressionTV', 'LinearRegressionL1TV',
           'LinearRegressionL1L2TV', 'ElasticNetTV',
           'LinearRegressionGL',

           'RidgeRegression',
           'RidgeRegressionL1',
           'RidgeRegressionTV', 'RidgeRegressionL1TV',

           'LogisticRegression', 'LogisticRegressionGL',

           'EGMRidgeRegression', 'EGMLinearRegressionL1L2', 'EGMElasticNet',
           'EGMRidgeRegressionTV', 'EGMRidgeRegressionL1TV',
           'EGMElasticNetTV'
          ]

from sklearn.utils import check_arrays

import abc
import numpy as np
from numpy.linalg import pinv
from utils import direct
import utils

import algorithms
import copy
import prox_ops
import schemes
import modes
import loss_functions
import start_vectors


class BaseModel(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, algorithm, num_comp=2, norm_dir=False):

        super(BaseModel, self).__init__()

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

        return self.get_algorithm()._get_tolerance()

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
        # TODO: This does not works since algorithms are not stateless.
        # They must be made so. In particular, any methods g and h set will
        # not be available after this.
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


class PLSBaseModel(BaseModel):

    def __init__(self, algorithm=None, **kwargs):

        if algorithm == None:
            algorithm = algorithms.NIPALSAlgorithm()

        super(PLSBaseModel, self).__init__(algorithm=algorithm, **kwargs)

    def _check_inputs(self):

        super(PLSBaseModel, self)._check_inputs()

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

        Parameters
        ----------
        X : The matrices to center

        Returns
        -------
        Centered X
        """
        return self.Ws[index]

    def transform(self, *X, **kwargs):

        X = self._check_arrays(*X)

        T = []
        for i in xrange(self.n):
            # Apply rotation
            t = np.dot(X[i], self.get_transform(i))

            T.append(t)

        return T

    def fit_transform(self, *X, **kwargs):
        return self.fit(*X, **kwargs).transform(*X)


class PCA(PLSBaseModel):

    def __init__(self, **kwargs):

        super(PCA, self).__init__(**kwargs)

    def get_transform(self, index=0):

        return self.P

    def fit(self, *X, **kwargs):

        super(PCA, self).fit(X[0], **kwargs)
        self.T = self.T[0]
        self.P = self.W[0]
        del self.W
        del self.Ws

        return self

    def transform(self, *X, **kwargs):

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

        super(SVD, self).__init__(**kwargs)

    def get_transform(self, index=0):

        return self.V

    def fit(self, *X, **kwargs):

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


class PLSR(PLSBaseModel):
    """Performs PLS regression between two matrices X and Y.
    """
    def __init__(self, algorithm=None, **kwargs):

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
#        PLSBaseModel.fit(self, X, Y, **kwargs)
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
#            T = PLSBaseModel.transform(self, X, Y, **kwargs)
            T = super(PLSR, self).transform(X, Y, **kwargs)
        else:
#            T = PLSBaseModel.transform(self, X, **kwargs)
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


class RGCCA(PLSBaseModel):

    def __init__(self, num_comp=2, tau=None, **kwargs):

        super(RGCCA, self).__init__(num_comp=num_comp,
                                    algorithm=algorithms.RGCCAAlgorithm(tau),
                                    **kwargs)


class ContinuationRun(BaseModel):

    def __init__(self, model, gaps=None, mus=None, algorithm=None,
                 *args, **kwargs):
        """Performs continuation for the given method. I.e. runs the method
        with sucessively smaller values of mu and uses the output from the
        use of one mu as start vector in the run with the next smaller mu.

        Parameters
        ----------
        model : The NesterovProximalGradient model to perform continuation
                on.

        gaps : A list of successively smaller gap values. The gaps are used as
                terminating condition for the continuation run. Mu is computed
                from this list of gaps. Note that only one of gaps and mus can
                be given.

        mus : A list of successively smaller values of mu, the regularisation
                parameter in the Nesterov smoothing. The gaps are
                computed from this list of mus. Note that only one of mus and
                gaps can be given.

        algorithm : The particular algorithm to use.
        """
        if algorithm == None:
            algorithm = model.get_algorithm()
        else:
            model.set_algorithm(algorithm)

        super(ContinuationRun, self).__init__(num_comp=1, algorithm=algorithm,
                                              *args, **kwargs)
        self.model = model
        self.gaps = gaps
        self.mus = mus

    def get_transform(self, index=0):

        return self._beta

    def get_algorithm(self):

        return self.model.get_algorithm()

    def set_algorithm(self, algorithm):

        self.model.set_algorithm(algorithm)

    def fit(self, X, y, **kwargs):

        start_vector = self.model.get_start_vector()
        f = []
        self.model.set_data(X, y)

        if self.mus != None:
            lst = self.mus
        else:
            lst = self.gaps

        beta_new = 0

        for item in lst:
            if self.mus != None:
                self.model.set_mu(item)
#                self.model.set_tolerance(self.model.compute_gap(item))
            else:
#                self.model.set_tolerance(item)
                self.model.set_mu(self.model.compute_mu(item))

            self.model.set_start_vector(start_vector)
            self.model.fit(X, y, **kwargs)

            utils.debug("Continuation with mu = ", self.model.get_mu(), \
                    ", tolerance = ", self.model.get_tolerance(), \
                    ", iterations = ", self.model.get_algorithm().iterations)

            beta_old = beta_new
            beta_new = self.model.get_transform()
            f = f + self.model.get_algorithm().f[1:]  # Skip the first, same

#            if len(f) > 1 and abs(f[-2] - f[-1]) < self.model.get_tolerance():
#                print "Converged in f!!"
#                break

            if utils.norm1(beta_old - beta_new) < self.model.get_tolerance():
                print "Converged in beta!!"
                print utils.norm1(beta_old - beta_new)
                print self.model.get_tolerance()
                break

            start_vector = start_vectors.IdentityStartVector(beta_new)

        self._beta = beta_new
        self.model.get_algorithm().f = f
        self.model.get_algorithm().iterations = len(f)

        return self


class Continuation(BaseModel):

    def __init__(self, model, iterations=100, gap=None, algorithm=None,
                 *args, **kwargs):
        """Performs continuation for the given model. I.e. builds
        NesterovProximalGradientMethod models with sucessively, and optimally,
        smaller values of mu and uses the output from the use of one mu as
        start vector in the fit of model with the next smaller mu.

        Parameters
        ----------
        model : The NesterovProximalGradient model to perform continuation
                on.

        iterations : The number of iterations in each continuation.

        gap : The gap to use in the first continuation. Default is
                mu = max(abs(cov(X,y))) and then
                gap = model.compute_gap(mu).

        algorithm : The particular algorithm to use.
        """
        if algorithm == None:
            algorithm = model.get_algorithm()
        else:
            model.set_algorithm(algorithm)

        super(Continuation, self).__init__(num_comp=1, algorithm=algorithm,
                                              *args, **kwargs)

        self.model = model
        self.iterations = iterations
        self.gap = gap

    def get_transform(self, index=0):

        return self._beta

    def get_algorithm(self):

        return self.model.get_algorithm()

    def set_algorithm(self, algorithm):

        self.model.set_algorithm(algorithm)

    def fit(self, X, y, **kwargs):

        max_iter = self.get_max_iter()
        self.model.set_max_iter(self.iterations)
        self.model.set_data(X, y)
        start_vector_mu = self.model.get_start_vector()
#        start_vector_nomu = self.model.get_start_vector()
        if self.gap == None:
            mu = max(np.max(np.abs(utils.corr(X, y))), 0.01)  # Necessary?
            gap_mu = self.model.compute_gap(mu)
        else:
            gap_mu = self.gap
            mu = self.model.compute_mu(gap_mu)

        gap_nomu = gap_mu
        beta_old = 0.0
        beta_new = 0.0

        tau = 1.1
        eta = 2.0
        mu_zero = 5e-12

        f = []
        for i in xrange(1, max_iter + 1):

#            self.model.set_max_iter(float(self.iterations) / float(i))

            # With computed mu
            self.model.set_mu(mu)
            self.model.set_start_vector(start_vector_mu)
            self.model.fit(X, y, **kwargs)
            f = f + self.model.get_algorithm().f[1:]  # Skip the first, same
            beta_old = beta_new
            beta_new = self.model.get_transform()
            start_vector_mu = start_vectors.IdentityStartVector(beta_new)

            self.model.set_start_vector(start_vector_mu)
            alpha_mu = self.model.get_g().alpha()
            gap_mu = self.model.phi(beta=beta_new, alpha=alpha_mu) \
                        - self.model.phi(beta=None, alpha=alpha_mu)

            utils.debug("With mu: Continuation with mu = ",
                                self.model.get_mu(), \
                    ", tolerance = ", self.model.get_tolerance(), \
                    ", iterations = ", self.model.get_algorithm().iterations, \
                    ", gap = ", gap_mu)

            # With mu "very small"
            self.model.set_mu(min(mu, mu_zero))
            alpha_nomu = self.model.get_g().alpha(beta_new, min(mu, mu_zero))
            gap_nomu = self.model.phi(beta=beta_new, alpha=alpha_nomu) \
                        - self.model.phi(beta=None, alpha=alpha_nomu,
                                         mu=min(mu, mu_zero))

            utils.debug("No mu: Continuation with mu = ",
                                self.model.get_mu(), \
                    ", tolerance = ", self.model.get_tolerance(), \
                    ", iterations = ", self.model.get_algorithm().iterations, \
                    ", gap = ", gap_nomu)

            if gap_nomu < self.model.get_tolerance():
                print "Converged in G!!"
                break

#            if len(f) > 1 and abs(f[-2] - f[-1]) < self.model.get_tolerance():
#                print "Converged in f!!"
#                break

            if utils.norm1(beta_old - beta_new) < self.model.get_tolerance():
                print "Converged in beta!!"
                break

            self.model.set_mu(mu)
            mu = min(mu, self.model.compute_mu(gap_nomu))
            if gap_mu < gap_nomu / (2.0 * tau):
                mu = mu / eta

            mu = max(mu, utils.TOLERANCE)

        self._beta = beta_new

        self.model.get_algorithm().f = f
        self.model.get_algorithm().iterations = len(f)

        self.model.set_max_iter(max_iter)

        return self


class NesterovProximalGradientMethod(BaseModel):

    def __init__(self, algorithm=None, **kwargs):

        if algorithm == None:
#            algorithm = algorithms.ISTARegression()
#            algorithm = algorithms.FISTARegression()
            algorithm = algorithms.MonotoneFISTARegression()

        super(NesterovProximalGradientMethod, self).__init__(num_comp=1,
                                                           algorithm=algorithm,
                                                           **kwargs)

    def fit(self, X, y, **kwargs):
        """Fit the model to the given data.

        Parameters
        ----------
        X : The independent variables.

        y : The dependent variable.

        Returns
        -------
        self: The model object.
        """
        self.set_data(X, y)

        self._beta = self.algorithm.run(X, y, **kwargs)

        return self

    def f(self, *args, **kwargs):

        return self.get_g().f(*args, **kwargs) \
                + self.get_h().f(*args, **kwargs)

    # TODO: Decide if phi(beta, alpha) should be in the general API for all
    # Nesterov functions.
    def phi(self, beta=None, alpha=None, *args, **kwargs):
        """This function returns the associated loss function value for the
        given alpha and beta.
        """
        return self.get_g().phi(beta, alpha) + self.get_h().f(beta)

    def beta(self, alpha=None, mu=None):
        """Computes the beta that minimises the dual function value for the
        current computed alpha.
        """
        dual_model = ConstantNesterovModelCopy(self, alpha)
#        dual_model.set_h(loss_functions.ZeroErrorFunction())
        if mu != None:
            dual_model.set_mu(mu)
        dual_model.fit(*self.get_data())

        return dual_model._beta

    def alpha(self, beta=None, mu=None):
        """Computes the alpha that maximises the smoothed loss function for the
        current computed beta.
        """
        g = self.get_g()
        if isinstance(g.a, loss_functions.NesterovFunction) and \
                isinstance(g.b, loss_functions.NesterovFunction):
            return g.alpha(beta, mu)

        elif isinstance(g.a, loss_functions.NesterovFunction):
            return g.a.alpha(beta, mu)

        elif isinstance(g.b, loss_functions.NesterovFunction):
            return g.b.alpha(beta, mu)

        else:
            raise ValueError("The given functions must be Nesterov functions")

    def get_transform(self, **kwargs):

        return self._beta

    def compute_gap(self, mu, max_iter=100):

        def f(eps):
            return self.compute_mu(eps) - mu

        D = self.get_g().num_compacts() / 2.0
        lb = 2.0 * D * mu
        ub = lb * 100.0
        a = f(lb)
        b = f(ub)

        # Do a binary search for the upper limit. It seems that when mu become
        # very small (e.g. < 1e-8), the lower bound is not correct. Therefore
        # we do a full search for both the lower and upper bounds.
        for i in xrange(max_iter):

            if a > 0.0 and b > 0.0 and a < b:
                ub = lb
                lb /= 2.0
            elif a < 0.0 and b < 0.0 and a < b:
                lb = ub
                ub *= 2.0
            elif a > 0.0 and b > 0.0 and a > b:
                lb = ub
                ub *= 2.0
            elif a < 0.0 and b < 0.0 and a > b:
                ub = lb
                lb /= 2.0
            else:
                break

            a = f(lb)
            b = f(ub)

#            print "lb:", lb, ", f(lb):", a
#            print "ub:", ub, ", f(ub):", b

        bm = algorithms.BisectionMethod(max_iter=max_iter)
        bm.run(utils.AnonymousClass(f=f), lb, ub)

        return bm.x

    def compute_mu(self, eps):

        g = self.get_g()
        D = g.num_compacts() / 2.0

        def f(mu):
            return -(eps - mu * D) / g.Lipschitz(mu)

        gs = algorithms.GoldenSectionSearch()
        gs.run(utils.AnonymousClass(f=f), utils.TOLERANCE, eps / (2.0 * D))

#        ts = algorithms.TernarySearch(utils.AnonymousClass(f=f))
#        ts.run(utils.TOLERANCE, eps / D)
#        print "gs.iterations: ", gs.iterations
#        print "ts.iterations: ", ts.iterations

        return gs.x

    def predict(self, X, **kwargs):

        yhat = np.dot(X, self.get_transform())

        return yhat

    def set_mu(self, mu):

        self.get_g().set_mu(mu)

    def get_mu(self):

        return self.get_g().get_mu()

    def get_g(self):

        return self.algorithm.g

    def set_g(self, g):

        self.algorithm.g = g

    def get_h(self):

        return self.algorithm.h

    def set_h(self, h):

        self.algorithm.h = h

    def set_data(self, X, y):

        self.get_g().set_data(X, y)

    def get_data(self):

        return self.get_g().get_data()


class ConstantNesterovModelCopy(NesterovProximalGradientMethod):

    def __init__(self, model, alpha=None):
        """Constructs a copy of the given model's Nesterov functions, such that
        all alphas and gradients are constant.
        """
        super(ConstantNesterovModelCopy, self).__init__()

        # TODO: Potential memory issues here, since algorithms are not
        # stateless!

        # TODO: Potential memory issues here, since algorithms are not
        # stateless!

        # Copy the algorithm
        self.set_algorithm(copy.copy(model.get_algorithm()))

        g = model.get_g()
        if isinstance(g.a, loss_functions.NesterovFunction) \
                and isinstance(g.b, loss_functions.NesterovFunction):

            const = loss_functions.ConstantNesterovCopy(g)
            loss = const

        elif isinstance(g.a, loss_functions.NesterovFunction):

            const = loss_functions.ConstantNesterovCopy(g.a)
            loss = loss_functions.CombinedNesterovLossFunction(g.b, const)

        elif isinstance(g.b, loss_functions.NesterovFunction):

            const = loss_functions.ConstantNesterovCopy(g.b)
            loss = loss_functions.CombinedNesterovLossFunction(g.a, const)

        if alpha != None:
            const.set_alpha(alpha)

        self.set_g(loss)
        self.set_h(model.get_h())

    # TODO: Decide if phi(beta, alpha) should be in the general API for all
    # Nesterov functions.
    def phi(self, beta=None, alpha=None, mu=None, *args, **kwargs):
        """This function returns the associated loss function value for the
        given alpha and beta.
        """
        if mu == None:
            mu = self.get_mu()

        print "Computing phi!"
        return self.get_g().phi(beta, alpha) + self.get_h().f(beta)


#class ConstantNesterovModel(NesterovProximalGradientMethod):
#
#    def __init__(self, model):
#        """Constructs a constant copy of the given model's functions.
#        """
#        super(ConstantNesterovModel, self).__init__()
#
#        # TODO: Potential memory issues here, since algorithms are not yet
#        # stateless!
#
#        # Copy the algorithm
#        self.set_algorithm(copy.copy(model.get_algorithm()))
#
#        # Copy the loss functions
#        self.set_h(copy.copy(model.get_h()))
#
#        g = model.get_g()
#        if isinstance(g.a, loss_functions.NesterovFunction):
#            b = loss_functions.ConstantNesterovFunction(g.a.gamma,
#                                                        g.a.get_mu(),
#                                                        g.a.A(),
#                                                        g.a.alpha(),
#                                                        g.a.num_compacts())
#        else:
#            a = copy.copy(g.b)
#
#        if isinstance(g.b, loss_functions.NesterovFunction):
#            b = loss_functions.ConstantNesterovFunction(g.b.gamma,
#                                                        g.b.get_mu(),
#                                                        g.b.A(),
#                                                        g.b.alpha(),
#                                                        g.b.num_compacts())
#        else:
#            a = copy.copy(g.a)
#
#        self.set_g(loss_functions.CombinedNesterovLossFunction(a, b))


class LinearRegression(NesterovProximalGradientMethod):

    def __init__(self, **kwargs):

        super(LinearRegression, self).__init__(**kwargs)

        self.set_g(loss_functions.LinearRegressionError())


class Lasso(NesterovProximalGradientMethod):
    """Lasso (linear regression + L1 constraint).

    Optimises the function

        f(b) = ||y - X.b||² + l.||b||_1,

    where ||.||_1 is the L1 norm and ||.||² is the squared L2 norm.

    Parameters
    ----------
    l: The Lasso parameter.
    """
    def __init__(self, l, **kwargs):

        super(Lasso, self).__init__(**kwargs)

        self.set_g(loss_functions.LinearRegressionError())
        self.set_h(loss_functions.L1(l))


class LinearRegressionTV(NesterovProximalGradientMethod):
    """Linear regression with total variation constraint.

    Optimises the function

        f(b) = ||y - X.b||² + gamma.TV(b),

    where ||.||² is the squared L2 norm and TV(.) is the total variation
    constraint.

    Parameters
    ----------
    gamma: The TV regularisation parameter.

    shape: The shape of the 3D image. Must be a 3-tuple. If the image is 2D,
           let the Z dimension be 1, and if the "image" is 1D, let the Y and
           Z dimensions be 1. The tuple must be on the form (Z, Y, X).

    mu   : The Nesterov function regularisation parameter.

    mask : A 1-dimensional mask representing the 3D image mask. Must be a
           list of 1s and 0s.
    """
    def __init__(self, gamma, shape, mu=None, mask=None, **kwargs):

        super(LinearRegressionTV, self).__init__(**kwargs)

        tv = loss_functions.TotalVariation(gamma, shape=shape, mu=mu,
                                           mask=mask)
        lr = loss_functions.LinearRegressionError()

        self.set_g(loss_functions.CombinedNesterovLossFunction(lr, tv))

    def compute_mu(self, eps):

        g = self.get_g()
        lr = g.a
        tv = g.b

        D = tv.num_compacts() / 2.0
        A = tv.Lipschitz(1.0)
        l = lr.Lipschitz()

        return (-2.0 * D * A + np.sqrt((2.0 * D * A) ** 2.0 \
                + 4.0 * D * l * eps * A)) / (2.0 * D * l)

    def compute_gap(self, mu, max_iter=100):

        g = self.get_g()
        lr = g.a
        tv = g.b

        D = tv.num_compacts() / 2.0
        A = tv.Lipschitz(1.0)
        l = lr.Lipschitz()

        return ((2.0 * mu * D * l + 2.0 * D * A) ** 2.0 \
                - (2.0 * D * A) ** 2.0) / (4.0 * D * l * A)


class LinearRegressionL1TV(LinearRegressionTV):
    """Linear regression with total variation and L1 constraints.

    Optimises the function

        f(b) = ||y - X.b||² + l.||b||_1 + gamma.TV(b),

    where ||.||_1 is the L1 norm, ||.||² is the squared L2 norm and TV(.) is
    the total variation constraint.

    Parameters
    ----------
    l    : The L1 parameter.

    gamma: The TV regularisation parameter.

    shape: The shape of the 3D image. Must be a 3-tuple. If the image is 2D,
           let the Z dimension be 1, and if the "image" is 1D, let the Y and
           Z dimensions be 1. The tuple must be on the form (Z, Y, X).

    mu   : The Nesterov function regularisation parameter.

    mask : A 1-dimensional mask representing the 3D image mask. Must be a
           list of 1s and 0s.
    """

    def __init__(self, l, gamma, shape, mu=None, mask=None, **kwargs):

        super(LinearRegressionL1TV, self).__init__(gamma, shape, mu, mask,
                                                   **kwargs)

#        lr = loss_functions.LinearRegressionError()
#        tv = loss_functions.TotalVariation(gamma, shape=shape, mu=mu,
#                                           mask=mask)
#
#        self.set_g(loss_functions.CombinedNesterovLossFunction(lr, tv))
        self.set_h(loss_functions.L1(l))


class LinearRegressionL1L2TV(LinearRegressionTV):
    """Linear regression with L1, L2 and total variation constraints.

    Optimises the function

        f(b) = ||y - X.b||² + l.||b||_1 + (k / 2.0).||b||² + gamma.TV(b),

    where ||.||_1 is the L1 norm, ||.||² is the squared L2 norm and TV(.) is
    the total variation constraint.

    Parameters
    ----------
    l    : The L1 regularisation parameter.

    k    : The L2 regularisation parameter.

    gamma: The TV regularisation parameter.

    shape: The shape of the 3D image. Must be a 3-tuple. If the image is 2D,
           let the Z dimension be 1, and if the "image" is 1D, let the Y and
           Z dimensions be 1. The tuple must be on the form (Z, Y, X).

    mu   : The Nesterov function regularisation parameter.

    mask : A 1-dimensional mask representing the 3D image mask. Must be a
           list of 1s and 0s.
    """
    def __init__(self, l, k, gamma, shape, mu=None, mask=None, **kwargs):

        super(LinearRegressionL1L2TV, self).__init__(gamma, shape, mu, mask,
                                                     **kwargs)

#        lr = loss_functions.LinearRegressionError()
#        tv = loss_functions.TotalVariation(gamma, shape=shape, mu=mu,
#                                           mask=mask)
#
#        self.set_g(loss_functions.CombinedNesterovLossFunction(lr, tv))
        self.set_h(loss_functions.L1L2(l, k))


class ElasticNet(NesterovProximalGradientMethod):
    """ElasticNet in linear regression.

    Optimises the function

        f(b) = ||y - X.b||² + l.||b||_1 + (1 - l).1/2.||b||²,

    where ||.||_1 is the L1 norm and ||.||² is the squared L2 norm.

    Parameters
    ----------
    l : The L1 and L2 parameter.
    """
    def __init__(self, l, **kwargs):

        super(ElasticNet, self).__init__(**kwargs)

        self.set_g(loss_functions.LinearRegressionError())
        self.set_h(loss_functions.ElasticNet(l))


class ElasticNetTV(LinearRegressionTV):
    """Linear regression with total variation and Elastic Net constraints.

    Optimises the function

        f(b) = ||y - X.b||² + l.||b||_1 + ((1 - l)/2).||b||² + gamma.TV(b),

    where ||.||_1 is the L1 norm, ||.||² is the squared L2 norm and TV(.) is
    the total variation constraint.

    Parameters
    ----------
    l    : The Elastic Net parameter.

    gamma: The TV regularisation parameter.

    shape: The shape of the 3D image. Must be a 3-tuple. If the image is 2D,
           let the Z dimension be 1, and if the "image" is 1D, let the Y and
           Z dimensions be 1. The tuple must be on the form (Z, Y, X).

    mu   : The Nesterov function regularisation parameter.

    mask : A 1-dimensional mask representing the 3D image mask. Must be a
           list of 1s and 0s.
    """
    def __init__(self, l, gamma, shape, mu=None, mask=None, **kwargs):

        super(ElasticNetTV, self).__init__(gamma, shape, mu, mask, **kwargs)

#        lr = loss_functions.LinearRegressionError()
#        tv = loss_functions.TotalVariation(gamma, shape=shape, mu=mu,
#                                           mask=mask)
#
#        self.set_g(loss_functions.CombinedNesterovLossFunction(lr, tv))
        self.set_h(loss_functions.ElasticNet(l))


class LinearRegressionL1L2(NesterovProximalGradientMethod):
    """Linear regression with L1 and L2 regularisation.

    Optimises the function

        f(b) = ||y - X.b||² + l.||b||_1 + (k / 2).||b||²,

    where ||.||_1 is the L1 norm and ||.||² is the squared L2 norm.

    Parameters
    ----------
    l : The L1 parameter.

    k : The L2 parameter.
    """
    def __init__(self, l, k, **kwargs):

        super(LinearRegressionL1L2, self).__init__(**kwargs)

        self.set_g(loss_functions.LinearRegressionError())
        self.set_h(loss_functions.L1L2(l, k))


class LinearRegressionGL(NesterovProximalGradientMethod):
    """Linear regression with group lasso constraint.

    Optimises the function

        f(b) = ||y - X.b||² + gamma.GL(b),

    where ||.||² is the squared L2 norm and GL(.) is the group lasso
    constraint.

    Parameters
    ----------
    gamma : The GL regularisation parameter.

    num_variables : The number of variable being regularised.

    groups : A list of lists, with the outer list being the groups and the
            inner lists the variables in the groups. E.g. [[1,2],[2,3]]
            contains two groups ([1,2] and [2,3]) with variable 1 and 2 in the
            first group and variables 2 and 3 in the second group.

    mu : The Nesterov function regularisation parameter.

    weights : Weights put on the groups. Default is weight 1 for each group.
    """
    def __init__(self, gamma, num_variables, groups, mu=None, weights=None,
                 **kwargs):

        super(LinearRegressionGL, self).__init__(**kwargs)

        lr = loss_functions.LinearRegressionError()
        gl = loss_functions.GroupLassoOverlap(gamma, num_variables, groups,
                                              mu, weights)

        self.set_g(loss_functions.CombinedNesterovLossFunction(lr, gl))

    def compute_mu(self, eps):

        g = self.get_g()
        lr = g.a
        gl = g.b

        D = gl.num_compacts() / 2.0
        A = gl.Lipschitz(1.0)
        l = lr.Lipschitz()

        return (-2.0 * D * A + np.sqrt((2.0 * D * A) ** 2.0 \
                + 4.0 * D * l * eps * A)) / (2.0 * D * l)

    def compute_gap(self, mu, max_iter=100):

        g = self.get_g()
        lr = g.a
        gl = g.b

        D = gl.num_compacts() / 2.0
        A = gl.Lipschitz(1.0)
        l = lr.Lipschitz()

        return ((2.0 * mu * D * l + 2.0 * D * A) ** 2.0 \
                - (2.0 * D * A) ** 2.0) / (4.0 * D * l * A)


class RidgeRegression(NesterovProximalGradientMethod):
    """Ridge regression.

    Optimises the function

        f(b) = (1 / 2).||y - X.b||² + (l / 2).||b||²

    Parameters
    ----------
    l : The ridge parameter.
    """
    def __init__(self, l, **kwargs):

        super(RidgeRegression, self).__init__(**kwargs)

        self.set_g(loss_functions.RidgeRegression(l))


class RidgeRegressionL1(RidgeRegression):
    """Ridge regression with L1 regularisation, i.e. linear regression with L1
    and L2 constraints.

    Optimises the function

        f(b) = ||y - X.b||² + l.||b||_1 + (k / 2).||b||²,

    where ||.||_1 is the L1 norm and ||.||² is the squared L2 norm.

    Parameters
    ----------
    l : The L1 parameter.

    k : The L2 parameter.
    """
    def __init__(self, l, k, **kwargs):

        super(RidgeRegression, self).__init__(k, **kwargs)

        self.set_h(loss_functions.L1(l))


class RidgeRegressionTV(RidgeRegression):
    """Ridge regression with total variation constraint, i.e. linear regression
    with L2 and TV constraints.

    Optimises the function

        f(b) = ||y - X.b||² + (l / 2).||b||² + gamma.TV(b),

    where ||.||² is the squared L2 norm and TV(.) is the total variation
    constraint.

    Parameters
    ----------
    l: The L1 regularisation parameter.

    gamma: The TV regularisation parameter.

    shape: The shape of the 3D image. Must be a 3-tuple. If the image is 2D,
           let the Z dimension be 1, and if the "image" is 1D, let the Y and
           Z dimensions be 1. The tuple must be on the form (Z, Y, X).

    mu   : The Nesterov function regularisation parameter.

    mask : A 1-dimensional mask representing the 3D image mask. Must be a
           list of 1s and 0s.
    """
    def __init__(self, l, gamma, shape, mu=None, mask=None, compress=True,
                 **kwargs):

        super(RidgeRegressionTV, self).__init__(l, **kwargs)

        tv = loss_functions.TotalVariation(gamma, shape=shape, mu=mu,
                                           mask=mask, compress=compress)

        rr = self.get_g()
        self.set_g(loss_functions.CombinedNesterovLossFunction(rr, tv))


class RidgeRegressionL1TV(RidgeRegressionTV):
    """Ridge regression with L1 and Total variation regularisation, i.e. linear
    regression with L1, L2 and TV constraints.

    Optimises the function

        f(b) = ||y - X.b||² + l.||b||_1 + (k / 2).||b||² + gamma.TV(b),

    where ||.||_1 is the L1 norm, ||.||² is the squared L2 norm and TV is the
    total variation function.

    Parameters
    ----------
    l : The L1 regularisation parameter.

    k : The L2 regularisation parameter.

    gamma : The TV regularisation parameter.

    mu : The Nesterov function regularisation parameter.

    mask : A 1-dimensional mask representing the 3D image mask. Must be a
           list of 1s and 0s.
    """
    def __init__(self, l, k, gamma, shape, mu=None, mask=None, compress=True,
                 **kwargs):

        super(RidgeRegressionL1TV, self).__init__(k, gamma, shape=shape,
                                                  mu=mu, mask=mask,
                                                  compress=compress, **kwargs)
        self.set_h(loss_functions.L1(l))

        self._l1 = loss_functions.SmoothL1(k, num_variables=np.prod(shape),
                                           mu=1e-12, mask=mask)
        # TODO: Reuse the A matrices from self.get_g().b
        self._tv = loss_functions.TotalVariation(gamma, shape=shape, mu=mu,
                                                 mask=mask, compress=False)

    # TODO: Decide if phi(beta, alpha) should be in the general API for all
    # Nesterov functions.
    def phi(self, beta=None, alpha=None, mu=None, *args, **kwargs):
        """This function returns the associated loss function value for the
        given alpha and beta.

        If alpha or beta is not given, they are computed.
        """
        if mu == None:
            mu = self.get_mu()

        if alpha == None:

            alpha = self.get_g().alpha(beta)

        elif beta == None:

            rr = self.get_g().a
            X, y = self.get_data()

            #####
            mu_zero = min(mu, 1e-12)
#            beta_ = self._beta
#            mask_ = tv.get_mask()
#            shape_ = tv.get_shape()

            alpha_l1_ = self._l1.alpha(self._beta, mu=mu_zero)
            alpha_tv_ = alpha  # self._tv.alpha(self._beta, mu=mu)
            Aa_l1_ = self._l1.grad(self._beta, alpha=alpha_l1_, mu=mu_zero)
            Aa_tv_ = self._tv.grad(self._beta, alpha=alpha_tv_, mu=mu)
            Aa_ = Aa_l1_ + Aa_tv_
            if not hasattr(self, '_XtinvXXtlI'):
#                XtX_ = np.dot(X.T, X)
#                self._invXtXlI = np.linalg.inv(XtX_ \
#                                                + rr.l * np.eye(*XtX_.shape))
                invXXtlI = np.linalg.inv(np.dot(X, X.T) \
                                            + rr.l * np.eye(X.shape[0]))
                self._XtinvXXtlI = np.dot(X.T, invXXtlI)
                self._Xty = np.dot(y.T, X).T

            wk_ = (self._Xty - Aa_) / rr.l
            beta = wk_ - np.dot(self._XtinvXXtlI, np.dot(X, wk_))

#            beta_ = np.dot(self._invXtXlI, np.dot(X.T, y) - Aa_)
#            beta = beta_

            return rr.f(beta) \
                    + self._l1.phi(beta, alpha_l1_) \
                    + self._tv.phi(beta, alpha_tv_)
                    # TODO: NOT WORKING!! _tv.phi(beta, alpha_tv_) is negative!
            #####

#            tv = self.get_g().b
#            Aa = tv.grad() <-- WARNING!
#            tv = loss_functions.LinearLossFunction(Aa)
#            dual_model = NesterovProximalGradientMethod()
#            dual_model.set_start_vector(self.get_start_vector())
#            dual_model.set_max_iter(self.get_max_iter())
#            dual_model.set_g(loss_functions.CombinedNesterovLossFunction(rr,
#                                                                         tv))
#            dual_model.set_h(self.get_h())
#
#            dual_model.fit(X, y, early_stopping=False)
#            beta = dual_model._beta

#            print "diff:", np.sum((beta_ - beta) ** 2.0)

        return self.get_g().phi(beta, alpha) + self.get_h().f(beta)

    def beta(self, alpha=None, mu=None):
        """Computes the beta that minimises the dual function value for the
        current computed or given alpha.
        """
        raise ValueError("Do not call this function!")
#        return self._rr_l1_tv.beta(alpha=alpha, mu=mu)

    def alpha(self, beta=None, mu=None):
        """Computes the alpha that maximises the smoothed loss function for the
        current computed beta.
        """
        raise ValueError("Do not call this function!")
#        return self._rr_l1_tv.alpha(beta=beta, mu=mu)

    def set_data(self, X, y):

        super(RidgeRegressionL1TV, self).set_data(X, y)

#        # We will need to recompute this matrix
#        if hasattr(self, "_XtinvXXtlI"):
#            del self._XtinvXXtlI


class LogisticRegression(NesterovProximalGradientMethod):

    def __init__(self, **kwargs):

        super(LogisticRegression, self).__init__(**kwargs)

        self.set_g(loss_functions.LogisticRegressionError())


class LogisticRegressionGL(NesterovProximalGradientMethod):
    """Logistic regression with group lasso constraint.

    Optimises the function

        f(b) = LR(b) + gamma.GL(b),

    where LR(.) is the logistic regression error and GL(.) is the group lasso
    constraint.

    Parameters
    ----------
    gamma : The GL regularisation parameter.

    num_variables : The number of variable being regularised.

    groups : A list of lists, with the outer list being the groups and the
            inner lists the variables in the groups. E.g. [[1,2],[2,3]]
            contains two groups ([1,2] and [2,3]) with variable 1 and 2 in the
            first group and variables 2 and 3 in the second group.

    mu : The Nesterov function regularisation parameter.

    weights : Weights put on the groups. Default is weight 1 for each group.
    """
    def __init__(self, gamma, num_variables, groups, mu=None, weights=None,
                 **kwargs):

        super(LogisticRegressionGL, self).__init__(**kwargs)

        lr = loss_functions.LogisticRegressionError()
        gl = loss_functions.GroupLassoOverlap(gamma, num_variables, groups,
                                              mu=mu, weights=weights)

        self.set_g(loss_functions.CombinedNesterovLossFunction(lr, gl))


class LogisticRegressionL1TV(NesterovProximalGradientMethod):

    def __init__(self, l, gamma, shape, mu=None, mask=None, **kwargs):

        super(LogisticRegressionL1TV, self).__init__(**kwargs)

        lr = loss_functions.LogisticRegressionError()
        tv = loss_functions.TotalVariation(gamma, shape=shape, mu=mu,
                                           mask=mask)

        self.set_g(loss_functions.CombinedNesterovLossFunction(lr, tv))
        self.set_h(loss_functions.L1(l))


class ExcessiveGapMethod(BaseModel):

    __metaclass__ = abc.ABCMeta

    def __init__(self, algorithm=None, **kwargs):

        if algorithm == None:
            algorithm = algorithms.ExcessiveGapRidgeRegression()

        super(ExcessiveGapMethod, self).__init__(num_comp=1,
                                                 algorithm=algorithm,
                                                 **kwargs)

    def fit(self, X, y, **kwargs):
        """Fit the model to the given data.

        Parameters
        ----------
        X : The independent variables.

        y : The dependent variable.

        Returns
        -------
        self: The model object.
        """
        self.get_g().set_data(X, y)

        self._beta = self.algorithm.run(X, y, **kwargs)

        return self

    def get_transform(self, index=0):

        return self._beta

    def predict(self, X, **kwargs):

        yhat = np.dot(X, self.get_transform(**kwargs))

        return yhat

    def get_g(self):

        return self.algorithm.g

    def set_g(self, g):

        self.algorithm.g = g

    def get_h(self):

        return self.algorithm.h

    def set_h(self, h):

        self.algorithm.h = h


class EGMRidgeRegression(ExcessiveGapMethod):
    """Linear regression with L2 regularisation. Uses the excessive gap method.

    Optimises the function

        f(b) = ||y - X.b||² + (k / 2.0).||b||²,

    where ||.||² is the squared L2 norm.

    Parameters
    ----------
    l : The L2 parameter.
    """
    def __init__(self, l, **kwargs):

        super(EGMRidgeRegression, self).__init__(**kwargs)

        self.set_g(loss_functions.RidgeRegression(l))
        self.set_h(loss_functions.SmoothL1(0.0, 1, 0.0))  # We're cheating! ;-)


class EGMLinearRegressionL1L2(ExcessiveGapMethod):
    """Linear Regression with L1 and L2 regularisation. Uses the excessive
    gap method.

    Optimises the function

        f(b) = ||y - X.b||² + l.||b||_1 + (k / 2.0).||b||²,

    where ||.||_1 is the L1 norm and ||.||² is the squared L2 norm.

    Parameters
    ----------
    l     : The L1 parameter.

    k     : The L2 parameter.

    p     : The numbers of variables.

    mask  : A 1-dimensional mask representing the 3D image mask. Must be a
            list of 1s and 0s.
    """
    def __init__(self, l, k, p, mask=None, **kwargs):

        super(EGMLinearRegressionL1L2, self).__init__(**kwargs)

        self.set_g(loss_functions.RidgeRegression(k))
        self.set_h(loss_functions.SmoothL1(l, p, mask=mask))


class EGMElasticNet(EGMLinearRegressionL1L2):
    """Linear regression with the elastic net constraint.

    Optimises the function

        f(b) = ||y - X.b||² + l.||b||_1 + ((1.0 - l) / 2.0).||b||²,

    where ||.||_1 is the L1 norm and ||.||² is the squared L2 norm.

    Parameters
    ----------
    l : The regularisation parameter. Must be in the interval [0,1].

    p     : The numbers of variables.

    mask  : A 1-dimensional mask representing the 3D image mask. Must be a
            list of 1s and 0s.
    """
    def __init__(self, l, p, mask=None, **kwargs):

        super(EGMElasticNet, self).__init__(l, 1.0 - l, p, mask=mask, **kwargs)


class EGMRidgeRegressionTV(ExcessiveGapMethod):
    """Ridge regression with total variation constraint.

    Optimises the function

        f(b) = ||y - X.b||² + l.||b||² + gamma.TV(b),

    where ||.||² is the squared L2 norm and TV(.) is the total variation
    constraint.

    Parameters
    ----------
    l     : The ridge parameter.

    gamma : The TV regularisation parameter.

    shape : The shape of the 3D image. Must be a 3-tuple. If the image is
            2D, let the Z dimension be 1, and if the "image" is 1D, let the
            Y and Z dimensions be 1. The tuple must be on the form
            (Z, Y, X).

    mask  : A 1-dimensional mask representing the 3D image mask. Must be a
           list of 1s and 0s.
    """
    def __init__(self, l, gamma, shape, mask=None, **kwargs):

        super(EGMRidgeRegressionTV, self).__init__(**kwargs)

        self.set_g(loss_functions.RidgeRegression(l))
        self.set_h(loss_functions.TotalVariation(gamma, shape=shape,
                                                 mask=mask))


class EGMRidgeRegressionL1TV(ExcessiveGapMethod):
    """Linear regression with L1, L2 and total variation constraints.

    Optimises the function

        f(b) = ||y - X.b||² + l.||b||_1 + (k / 2.0).||b||² + gamma.TV(b),

    where ||.||_1 is the L1 norm, ||.||² is the squared L2 norm and TV(.) is
    the total variation constraint.

    Parameters
    ----------
    l     : The ridge regularisation parameter. Must be in the interval [0,1].

    k     : The L2 parameter.

    gamma : The TV regularisation parameter.

    shape : The shape of the 3D image. Must be a 3-tuple. If the image is
            2D, let the Z dimension be 1, and if the "image" is 1D, let the
            Y and Z dimensions be 1. The tuple must be on the form
            (Z, Y, X).

    mask  : A 1-dimensional mask representing the 3D image mask. Must be a
           list of 1s and 0s.
    """
    def __init__(self, l, k, gamma, shape, mask=None, **kwargs):

        super(EGMRidgeRegressionL1TV, self).__init__(**kwargs)

        self.set_g(loss_functions.RidgeRegression(k))

        a = loss_functions.SmoothL1(l, np.prod(shape), mask=mask)
        b = loss_functions.TotalVariation(gamma, shape=shape, mask=mask,
                                          compress=False)

        self.set_h(loss_functions.CombinedNesterovLossFunction(a, b))


class EGMElasticNetTV(EGMRidgeRegressionL1TV):
    """Linear regression with elastic net and total variation constraints.

    Optimises the function

        f(b) = ||y - X.b||² + l.||b||_1 + ((1-l) / 2).||b||² + gamma.TV(b),

    where ||.||_1 is the L1 norm, ||.||² is the squared L2 norm and TV(.) is
    the total variation constraint.

    Parameters
    ----------
    l     : The Elastic Net regularisation parameter. Must be in the
            interval [0,1).

    gamma : The TV regularisation parameter.

    shape : The shape of the 3D image. Must be a 3-tuple. If the image is
            2D, let the Z dimension be 1, and if the "image" is 1D, let the
            Y and Z dimensions be 1. The tuple must be on the form
            (Z, Y, X).

    mask  : A 1-dimensional mask representing the 3D image mask. Must be a
           list of 1s and 0s.
    """
    def __init__(self, l, gamma, shape, mask=None, **kwargs):

        super(EGMElasticNetTV, self).__init__(l, 1.0 - l, gamma, shape,
                                              mask=mask, **kwargs)