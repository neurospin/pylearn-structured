# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 15:19:17 2013

@author:  Tommy Löfstedt, Edouard Duchesnay
@email:   tommy.loefstedt@cea.fr, edouard.duchesnay@cea.fr
@license: TBD.
"""
import abc
import numpy as np
import parsimony.functions as functions
import parsimony.algorithms as algorithms
import parsimony.utils as utils

__all__ = ['LinearRegressionL1L2TV']


class BaseEstimator(object):

    __metaclass__ = abc.ABCMeta

    def set_params(self, **kwargs):

        for k, v in kwargs:
            self.__setattr__(k, v)

    @abc.abstractmethod
    def get_params():
        raise NotImplementedError('Abstract method "get_params" must be ' \
                                  'specialised!')

    @abc.abstractmethod
    def fit(X):
        raise NotImplementedError('Abstract method "fit" must be ' \
                                  'specialised!')

    @abc.abstractmethod
    def predict(X):
        raise NotImplementedError('Abstract method "predict" must be ' \
                                  'specialised!')


class LinearRegressionL1L2TV(BaseEstimator):
    """
    Arguments
    ---------
    k float
    l float
    g float
    A Sparse matrix
    algorithm: string
        "conesta_static", "conesta_dynamic", "fista", "excessive_gap"
    """
    def __init__(self, k, l, g, A, algorithm=None, func_class=None):

        self.k = k
        self.l = l
        self.g = g
        self._A = A

        if algorithm == None:
            algorithm = "conesta_static"#algorithms.CONESTA
#            algorithm = algorithms.ExcessiveGapMethod
        if func_class == None:
            func_class = functions.OLSL2_L1_TV
#            func_class = functions.OLSL2_SmoothedL1TV

        self.func_class = func_class
        self.algorithm = algorithm

    def get_params(self):

        return {"k": self.k, "l": self.l, "g": self.g}

    def fit(self, X, y):

        function = self.func_class(self.k, self.l, self.g, self._A)
#        function.set_params(X=X, y=y)

        # TODO: Use start_vectors for this!
        betastart = np.random.rand(X.shape[1], 1)

        if self.algorithm == "conesta_static":

            mu_zero = utils.TOLERANCE
            eps = utils.TOLERANCE
            conts = 25
            max_iter = int(utils.MAX_ITER / conts)

            output = algorithms.CONESTA(X, y, function, betastart,
                                    mu_start=None,
                                    mumin=mu_zero,
                                    tau=0.5,
                                    dynamic=False,
                                    eps=eps, conts=conts, max_iter=max_iter)

            beta, f, t, mu, Gval, b, g = output

        elif self.algorithm == "conesta_dynamic":

            mu_zero = utils.TOLERANCE
            eps = utils.TOLERANCE
            conts = 25
            max_iter = int(utils.MAX_ITER / conts)

            output = algorithms.CONESTA(X, y, function, betastart,
                                    mu_start=None,
                                    mumin=mu_zero,
                                    tau=0.5,
                                    dynamic=True,
                                    eps=eps, conts=conts, max_iter=max_iter)

            beta, f, t, mu, Gval, b, g = output

        elif self.algorithm == "fista":

            eps = utils.TOLERANCE
            max_iter = utils.MAX_ITER

            output = algorithms.FISTA(X, y, function, betastart,
                                      step=None, mu=None,
                                      eps=eps, max_iter=max_iter)

            beta, f, t, b, g = output

        elif self.algorithm == "excessive_gap":

            eps = utils.TOLERANCE
            max_iter = utils.MAX_ITER

            output = algorithms.ExcessiveGapMethod(X, y, function,
                                                   eps=eps, max_iter=max_iter)
            beta, f, t, mu, ulim, beta0, b = output

        else:
            raise NotImplementedError("Unknown algorithm!")

        self.beta = beta
        self.f = f
        self.t = t
#        self.mu = mu
#        self.G = G

        return self

    def predict(self, X):

        return np.dot(X, self.beta)