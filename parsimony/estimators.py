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
    def get_params(**kwargs):
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

    def __init__(self, k, l, g, shape, func_class=None, algorithm=None):

        self.k = k
        self.l = l
        self.g = g
        self.shape = shape

        if func_class == None:
            func_class = functions.OLSL2_L1_TV
#            func_class = functions.OLSL2_SmoothedL1TV
        if algorithm == None:
            algorithm = algorithms.CONESTA
#            algorithm = algorithms.ExcessiveGapMethod

        self.func_class = func_class
        self.algorithm = algorithm

    def get_params(self, **kwargs):

        return {"k": self.k, "l": self.l, "g": self.g}

    def fit(self, X, y):

        function = self.func_class(self.k, self.l, self.g, self.shape)
#        function.set_params(X=X, y=y)

        # TODO: Use start_vectors for this!
        betastart = np.random.rand(X.shape[1], 1)

        if self.algorithm == algorithms.CONESTA:

            mu_zero = utils.TOLERANCE
            eps = utils.TOLERANCE
            conts = 25
            max_iter = int(utils.MAX_ITER / conts)

            output = self.algorithm(X, y, function, betastart,
                                    mu_start=None,
                                    mumin=mu_zero,
                                    tau=0.5,
                                    dynamic=True,
                                    eps=eps, conts=conts, max_iter=max_iter)

            beta, f, t, mu, Gval, b, g = output

        elif self.algorithm == algorithms.FISTA:

            eps = utils.TOLERANCE
            max_iter = utils.MAX_ITER

            output = algorithms.FISTA(X, y, function, betastart,
                                      step=None, mu=None,
                                      eps=eps, max_iter=max_iter)

            beta, f, t, b, g = output

        elif self.algorithm == algorithms.ExcessiveGapMethod:

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