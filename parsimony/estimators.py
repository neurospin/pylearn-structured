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

    def __init__(self, **opt):
        self.opt = opt

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


class BaseEstimator(object):
    """
    Arguments
    ---------
    k float
    l float
    g float
    A Sparse matrix
    algorithm: function
        conesta_static, conesta_dynamic, fista, ExcessiveGapMethod
    """
    def __init__(self, alg):

        self.alg = alg

    def fit(self, X, y=None):

        self.func.set_params(X=X, y=y)
        self.beta = self.alg(self.func, beta)


class OLS_L1(BaseEstimator):

    def __init__(self, l, alg=algorithms.fista):

        self.func = functions.OLS_L1(l)
        beta = ...
        super(OLS_L1, self).__init__(alg=alg)



e_fista = OLS_L1(0.5)
e_fista.fit(X, y)

e_conesta = OLS_L1(0.5, alg=algorithms.conesta(tau=0.9))
e_conesta.fit(X, y)






#        function = self.func_class(options)#self.k, self.l, self.g, self.shape)
#        output = self.algorithm.fit(X, y, function)
        function = self.func_class(self.k, self.l, self.g, self._A)
#        function.set_params(X=X, y=y)

        # TODO: Use start_vectors for this!
        betastart = np.random.rand(X.shape[1], 1)

        if self.algorithm == algorithms.conesta_static:

            mu_zero = utils.TOLERANCE
            eps = utils.TOLERANCE
            conts = 25
            max_iter = int(utils.MAX_ITER / conts)

            output = algorithms.conesta_static(X, y, function, betastart,
                                    mu_start=None,
                                    mumin=mu_zero,
                                    tau=0.5,
                                    eps=eps, conts=conts, max_iter=max_iter)

            beta, f, t, mu, Gval, b, g = output

        elif self.algorithm == algorithms.conesta_dynamic:

            mu_zero = utils.TOLERANCE
            eps = utils.TOLERANCE
            conts = 25
            max_iter = int(utils.MAX_ITER / conts)

            output = algorithms.conesta_dynamic(X, y, function, betastart,
                                    mu_start=None,
                                    mumin=mu_zero,
                                    tau=0.5,
                                    eps=eps, conts=conts, max_iter=max_iter)

            beta, f, t, mu, Gval, b, g = output

        elif self.algorithm == algorithms.fista:

            eps = utils.TOLERANCE
            max_iter = utils.MAX_ITER

            output = algorithms.fista(X, y, function, betastart,
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
        

#class EstimatorConestaOLS2L1TV(LinearRegressionL1L2TV):
#
#    def __init__(self, k, l, g, shape):
##        options = {...}
#        super(LinearRegressionL1L2TV, self).__init__(k, l, g, shape, 
#                                               func_class=functions.OLSL2_L1_TV,
#                                               algorithm=algorithms.CONESTA)
