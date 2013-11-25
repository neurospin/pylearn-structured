# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 15:19:17 2013

@author:  Tommy Löfstedt, Edouard Duchesnay
@email:   tommy.loefstedt@cea.fr, edouard.duchesnay@cea.fr
@license: TBD.
"""
import abc
import numpy as np
import numbers

import parsimony.functions as functions
import parsimony.algorithms as algorithms
import parsimony.start_vectors as start_vectors

__all__ = ['BaseEstimator', 'RegressionEstimator',

           'RidgeRegression_L1_TV']


class BaseEstimator(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, algorithm):

        self.algorithm = algorithm

    def fit(self, X):
        raise NotImplementedError('Abstract method "fit" must be ' \
                                  'specialised!')

    def set_params(self, **kwargs):

        for k in kwargs:
            self.__setattr__(k, kwargs[k])

    @abc.abstractmethod
    def get_params(self):
        raise NotImplementedError('Abstract method "get_params" must be ' \
                                  'specialised!')

    @abc.abstractmethod
    def predict(self, X):
        raise NotImplementedError('Abstract method "predict" must be ' \
                                  'specialised!')

    # TODO: Is this a good name?
    @abc.abstractmethod
    def score(self, X, y):
        raise NotImplementedError('Abstract method "score" must be ' \
                                  'specialised!')


class RegressionEstimator(BaseEstimator):

    __metaclass__ = abc.ABCMeta

    def __init__(self, algorithm, output=False,
                       start_vector=start_vectors.RandomStartVector()):

        self.output = output
        self.start_vector = start_vector

        super(RegressionEstimator, self).__init__(algorithm=algorithm)

    @abc.abstractmethod
    def fit(self, X, y):
        raise NotImplementedError('Abstract method "fit" must be ' \
                                  'specialised!')

#        self.function.set_params(X=X, y=y)
#        # TODO: Should we use a seed here so that we get deterministic results?
#        beta = self.start_vector.get_vector((X.shape[1], 1))
#        if self.output:
#            self.beta, self.output = self.algorithm(X, y, self.function, beta)
#        else:
#            self.beta = self.algorithm(X, y, self.function, beta)

    def predict(self, X):
        return np.dot(X, self.beta)

    def score(self, X, y):

        self.function.reset()
        self.function.set_params(X=X, y=y)
        return self.function.f(self.beta)


class RidgeRegression_L1_TV(RegressionEstimator):

    def __init__(self, k, l, g, A, mu=None, output=False,
                 algorithm=algorithms.CONESTA(dynamic=True)):
#                 algorithm=algorithms.FISTA()):

        self.k = float(k)
        self.l = float(l)
        self.g = float(g)
        self.A = A
        if isinstance(mu, numbers.Number):
            self.mu = float(mu)
        else:
            self.mu = None

        super(RidgeRegression_L1_TV, self).__init__(algorithm=algorithm,
                                                    output=output)

    def get_params(self):

        return {"k": self.k, "l": self.l, "g": self.g,
                "A": self.A, "mu": self.mu}

    def fit(self, X, y):

        self.function = functions.RR_L1_TV(X, y, self.k, self.l, self.g,
                                           A=self.A)

        # TODO: Should we use a seed here so that we get deterministic results?
        beta = self.start_vector.get_vector((X.shape[1], 1))

        if self.mu == None:
            self.mu = 0.9 * self.function.mu(beta)
        else:
            self.mu = float(self.mu)

        self.function.set_params(mu=self.mu)
        self.algorithm.set_params(output=self.output)

        if self.output:
            self.beta, self.info = self.algorithm(self.function, beta)
        else:
            self.beta = self.algorithm(self.function, beta)

        return self