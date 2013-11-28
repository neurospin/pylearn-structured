# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 15:19:17 2013

@author:  Tommy Löfstedt, Edouard Duchesnay
@email:   tommy.loefstedt@cea.fr, edouard.duchesnay@cea.fr
@license: TBD.
"""
import abc
import numbers

import numpy as np

import parsimony.functions as functions
import parsimony.algorithms as algorithms
import parsimony.start_vectors as start_vectors

__all__ = ['BaseEstimator', 'RegressionEstimator',

           'RidgeRegression_L1_TV',

           'RidgeRegression_SmoothedL1TV']


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
    """

    Parameters
    ----------
    l : The L1 regularisation parameter.

    k : The L2 regularisation parameter.

    g : The total variation regularisation parameter.

    A : Matrix allocation for regression

    # TODO : what is mu?
    mu : ???

    output : boolean, need output information

    algorithm : which algorithm willbe applied :
        1. algorithms.StaticCONESTA()
        2. algorithms.DynamicCONESTA()
        3. algorithms.FISTA()

    Example
    -------
    >>> import numpy as np
    >>> import parsimony.estimators as estimators
    >>> import parsimony.algorithms as algorithms
    >>> import parsimony.tv
    >>> shape = (4, 4, 1)
    >>> num_samples = 10
    >>> num_ft = shape[0] * shape[1] * shape[2]
    >>> np.random.seed(seed=1)
    >>> X = np.random.random((num_samples, num_ft))
    >>> y = np.random.randint(0, 2, (num_samples, 1))
    >>> k = 0.9  # ridge regression coefficient
    >>> l = 0.1  # l1 coefficient
    >>> g = 1.0  # tv coefficient
    >>> A, n_compacts = parsimony.tv.A_from_shape(shape)
    >>> ridge_l1_tv = estimators.RidgeRegression_L1_TV(k, l, g, A,
    ...                     algorithm=algorithms.StaticCONESTA(max_iter=1000))
    >>> res = ridge_l1_tv.fit(X, y)
    >>> error = np.sum(np.abs(np.dot(X, ridge_l1_tv.beta) - y))
    >>> print "error = ", error
    error =  4.70079220678
    >>> ridge_l1_tv = estimators.RidgeRegression_L1_TV(k, l, g, A,
    ...                     algorithm=algorithms.DynamicCONESTA(max_iter=1000))
    >>> res = ridge_l1_tv.fit(X, y)
    >>> error = np.sum(np.abs(np.dot(X, ridge_l1_tv.beta) - y))
    >>> print "error = ", error
    error =  4.70096488794
    >>> ridge_l1_tv = estimators.RidgeRegression_L1_TV(k, l, g, A,
    ...                     algorithm=algorithms.FISTA(max_iter=1000))
    >>> res = ridge_l1_tv.fit(X, y)
    >>> error = np.sum(np.abs(np.dot(X, ridge_l1_tv.beta) - y))
    >>> print "error = ", error
    error =  4.27776729699
    """
    def __init__(self, k, l, g, A, mu=None, output=False,
                 algorithm=algorithms.StaticCONESTA()):
#                 algorithm=algorithms.DynamicCONESTA()):
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
        self.algorithm.check_compatability(self.function,
                                           self.algorithm.INTERFACES)

        # TODO: Should we use a seed here so that we get deterministic results?
        beta = self.start_vector.get_vector((X.shape[1], 1))

        if self.mu == None:
            self.mu = 0.9 * self.function.estimate_mu(beta)
        else:
            self.mu = float(self.mu)

        self.function.set_params(mu=self.mu)
        self.algorithm.set_params(output=self.output)

        if self.output:
            (self.beta, self.info) = self.algorithm(self.function, beta)
        else:
            self.beta = self.algorithm(self.function, beta)

        return self


class RidgeRegression_SmoothedL1TV(RegressionEstimator):
    """
    Parameters
    ----------
    
    Example
    -------
    >>> import numpy as np
    >>> import scipy.sparse as sparse
    >>> import parsimony.estimators as estimators
    >>> import parsimony.algorithms as algorithms
    >>> import parsimony.tv
    >>> shape = (4, 4, 1)
    >>> num_samples = 10
    >>> num_ft = shape[0] * shape[1] * shape[2]
    >>> np.random.seed(seed=1)
    >>> X = np.random.random((num_samples, num_ft))
    >>> y = np.random.randint(0, 2, (num_samples, 1))
    >>> k = 0.05  # ridge regression coefficient
    >>> l = 0.05  # l1 coefficient
    >>> g = 0.05  # tv coefficient
    >>> Atv, n_compacts = parsimony.tv.A_from_shape(shape)
    >>> Al1 = sparse.eye(num_ft, num_ft)
    >>> ridge_smoothed_l1_tv = estimators.RidgeRegression_SmoothedL1TV(k, l, g,
    ...                 Atv=Atv, Al1=Al1,
    ...                 algorithm=algorithms.ExcessiveGapMethod(max_iter=1000))
    >>> res = ridge_smoothed_l1_tv.fit(X, y)
    >>> error = np.sum(np.abs(np.dot(X, ridge_smoothed_l1_tv.beta) - y))
    >>> print "error = ", error
    error =  1.69470206808
    """
    def __init__(self, k, l, g, Atv, Al1, mu=None, output=False,
                 algorithm=algorithms.ExcessiveGapMethod()):

        self.k = float(k)
        self.l = float(l)
        self.g = float(g)
        self.Atv = Atv
        self.Al1 = Al1
        if isinstance(mu, numbers.Number):
            self.mu = float(mu)
        else:
            self.mu = None

        super(RidgeRegression_SmoothedL1TV, self).__init__(algorithm=algorithm,
                                                           output=output)

    def get_params(self):

        return {"k": self.k, "l": self.l, "g": self.g,
                "A": self.A, "mu": self.mu}

    def fit(self, X, y):

        self.function = functions.RR_SmoothedL1TV(X, y,
                                                  self.k, self.l, self.g,
                                                  Atv=self.Atv, Al1=self.Al1)

        self.algorithm.check_compatability(self.function,
                                           self.algorithm.INTERFACES)

        self.algorithm.set_params(output=self.output)
        if self.output:
            (self.beta, self.info) = self.algorithm(self.function)
        else:
            self.beta = self.algorithm(self.function)

        return self


if __name__ == "__main__":
    import doctest
    doctest.testmod()