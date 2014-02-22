# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 15:19:17 2013

@author:  Tommy Löfstedt, Edouard Duchesnay
@email:   lofstedt.tommy@gmail.com, edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
import abc
import numbers
import warnings

import numpy as np

import parsimony.utils.consts as consts
import parsimony.functions as functions
import parsimony.algorithms.explicit as explicit
import parsimony.start_vectors as start_vectors

__all__ = ['BaseEstimator', 'RegressionEstimator',

           'RidgeRegression_L1_TV', 'RidgeLogisticRegression_L1_TV',

           'RidgeRegression_SmoothedL1TV']


class BaseEstimator(object):
    """Base estimator for all kinds of estimation

    Parameters
    ----------
    algorithm : BaseAlgorithm. The algorithm that will be applied.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, algorithm):

        self.algorithm = algorithm

    def fit(self, X):
        """Fit the estimator to the data
        """
        raise NotImplementedError('Abstract method "fit" must be '
                                  'specialised!')

    def set_params(self, **kwargs):
        for k in kwargs:
            self.__setattr__(k, kwargs[k])

    @abc.abstractmethod
    def get_params(self):
        """Return a dictionary containing all the estimator's parameters
        """
        raise NotImplementedError('Abstract method "get_params" must be '
                                  'specialised!')

    @abc.abstractmethod
    def predict(self, X):
        """Return a predicted y corresponding to the X given and the beta
        previously determined
        """
        raise NotImplementedError('Abstract method "predict" must be '
                                  'specialised!')

    # TODO: Is this a good name?
    @abc.abstractmethod
    def score(self, X, y):
        raise NotImplementedError('Abstract method "score" must be '
                                  'specialised!')


class RegressionEstimator(BaseEstimator):
    """Base estimator for regression estimation

    Parameters
    ----------
    algorithm : ExplicitAlgorithm. The algorithm that will be applied.

    output : Boolean. Whether or not to return extra output information.

    start_vector : Numpy array. Generates the start vector that will be used.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, algorithm, output=False,
                 start_vector=start_vectors.RandomStartVector()):

        self.output = output
        self.start_vector = start_vector

        super(RegressionEstimator, self).__init__(algorithm=algorithm)

    @abc.abstractmethod
    def fit(self, X, y):
        """Fit the estimator to the data
        """
        raise NotImplementedError('Abstract method "fit" must be '
                                  'specialised!')

#        self.function.set_params(X=X, y=y)
#        # TODO: Should we use a seed here so that we get deterministic results?
#        beta = self.start_vector.get_vector((X.shape[1], 1))
#        if self.output:
#            self.beta, self.output = self.algorithm(X, y, self.function, beta)
#        else:
#            self.beta = self.algorithm(X, y, self.function, beta)

    def predict(self, X):
        """Return a predicted y corresponding to the X given and the beta
        previously determined
        """
        return np.dot(X, self.beta)

    def score(self, X, y):

        self.function.reset()
        self.function.set_params(X=X, y=y)
        return self.function.f(self.beta)


class LogisticRegressionEstimator(BaseEstimator):
    """Base estimator for logistic regression estimation

    Parameters
    ----------
    algorithm : ExplicitAlgorithm. The algorithm that will be applied.

    output : Boolean. Whether or not to return extra output information.

    start_vector : Numpy array. Generates the start vector that will be used.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, algorithm, output=False,
                 start_vector=start_vectors.RandomStartVector()):

        self.output = output
        self.start_vector = start_vector

        super(LogisticRegressionEstimator, self).__init__(algorithm=algorithm)

    @abc.abstractmethod
    def fit(self, X, y):
        """Fit the estimator to the data
        """
        raise NotImplementedError('Abstract method "fit" must be '
                                  'specialised!')

#        self.function.set_params(X=X, y=y)
#        # TODO: Should we use a seed here so that we get deterministic results?
#        beta = self.start_vector.get_vector((X.shape[1], 1))
#        if self.output:
#            self.beta, self.output = self.algorithm(X, y, self.function, beta)
#        else:
#            self.beta = self.algorithm(X, y, self.function, beta)

    def predict(self, X):
        """Return a predicted y corresponding to the X given and the beta
        previously determined
        """
        proba = self.predict_proba(X)
        y = np.ones((X.shape[0], 1))
        y[proba < .5] = 0
        return y

    def predict_probability(self, X):
        logit = np.dot(X, self.beta)
        proba = 1. / (1. + np.exp(-logit))
        return proba

    def score(self, X, y):

        self.function.reset()
        self.function.set_params(X=X, y=y)
        return self.function.f(self.beta)


class RidgeRegression_L1_TV(RegressionEstimator):
    """
    Parameters
    ----------
    l : Non-negative float. The L1 regularization parameter.

    k : Non-negative float. The L2 regularization parameter.

    g : Non-negative float. The total variation regularization parameter.

    A : Numpy or (usually) scipy.sparse array. The linear operator for the
            smoothed total variation Nesterov function.

    mu : Non-negative float. The regularisation constant for the smoothing.

    output : Boolean. Whether or not to return extra output information.

    algorithm : ExplicitAlgorithm. The algorithm that be applied. Should be
            one of:
                1. algorithms.StaticCONESTA()
                2. algorithms.DynamicCONESTA()
                3. algorithms.FISTA()

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.estimators as estimators
    >>> import parsimony.algorithms.explicit as explicit
    >>> import parsimony.functions.nesterov.tv as tv
    >>> shape = (1, 4, 4)
    >>> num_samples = 10
    >>> num_ft = shape[0] * shape[1] * shape[2]
    >>> np.random.seed(seed=1)
    >>> X = np.random.random((num_samples, num_ft))
    >>> y = np.random.randint(0, 2, (num_samples, 1))
    >>> k = 0.9  # ridge regression coefficient
    >>> l = 0.1  # l1 coefficient
    >>> g = 1.0  # tv coefficient
    >>> A, n_compacts = tv.A_from_shape(shape)
    >>> ridge_l1_tv = estimators.RidgeRegression_L1_TV(k, l, g, A,
    ...                     algorithm=explicit.StaticCONESTA(max_iter=1000))
    >>> res = ridge_l1_tv.fit(X, y)
    >>> error = np.sum(np.abs(np.dot(X, ridge_l1_tv.beta) - y))
    >>> print "error = ", error
    error =  4.70079220678
    >>> ridge_l1_tv = estimators.RidgeRegression_L1_TV(k, l, g, A,
    ...                     algorithm=explicit.DynamicCONESTA(max_iter=1000))
    >>> res = ridge_l1_tv.fit(X, y)
    >>> error = np.sum(np.abs(np.dot(X, ridge_l1_tv.beta) - y))
    >>> print "error = ", error
    error =  4.70096488794
    >>> ridge_l1_tv = estimators.RidgeRegression_L1_TV(k, l, g, A,
    ...                     algorithm=explicit.FISTA(max_iter=1000))
    >>> res = ridge_l1_tv.fit(X, y)
    >>> error = np.sum(np.abs(np.dot(X, ridge_l1_tv.beta) - y))
    >>> print "error = ", error
    error =  4.27776729699
    """
    def __init__(self, k, l, g, A, mu=None, output=False,
                 algorithm=explicit.StaticCONESTA()):
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
        """Return a dictionary containing all the estimator's parameters
        """
        return {"k": self.k, "l": self.l, "g": self.g,
                "A": self.A, "mu": self.mu}

    def fit(self, X, y, beta=None):
        """Fit the estimator to the data
        """
        self.function = functions.RR_L1_TV(X, y, self.k, self.l, self.g,
                                           A=self.A)
        self.algorithm.check_compatibility(self.function,
                                           self.algorithm.INTERFACES)

        # TODO: Should we use a seed here so that we get deterministic results?
        if beta is None:
            beta = self.start_vector.get_vector((X.shape[1], 1))

        if self.mu is None:
            self.mu = 0.9 * self.function.estimate_mu(beta)
        else:
            self.mu = float(self.mu)

        self.function.set_params(mu=self.mu)
        self.algorithm.set_params(output=self.output)

        if self.output:
            (self.beta, self.info) = self.algorithm.run(self.function, beta)
        else:
            self.beta = self.algorithm.run(self.function, beta)

        return self


class RidgeLogisticRegression_L1_TV(LogisticRegressionEstimator):
    """
    Minimize RidgeLogisticRegression (re-weighted log-likelihood
    aka cross-entropy) with L1 and TV penalties:
    Ridge (re-weighted) log-likelihood (cross-entropy):
    f(beta, X, y) = - Sum wi * (yi * log(pi) + (1 − yi) * log(1 − pi))
                    + k/2 * ||beta||^2_2
                    + l * ||beta||_1
                    + g * TV(beta)
    With pi = p(y=1|xi, beta) = 1 / (1 + exp(-xi' beta)) and wi: sample i
    weight

    Parameters
    ----------
    l : Non-negative float. The L1 regularization parameter.

    k : Non-negative float. The L2 regularization parameter.

    g : Non-negative float. The total variation regularization parameter.

    A : Numpy or (usually) scipy.sparse array. The linear operator for the
            smoothed total variation Nesterov function.

    weights: Numpy array with shape = (n_samples,). The samples weights.

    mu : Non-negative float. The regularisation constant for the smoothing.

    output : Boolean. Whether or not to return extra output information.

    algorithm : ExplicitAlgorithm. The algorithm that will be run. Should be
            one of:
                1. algorithms.StaticCONESTA()
                2. algorithms.DynamicCONESTA()
                3. algorithms.FISTA()

    Examples
    --------
    """
    def __init__(self, k, l, g, A, weigths=None, mu=None, output=False,
                 algorithm=explicit.StaticCONESTA()):
#                 algorithm=algorithms.DynamicCONESTA()):
#                 algorithm=algorithms.FISTA()):
        self.k = float(k)
        self.l = float(l)
        self.g = float(g)
        self.A = A
        self.weigths = weigths
        if isinstance(mu, numbers.Number):
            self.mu = float(mu)
        else:
            self.mu = None

        super(RidgeLogisticRegression_L1_TV,
              self).__init__(algorithm=algorithm, output=output)

    def get_params(self):
        """Return a dictionary containing all the estimator's parameters
        """
        return {"k": self.k, "l": self.l, "g": self.g,
                "A": self.A, "mu": self.mu, "weigths": self.weigths}

    def fit(self, X, y):
        """Fit the estimator to the data
        """
        self.function = functions.RLR_L1_TV(X, y, self.k, self.l, self.g,
                                           A=self.A, weights=self.weigths)
        self.algorithm.check_compatibility(self.function,
                                           self.algorithm.INTERFACES)

        # TODO: Should we use a seed here so that we get deterministic results?
        beta = self.start_vector.get_vector((X.shape[1], 1))

        if self.mu is None:
            self.mu = 0.9 * self.function.estimate_mu(beta)
        else:
            self.mu = float(self.mu)

        self.function.set_params(mu=self.mu)
        self.algorithm.set_params(output=self.output)

        if self.output:
            (self.beta, self.info) = self.algorithm.run(self.function, beta)
        else:
            self.beta = self.algorithm.run(self.function, beta)

        return self


class RidgeRegression_SmoothedL1TV(RegressionEstimator):
    """
    Parameters
    ----------
    l : Non-negative float. The L1 regularisation parameter.

    k : Non-negative float. The L2 regularisation parameter.

    g : Non-negative float. The total variation regularization parameter.

    Atv : Numpy array (usually sparse). The linear operator for the smoothed
            total variation Nesterov function.

    Al1 : Numpy array (usually sparse). The linear operator for the smoothed
            L1 Nesterov function.

    mu : Non-negative float. The regularisation constant for the smoothing.

    output : Boolean. Whether or not to return extra output information.

    algorithm : ExplicitAlgorithm. The algorithm that will be applied.

    Examples
    --------
    >>> import numpy as np
    >>> import scipy.sparse as sparse
    >>> import parsimony.estimators as estimators
    >>> import parsimony.algorithms.explicit as explicit
    >>> import parsimony.functions.nesterov.tv as tv
    >>> shape = (1, 4, 4)
    >>> num_samples = 10
    >>> num_ft = shape[0] * shape[1] * shape[2]
    >>> np.random.seed(seed=1)
    >>> X = np.random.random((num_samples, num_ft))
    >>> y = np.random.randint(0, 2, (num_samples, 1))
    >>> k = 0.05  # ridge regression coefficient
    >>> l = 0.05  # l1 coefficient
    >>> g = 0.05  # tv coefficient
    >>> Atv, n_compacts = tv.A_from_shape(shape)
    >>> Al1 = sparse.eye(num_ft, num_ft)
    >>> ridge_smoothed_l1_tv = estimators.RidgeRegression_SmoothedL1TV(k, l, g,
    ...                 Atv=Atv, Al1=Al1,
    ...                 algorithm=explicit.ExcessiveGapMethod(max_iter=1000))
    >>> res = ridge_smoothed_l1_tv.fit(X, y)
    >>> error = np.sum(np.abs(np.dot(X, ridge_smoothed_l1_tv.beta) - y))
    >>> print "error = ", error
    error =  1.69470206808
    """
    def __init__(self, k, l, g, Atv, Al1, mu=None, output=False,
                 algorithm=explicit.ExcessiveGapMethod()):

        self.k = float(k)
        self.l = float(l)
        self.g = float(g)
        if self.k < consts.TOLERANCE:
            warnings.warn("The ridge parameter should be non-zero.")
        self.Atv = Atv
        self.Al1 = Al1
        if isinstance(mu, numbers.Number):
            self.mu = float(mu)
        else:
            self.mu = None

        super(RidgeRegression_SmoothedL1TV, self).__init__(algorithm=algorithm,
                                                           output=output)

    def get_params(self):
        """Return a dictionary containing all the estimator's parameters
        """
        return {"k": self.k, "l": self.l, "g": self.g,
                "A": self.A, "mu": self.mu}

    def fit(self, X, y):
        """Fit the estimator to the data
        """
        self.function = functions.RR_SmoothedL1TV(X, y,
                                                  self.k, self.l, self.g,
                                                  Atv=self.Atv, Al1=self.Al1)

        self.algorithm.check_compatibility(self.function,
                                           self.algorithm.INTERFACES)

        self.algorithm.set_params(output=self.output)
        if self.output:
            (self.beta, self.info) = self.algorithm.run(self.function)
        else:
            self.beta = self.algorithm.run(self.function)

        return self


if __name__ == "__main__":

    import doctest
    doctest.testmod()