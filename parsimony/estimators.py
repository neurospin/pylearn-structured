# -*- coding: utf-8 -*-
"""Estimators encapsulates a loss function with penalties and a corresponding
algorithm.

Created on Sat Nov  2 15:19:17 2013

@author:  Tommy Löfstedt, Edouard Duchesnay
@email:   lofstedt.tommy@gmail.com, edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
import abc
import warnings

import numpy as np

import parsimony.utils.consts as consts
import parsimony.functions as functions
import parsimony.functions.losses as losses
import parsimony.functions.penalties as penalties
import parsimony.algorithms.explicit as explicit
import parsimony.utils.start_vectors as start_vectors
from parsimony.utils import check_arrays
from parsimony.utils import class_weight_to_sample_weight, check_labels

__all__ = ["BaseEstimator", "RegressionEstimator",

           "LinearRegressionL1L2TV",
           "LinearRegressionL1L2GL",

           "LogisticRegressionL1L2TV",
           "LogisticRegressionL1L2GL",

           "LinearRegressionL2SmoothedL1TV"]


class BaseEstimator(object):
    """Base class for estimators.

    Parameters
    ----------
    algorithm : BaseAlgorithm. The algorithm that will be used.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, algorithm):

        self.algorithm = algorithm

    def fit(self, X):
        """Fit the estimator to the data.
        """
        raise NotImplementedError('Abstract method "fit" must be '
                                  'specialised!')

    def set_params(self, **kwargs):
        for k in kwargs:
            self.__setattr__(k, kwargs[k])

    @abc.abstractmethod
    def get_params(self):
        """Return a dictionary containing the estimator's own parameters.
        """
        raise NotImplementedError('Abstract method "get_params" must be '
                                  'specialised!')

    @abc.abstractmethod
    def predict(self, X):
        """Perform prediction using the fitted parameters.
        """
        raise NotImplementedError('Abstract method "predict" must be '
                                  'specialised!')

    # TODO: Is this a good name?
    @abc.abstractmethod
    def score(self, X, y):
        raise NotImplementedError('Abstract method "score" must be '
                                  'specialised!')


class RegressionEstimator(BaseEstimator):
    """Base estimator for regression estimation.

    Parameters
    ----------
    algorithm : ExplicitAlgorithm. The algorithm that will be applied.

    start_vector : BaseStartVector. Generates the start vector that will be
            used.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, algorithm,
                 start_vector=start_vectors.RandomStartVector()):

        super(RegressionEstimator, self).__init__(algorithm=algorithm)

        self.start_vector = start_vector

    @abc.abstractmethod
    def fit(self, X, y):
        """Fit the estimator to the data.
        """
        raise NotImplementedError('Abstract method "fit" must be '
                                  'specialised!')

    def predict(self, X):
        """Perform prediction using the fitted parameters.
        """
        return np.dot(check_arrays(X), self.beta)

    @abc.abstractmethod
    def score(self, X, y):
        """Return the score of the estimator.

        The score is a measure of "goodness" of the fit to the data.
        """
#        self.function.reset()
#        self.function.set_params(X=X, y=y)
#        return self.function.f(self.beta)
        raise NotImplementedError('Abstract method "score" must be '
                                  'specialised!')


class LogisticRegressionEstimator(BaseEstimator):
    """Base estimator for logistic regression estimation

    Parameters
    ----------
    algorithm : ExplicitAlgorithm. The algorithm that will be applied.

    start_vector : Numpy array. Generates the start vector that will be used.

    class_weight : {dict, "auto"}, optional. Set the parameter weight of
            sample belonging to class i to class_weight[i]. If not given, all
            classes are supposed to have weight one. The "auto" mode uses the
            values of y to automatically adjust weights inversely proportional
            to class frequencies.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, algorithm,
                 start_vector=start_vectors.RandomStartVector(),
                 class_weight=None):

        super(LogisticRegressionEstimator, self).__init__(algorithm=algorithm)

        self.start_vector = start_vector
        self.class_weight = class_weight

    @abc.abstractmethod
    def fit(self, X, y):
        """Fit the model to the data.
        """
        raise NotImplementedError('Abstract method "fit" must be '
                                  'specialised!')

    def predict(self, X):
        """Return a predicted y corresponding to the X given and the beta
        previously determined.
        """
        X = check_arrays(X)
        prob = self.predict_probability(X)
        y = np.ones((X.shape[0], 1))
        y[prob < 0.5] = 0.0

        return y

    def predict_probability(self, X):
        X = check_arrays(X)
        logit = np.dot(X, self.beta)
        prob = 1.0 / (1.0 + np.exp(-logit))

        return prob

    def score(self, X, y):
        """Rate of correct classification.
        """
        yhat = self.predict(X)
        rate = np.mean(y == yhat)

        return rate


class LinearRegression(RegressionEstimator):
    """Linear regression:

        f(beta, X, y) = (1 / 2 * n) * ||Xbeta - y||²_2,

    where ||.||²_2 is the squared L2-norm.

    Parameters
    ----------
    algorithm : ExplicitAlgorithm. The algorithm that should be used.
            Should be one of:
                1. GradientDescent(...)

            Default is GradientDescent(...).

    algorithm_params : A dict. The dictionary algorithm_params contains
            parameters that should be set in the algorithm. Passing
            algorithm=GradientDescent(**params) is equivalent to passing
            algorithm=GradientDescent() and algorithm_params=params. Default
            is an empty dictionary.

    start_vector : BaseStartVector. Generates the start vector that will be
            used.

    mean : Boolean. Whether to compute the squared loss or the mean squared
            loss. Default is True, the mean squared loss.

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.estimators as estimators
    >>> import parsimony.algorithms.explicit as explicit
    >>>
    >>> np.random.seed(42)
    >>>
    >>> n = 10
    >>> p = 16
    >>> X = np.random.rand(n, p)
    >>> y = np.random.rand(n, 1)
    >>> lr = estimators.LinearRegression(algorithm=explicit.GradientDescent(),
    ...                                  algorithm_params=dict(max_iter=1000),
    ...                                  mean=False)
    >>> error = lr.fit(X, y).score(X, y)
    >>> print "error = ", error
    error =  0.0116466703591
    """
    def __init__(self, algorithm=None, algorithm_params=dict(),
                 start_vector=start_vectors.RandomStartVector(),
                 mean=True):

        if algorithm is None:
            algorithm = explicit.GradientDescent(**algorithm_params)
        else:
            algorithm.set_params(**algorithm_params)

        super(LinearRegression, self).__init__(algorithm=algorithm,
                                               start_vector=start_vector)

        self.mean = bool(mean)

    def get_params(self):
        """Return a dictionary containing the estimator's parameters
        """
        return {"mean": self.mean}

    def fit(self, X, y, beta=None):
        """Fit the estimator to the data.
        """
        X, y = check_arrays(X, y)

        function = losses.LinearRegression(X, y, mean=self.mean)

        self.algorithm.check_compatibility(function,
                                           self.algorithm.INTERFACES)

        # TODO: Should we use a seed here so that we get deterministic results?
        if beta is None:
            beta = self.start_vector.get_vector(X.shape[1])

        self.beta = self.algorithm.run(function, beta)

        return self

    def score(self, X, y):
        """Returns the (mean) squared error of the estimator.
        """
        X, y = check_arrays(X, y)
        n, p = X.shape
        y_hat = np.dot(X, self.beta)
        err = np.sum((y_hat - y) ** 2.0)
        if self.mean:
            err /= float(n)

        return err


class Lasso(RegressionEstimator):
    """Linear regression with an L1 penalty:

        f(beta, X, y) = (1 / 2 * n) * ||Xbeta - y||²_2 + l1 * ||beta||_1,

    where ||.||²_2 is the squared L2-norm and ||.||_1 is the L1-norm.

    Parameters
    ----------
    l1 : Non-negative float. The L1 regularization parameter.

    algorithm : ExplicitAlgorithm. The algorithm that should be applied.
            Should be one of:
                1. FISTA(...)
                2. ISTA(...)

            Default is FISTA(...).

    algorithm_params : A dict. The dictionary algorithm_params contains
            parameters that should be set in the algorithm. Passing
            algorithm=FISTA(**params) is equivalent to passing
            algorithm=FISTA() and algorithm_params=params. Default is
            an empty dictionary.

    start_vector : BaseStartVector. Generates the start vector that will be
            used.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first
            index to be penalised. Default is 0, all columns are included.

    mean : Boolean. Whether to compute the squared loss or the mean squared
            loss. Default is True, the mean squared loss.

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.estimators as estimators
    >>> import parsimony.algorithms.explicit as explicit
    >>> import parsimony.functions.nesterov.tv as total_variation
    >>> n = 10
    >>> p = 16
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(n, p)
    >>> y = np.random.rand(n, 1)
    >>> l1 = 0.1  # L1 coefficient
    >>> lasso = estimators.Lasso(l1,
    ...                          algorithm=explicit.FISTA(),
    ...                          algorithm_params=dict(max_iter=1000),
    ...                          mean=False)
    >>> error = lasso.fit(X, y).score(X, y)
    >>> print "error = ", error
    error =  0.395494642796
    """
    def __init__(self, l1,
                 algorithm=None, algorithm_params=dict(),
                 start_vector=start_vectors.RandomStartVector(),
                 penalty_start=0,
                 mean=True):

        if algorithm is None:
            algorithm = explicit.FISTA(**algorithm_params)
        else:
            algorithm.set_params(**algorithm_params)

        super(Lasso, self).__init__(algorithm=algorithm,
                                    start_vector=start_vector)

        self.l1 = float(l1)

        self.penalty_start = int(penalty_start)
        self.mean = bool(mean)

    def get_params(self):
        """Return a dictionary containing the estimator's parameters
        """
        return {"l1": self.l1,
                "penalty_start": self.penalty_start, "mean": self.mean}

    def fit(self, X, y, beta=None):
        """Fit the estimator to the data.
        """
        X, y = check_arrays(X, y)

        function = functions.CombinedFunction()
        function.add_function(losses.LinearRegression(X, y, mean=self.mean))
        function.add_prox(penalties.L1(l=self.l1,
                                       penalty_start=self.penalty_start))

        self.algorithm.check_compatibility(function,
                                           self.algorithm.INTERFACES)

        # TODO: Should we use a seed here so that we get deterministic results?
        if beta is None:
            beta = self.start_vector.get_vector(X.shape[1])

        self.beta = self.algorithm.run(function, beta)

        return self

    def score(self, X, y):
        """Returns the (mean) squared error of the estimator.
        """
        X, y = check_arrays(X, y)
        n, p = X.shape
        y_hat = np.dot(X, self.beta)
        err = np.sum((y_hat - y) ** 2.0)
        if self.mean:
            err /= float(n)

        return err


class ElasticNet(RegressionEstimator):
    """Linear regression with L1 and L2 penalties:

        f(beta, X, y) = (1 / 2 * n) * ||Xbeta - y||²_2
                      + l * ||beta||_1
                      + ((1.0 - l) / 2) * ||beta||²_2,

    where ||.||²_2 is the squared L2-norm and ||.||_1 is the L1-norm.

    Parameters
    ----------
    l : Non-negative float. The regularization parameter.

    algorithm : ExplicitAlgorithm. The algorithm that should be applied.
            Should be one of:
                1. FISTA(...)
                2. ISTA(...)

            Default is FISTA(...).

    algorithm_params : A dict. The dictionary algorithm_params contains
            parameters that should be set in the algorithm. Passing
            algorithm=FISTA(**params) is equivalent to passing
            algorithm=FISTA() and algorithm_params=params. Default is
            an empty dictionary.

    start_vector : BaseStartVector. Generates the start vector that will be
            used.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first
            index to be penalised. Default is 0, all columns are included.

    mean : Boolean. Whether to compute the squared loss or the mean squared
            loss. Default is True, the mean squared loss.

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.estimators as estimators
    >>> import parsimony.algorithms.explicit as explicit
    >>> import parsimony.functions.nesterov.tv as total_variation
    >>> n = 10
    >>> p = 16
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(n, p)
    >>> y = np.random.rand(n, 1)
    >>> l = 0.1  # Regularisation coefficient
    >>> en = estimators.ElasticNet(l,
    ...                            algorithm=explicit.FISTA(),
    ...                            algorithm_params=dict(max_iter=1000),
    ...                            mean=False)
    >>> error = en.fit(X, y).score(X, y)
    >>> print "error = ", error
    error =  0.492096328053
    """
    def __init__(self, l, algorithm=None, algorithm_params=dict(),
                 start_vector=start_vectors.RandomStartVector(),
                 penalty_start=0, mean=True):

        if algorithm is None:
            algorithm = explicit.FISTA(**algorithm_params)
        else:
            algorithm.set_params(**algorithm_params)

        super(ElasticNet, self).__init__(algorithm=algorithm,
                                    start_vector=start_vector)

        self.l = float(l)

        self.penalty_start = int(penalty_start)
        self.mean = bool(mean)

    def get_params(self):
        """Return a dictionary containing the estimator's parameters.
        """
        return {"l": self.l,
                "penalty_start": self.penalty_start, "mean": self.mean}

    def fit(self, X, y, beta=None):
        """Fit the estimator to the data.
        """
        X, y = check_arrays(X, y)

        function = functions.CombinedFunction()
        function.add_function(losses.LinearRegression(X, y, mean=self.mean))
        function.add_penalty(penalties.L2(l=1.0 - self.l,
                                          penalty_start=self.penalty_start))
        function.add_prox(penalties.L1(l=self.l,
                                       penalty_start=self.penalty_start))

        self.algorithm.check_compatibility(function,
                                           self.algorithm.INTERFACES)

        # TODO: Should we use a seed here so that we get deterministic results?
        if beta is None:
            beta = self.start_vector.get_vector(X.shape[1])

        self.beta = self.algorithm.run(function, beta)

        return self

    def score(self, X, y):
        """Returns the (mean) squared error of the estimator.
        """
        X, y = check_arrays(X, y)
        n, p = X.shape
        y_hat = np.dot(X, self.beta)
        err = np.sum((y_hat - y) ** 2.0)
        if self.mean:
            err /= float(n)

        return err


class LinearRegressionL1L2TV(RegressionEstimator):
    """Linear regression with L1, L2 and TV penalties:

        f(beta, X, y) = (1 / 2 * n) * ||Xbeta - y||²_2
                        + l1 * ||beta||_1
                        + (l2 / 2) * ||beta||²_2
                        + tv * TV(beta)

    Parameters
    ----------
    l1 : Non-negative float. The L1 regularization parameter.

    l2 : Non-negative float. The L2 regularization parameter.

    tv : Non-negative float. The total variation regularization parameter.

    A : Numpy or (usually) scipy.sparse array. The linear operator for the
            smoothed total variation Nesterov function. A must be given.

    mu : Non-negative float. The regularisation constant for the smoothing.

    algorithm : ExplicitAlgorithm. The algorithm that should be applied.
            Should be one of:
                1. StaticCONESTA(...)
                2. DynamicCONESTA(...)
                3. FISTA(...)
                4. ISTA(...)

            Default is StaticCONESTA(...).

    algorithm_params : A dict. The dictionary algorithm_params contains
            parameters that should be set in the algorithm. Passing
            algorithm=StaticCONESTA(**params) is equivalent to passing
            algorithm=StaticCONESTA() and algorithm_params=params. Default is
            an empty dictionary.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first
            index to be penalised. Default is 0, all columns are included.

    mean : Boolean. Whether to compute the squared loss or the mean squared
            loss. Default is True, the mean squared loss.

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.estimators as estimators
    >>> import parsimony.algorithms.explicit as explicit
    >>> import parsimony.functions.nesterov.tv as total_variation
    >>> shape = (1, 4, 4)
    >>> n = 10
    >>> p = shape[0] * shape[1] * shape[2]
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(n, p)
    >>> y = np.random.rand(n, 1)
    >>> l1 = 0.1  # L1 coefficient
    >>> l2 = 0.9  # Ridge coefficient
    >>> tv = 1.0  # TV coefficient
    >>> A, n_compacts = total_variation.A_from_shape(shape)
    >>> lr = estimators.LinearRegressionL1L2TV(l1, l2, tv, A,
    ...                        algorithm=explicit.StaticCONESTA(max_iter=1000),
    ...                        mean=False)
    >>> res = lr.fit(X, y)
    >>> error = lr.score(X, y)
    >>> print "error = ", error
    error =  0.0683839364837
    >>> lr = estimators.LinearRegressionL1L2TV(l1, l2, tv, A,
    ...                       algorithm=explicit.DynamicCONESTA(max_iter=1000),
    ...                       mean=False)
    >>> res = lr.fit(X, y)
    >>> error = lr.score(X, y)
    >>> print "error = ", error
    error =  0.0683839265093
    >>> lr = estimators.LinearRegressionL1L2TV(l1, l2, tv, A,
    ...                                algorithm=explicit.FISTA(max_iter=1000),
    ...                                mean=False)
    >>> lr = lr.fit(X, y)
    >>> error = lr.score(X, y)
    >>> print "error = ", error
    error =  1.58175771272
    >>> lr = estimators.LinearRegressionL1L2TV(l1, l2, tv, A,
    ...                                 algorithm=explicit.ISTA(max_iter=1000),
    ...                                 mean=False)
    >>> lr = lr.fit(X, y)
    >>> error = lr.score(X, y)
    >>> print "error = ", error
    error =  2.075830689
    """
    def __init__(self, l1, l2, tv,
                 A=None, mu=consts.TOLERANCE,
                 algorithm=None, algorithm_params=dict(),
                 penalty_start=0,
                 mean=True):

        if algorithm is None:
            algorithm = explicit.StaticCONESTA(**algorithm_params)
        else:
            algorithm.set_params(**algorithm_params)

        super(LinearRegressionL1L2TV, self).__init__(algorithm=algorithm)

        self.l1 = float(l1)
        self.l2 = float(l2)
        self.tv = float(tv)

        if A is None:
            raise TypeError("A may not be None.")
        self.A = A

        try:
            self.mu = float(mu)
        except (ValueError, TypeError):
            self.mu = None

        self.penalty_start = int(penalty_start)
        self.mean = bool(mean)

    def get_params(self):
        """Return a dictionary containing all the estimator's parameters
        """
        return {"l1": self.l1, "l2": self.l2, "tv": self.tv,
                "A": self.A, "mu": self.mu,
                "penalty_start": self.penalty_start, "mean": self.mean}

    def fit(self, X, y, beta=None):
        """Fit the estimator to the data.
        """
        X, y = check_arrays(X, y)

        function = functions.LinearRegressionL1L2TV(X, y,
                                              self.l2, self.l1, self.tv,
                                              A=self.A,
                                              penalty_start=self.penalty_start,
                                              mean=self.mean)
        self.algorithm.check_compatibility(function,
                                           self.algorithm.INTERFACES)

        # TODO: Should we use a seed here so that we get deterministic results?
        if beta is None:
            beta = self.start_vector.get_vector(X.shape[1])

        if self.mu is None:
            self.mu = function.estimate_mu(beta)

        function.set_params(mu=self.mu)
        self.beta = self.algorithm.run(function, beta)

        return self

    def score(self, X, y):
        """Return the mean squared error of the estimator.
        """
        X, y = check_arrays(X, y)
        n, p = X.shape
        y_hat = np.dot(X, self.beta)
        return np.sum((y_hat - y) ** 2.0) / float(n)


class LinearRegressionL1L2GL(RegressionEstimator):
    """Linear regression with L1, L2 and Group lasso penalties:

        f(beta, X, y) = (1 / 2 * n) * ||Xbeta - y||²_2
                        + l1 * ||beta||_1
                        + (l2 / 2) * ||beta||²_2
                        + gl * GL(beta)

    Parameters
    ----------
    l1 : Non-negative float. The L1 regularization parameter.

    l2 : Non-negative float. The L2 regularization parameter.

    tv : Non-negative float. The group lasso regularization parameter.

    A : Numpy or (usually) scipy.sparse array. The linear operator for the
            smoothed group lasso Nesterov function. A must be given.

    mu : Non-negative float. The regularisation constant for the smoothing.

    algorithm : ExplicitAlgorithm. The algorithm that should be applied.
            Should be one of:
                1. StaticCONESTA(...)
                2. DynamicCONESTA(...)
                3. FISTA(...)
                4. ISTA(...)

            Default is StaticCONESTA(...).

    algorithm_params : A dict. The dictionary algorithm_params contains
            parameters that should be set in the algorithm. Passing
            algorithm=StaticCONESTA(**params) is equivalent to passing
            algorithm=StaticCONESTA() and algorithm_params=params. Default is
            an empty dictionary.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first
            index to be penalised. Default is 0, all columns are included.

    mean : Boolean. Whether to compute the squared loss or the mean squared
            loss. Default is True, the mean squared loss.

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.estimators as estimators
    >>> import parsimony.algorithms.explicit as explicit
    >>> import parsimony.functions.nesterov.gl as group_lasso
    >>> n = 10
    >>> p = 15
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(n, p)
    >>> y = np.random.rand(n, 1)
    >>> l1 = 0.1  # L1 coefficient
    >>> l2 = 0.9  # Ridge coefficient
    >>> gl = 1.0  # GL coefficient
    >>> groups = [range(0, 10), range(5, 15)]
    >>> A = group_lasso.A_from_groups(p, groups, weights=None, penalty_start=0)
    >>> lr = estimators.LinearRegressionL1L2GL(l1, l2, gl, A,
    ...                                   algorithm=explicit.StaticCONESTA(),
    ...                                   algorithm_params=dict(max_iter=1000),
    ...                                   mean=False)
    >>> res = lr.fit(X, y)
    >>> error = lr.score(X, y)
    >>> print "error = ", error
    error =  0.610191919564
    >>> lr = estimators.LinearRegressionL1L2GL(l1, l2, gl, A,
    ...                                   algorithm=explicit.DynamicCONESTA(),
    ...                                   algorithm_params=dict(max_iter=1000),
    ...                                   mean=False)
    >>> res = lr.fit(X, y)
    >>> error = lr.score(X, y)
    >>> print "error = ", error
    error =  0.610280682745
    >>> lr = estimators.LinearRegressionL1L2GL(l1, l2, gl, A,
    ...                                   algorithm=explicit.FISTA(),
    ...                                   algorithm_params=dict(max_iter=1000),
    ...                                   mean=False)
    >>> lr = lr.fit(X, y)
    >>> error = lr.score(X, y)
    >>> print "error = ", error
    error =  10.7465249393
    >>> lr = estimators.LinearRegressionL1L2GL(l1, l2, gl, A,
    ...                                   algorithm=explicit.ISTA(),
    ...                                   algorithm_params=dict(max_iter=1000),
    ...                                   mean=False)
    >>> lr = lr.fit(X, y)
    >>> error = lr.score(X, y)
    >>> print "error = ", error
    error =  11.0246211425
    """
    def __init__(self, l1, l2, gl,
                 A=None, mu=consts.TOLERANCE,
                 algorithm=None, algorithm_params=dict(),
                 penalty_start=0,
                 mean=True):

        if algorithm is None:
            algorithm = explicit.StaticCONESTA(**algorithm_params)
        else:
            algorithm.set_params(**algorithm_params)

        super(LinearRegressionL1L2GL, self).__init__(algorithm=algorithm)

        self.l1 = float(l1)
        self.l2 = float(l2)
        self.gl = float(gl)

        if A is None:
            raise TypeError("A may not be None.")
        self.A = A

        try:
            self.mu = float(mu)
        except (ValueError, TypeError):
            self.mu = None

        self.penalty_start = int(penalty_start)
        self.mean = bool(mean)

    def get_params(self):
        """Return a dictionary containing all the estimator's parameters.
        """
        return {"l1": self.l1, "l2": self.l2, "gl": self.gl,
                "A": self.A, "mu": self.mu,
                "penalty_start": self.penalty_start,
                "mean": self.mean}

    def fit(self, X, y, beta=None):
        """Fit the estimator to the data
        """
        X, y = check_arrays(X, y)

        function = functions.LinearRegressionL1L2GL(X, y,
                                              self.l1, self.l2, self.gl,
                                              A=self.A,
                                              penalty_start=self.penalty_start,
                                              mean=self.mean)
        self.algorithm.check_compatibility(function,
                                           self.algorithm.INTERFACES)

        # TODO: Should we use a seed here so that we get deterministic results?
        if beta is None:
            beta = self.start_vector.get_vector(X.shape[1])

        if self.mu is None:
            self.mu = function.estimate_mu(beta)

        function.set_params(mu=self.mu)
        self.beta = self.algorithm.run(function, beta)

        return self

    def score(self, X, y):
        """Return the (mean) squared error of the estimator.
        """
        X, y = check_arrays(X, y)
        n, p = X.shape
        y_hat = np.dot(X, self.beta)
        err = np.sum((y_hat - y) ** 2.0)
        if self.mean:
            err /= float(n)

        return err


#class RidgeRegression_L1_GL(RegressionEstimator):
#    """
#    Parameters
#    ----------
#    k : Non-negative float. The L2 regularisation parameter.
#
#    l : Non-negative float. The L1 regularisation parameter.
#
#    g : Non-negative float. The Group lasso regularisation parameter.
#
#    A : Numpy or (usually) scipy.sparse array. The linear operator for the
#            smoothed group lasso Nesterov function.
#
#    mu : Non-negative float. The regularisation constant for the smoothing.
#
#    algorithm : ExplicitAlgorithm. The algorithm that should be applied.
#            Should be one of:
#                1. algorithms.StaticCONESTA()
#                2. algorithms.DynamicCONESTA()
#                3. algorithms.FISTA()
#                4. algorithms.ISTA()
#
#    penalty_start : Non-negative integer. The number of columns, variables
#            etc., to be exempt from penalisation. Equivalently, the first
#            index to be penalised. Default is 0, all columns are included.
#
#    mean : Boolean. Whether to compute the squared loss or the mean
#            squared loss. Default is True, the mean squared loss.
#
#    Examples
#    --------
##    >>> import numpy as np
##    >>> import parsimony.estimators as estimators
##    >>> import parsimony.algorithms.explicit as explicit
##    >>> import parsimony.functions.nesterov.tv as tv
##    >>> shape = (1, 4, 4)
##    >>> num_samples = 10
##    >>> num_ft = shape[0] * shape[1] * shape[2]
##    >>> np.random.seed(seed=1)
##    >>> X = np.random.random((num_samples, num_ft))
##    >>> y = np.random.randint(0, 2, (num_samples, 1))
##    >>> k = 0.9  # ridge regression coefficient
##    >>> l = 0.1  # l1 coefficient
##    >>> g = 1.0  # tv coefficient
##    >>> A, n_compacts = tv.A_from_shape(shape)
##    >>> ridge_l1_tv = estimators.RidgeRegression_L1_TV(k, l, g, A,
##    ...                     algorithm=explicit.StaticCONESTA(max_iter=1000))
##    >>> res = ridge_l1_tv.fit(X, y)
##    >>> error = np.sum(np.abs(np.dot(X, ridge_l1_tv.beta) - y))
##    >>> print "error = ", error
##    error =  4.70079220678
##    >>> ridge_l1_tv = estimators.RidgeRegression_L1_TV(k, l, g, A,
##    ...                     algorithm=explicit.DynamicCONESTA(max_iter=1000))
##    >>> res = ridge_l1_tv.fit(X, y)
##    >>> error = np.sum(np.abs(np.dot(X, ridge_l1_tv.beta) - y))
##    >>> print "error = ", error
##    error =  4.70096544168
##    >>> ridge_l1_tv = estimators.RidgeRegression_L1_TV(k, l, g, A,
##    ...                     algorithm=explicit.FISTA(max_iter=1000))
##    >>> res = ridge_l1_tv.fit(X, y)
##    >>> error = np.sum(np.abs(np.dot(X, ridge_l1_tv.beta) - y))
##    >>> print "error = ", error
##    error =  4.24400179809
#    """
#    def __init__(self, k, l, g, A, mu=None,
#                 algorithm=explicit.StaticCONESTA(),
##                 algorithm=algorithms.DynamicCONESTA()):
##                 algorithm=algorithms.FISTA()):
#                 penalty_start=0, mean=True):
#
#        super(RidgeRegression_L1_GL, self).__init__(algorithm=algorithm)
#
#        self.k = float(k)
#        self.l = float(l)
#        self.g = float(g)
#        self.A = A
#        try:
#            self.mu = float(mu)
#        except (ValueError, TypeError):
#            self.mu = None
#        self.penalty_start = int(penalty_start)
#        self.mean = bool(mean)
#
#    def get_params(self):
#        """Return a dictionary containing all the estimator's parameters.
#        """
#        return {"k": self.k, "l": self.l, "g": self.g,
#                "A": self.A, "mu": self.mu,
#                "penalty_start": self.penalty_start,
#                "mean": self.mean}
#
#    def fit(self, X, y, beta=None):
#        """Fit the estimator to the data
#        """
#        X, y = check_arrays(X, y)
#        self.function = functions.LinearRegressionL1L2GL(X, y,
#                                             self.k, self.l, self.g,
#                                             A=self.A,
#                                             penalty_start=self.penalty_start,
#                                             mean=self.mean)
#        self.algorithm.check_compatibility(self.function,
#                                           self.algorithm.INTERFACES)
#
#        # TODO: Should we use a seed here so that we get deterministic results?
#        if beta is None:
#            beta = self.start_vector.get_vector(X.shape[1])
#
#        if self.mu is None:
#            self.mu = self.function.estimate_mu(beta)
#        else:
#            self.mu = float(self.mu)
#
#        self.function.set_params(mu=self.mu)
#        self.beta = self.algorithm.run(self.function, beta)
#
#        return self
#
#    def score(self, X, y):
#        """Return the mean squared error of the estimator.
#        """
#        X, y = check_arrays(X, y)
#        n, p = X.shape
#        y_hat = np.dot(X, self.beta)
#        return np.sum((y_hat - y) ** 2.0) / float(n)


class LogisticRegressionL1L2TV(LogisticRegressionEstimator):
    """Logistic regression (re-weighted log-likelihood aka. cross-entropy)
    with L1, L2 and TV penalties:

        f(beta) = - loglik/n_samples
                  + l1 * ||beta||_1
                  + (l2 / 2 * n) * ||beta||²_2
                  + tv * TV(beta)
    where
        loglik = Sum wi * (yi * log(pi) + (1 − yi) * log(1 − pi)),

        pi = p(y=1|xi, beta) = 1 / (1 + exp(-xi'*beta)),

        wi = weight of sample i.

    Parameters
    ----------
    l1 : Non-negative float. The L1 regularization parameter.

    l2 : Non-negative float. The L2 regularization parameter.

    tv : Non-negative float. The total variation regularization parameter.

    A : Numpy or (usually) scipy.sparse array. The linear operator for the
            smoothed total variation Nesterov function. A must be given.

    mu : Non-negative float. The regularisation constant for the smoothing.

    algorithm : ExplicitAlgorithm. The algorithm that should be applied.
            Should be one of:
                1. StaticCONESTA(...)
                2. FISTA(...)
                3. ISTA(...)

            Default is StaticCONESTA(...).

    algorithm_params : A dict. The dictionary algorithm_params contains
            parameters that should be set in the algorithm. Passing
            algorithm=StaticCONESTA(**params) is equivalent to passing
            algorithm=StaticCONESTA() and algorithm_params=params. Default is
            an empty dictionary.

    class_weight : Dict, 'auto' or None. If 'auto', class weights will be
            given inverse proportional to the frequency of the class in
            the data. If a dictionary is given, keys are classes and values
            are corresponding class weights. If None is given, the class
            weights will be uniform.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first
            index to be penalised. Default is 0, all columns are included.

    mean : Boolean. Whether to compute the squared loss or the mean squared
            loss. Default is True, the mean squared loss.

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.estimators as estimators
    >>> import parsimony.algorithms.explicit as explicit
    >>> import parsimony.functions.nesterov.tv as total_variation
    >>> shape = (1, 4, 4)
    >>> n = 10
    >>> p = shape[0] * shape[1] * shape[2]
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(n, p)
    >>> y = np.random.randint(0, 2, (n, 1))
    >>> l1 = 0.1  # L1 coefficient
    >>> l2 = 0.9  # Ridge coefficient
    >>> tv = 1.0  # TV coefficient
    >>> A, n_compacts = total_variation.A_from_shape(shape)
    >>> lr = estimators.LogisticRegressionL1L2TV(l1, l2, tv, A,
    ...                        algorithm=explicit.StaticCONESTA(max_iter=1000),
    ...                        mean=False)
    >>> res = lr.fit(X, y)
    >>> error = lr.score(X, y)
    >>> print "error = ", error
    error =  0.7
    >>> lr = estimators.LogisticRegressionL1L2TV(l1, l2, tv, A,
    ...                                algorithm=explicit.FISTA(max_iter=1000),
    ...                                mean=False)
    >>> lr = lr.fit(X, y)
    >>> error = lr.score(X, y)
    >>> print "error = ", error
    error =  0.7
    >>> lr = estimators.LogisticRegressionL1L2TV(l1, l2, tv, A,
    ...                                 algorithm=explicit.ISTA(max_iter=1000),
    ...                                 mean=False)
    >>> lr = lr.fit(X, y)
    >>> error = lr.score(X, y)
    >>> print "error = ", error
    error =  0.7
    """
    def __init__(self, l1, l2, tv,
                 A=None, mu=consts.TOLERANCE,
                 algorithm=None, algorithm_params=dict(),
                 class_weight=None,
                 penalty_start=0,
                 mean=True):

        if algorithm is None:
            algorithm = explicit.StaticCONESTA(**algorithm_params)
        else:
            algorithm.set_params(**algorithm_params)

        super(LogisticRegressionL1L2TV, self).__init__(algorithm=algorithm,
                                                     class_weight=class_weight)

        self.l1 = float(l1)
        self.l2 = float(l2)
        self.tv = float(tv)

        if isinstance(algorithm, explicit.CONESTA) \
                and self.tv < consts.TOLERANCE:
            warnings.warn("The TV parameter should be positive.")

        if A is None:
            raise TypeError("A may not be None.")
        self.A = A

        try:
            self.mu = float(mu)
        except (ValueError, TypeError):
            self.mu = None

        self.penalty_start = int(penalty_start)
        self.mean = bool(mean)

    def get_params(self):
        """Return a dictionary containing all the estimator's parameters.
        """
        return {"l1": self.l1, "l2": self.l2, "tv": self.tv,
                "A": self.A, "mu": self.mu, "class_weight": self.class_weight,
                "penalty_start": self.penalty_start, "mean": self.mean}

    def fit(self, X, y, beta=None, sample_weight=None):
        """Fit the estimator to the data.
        """
        X, y = check_arrays(X, check_labels(y))
        if sample_weight is None:
            sample_weight = class_weight_to_sample_weight(self.class_weight, y)
        y, sample_weight = check_arrays(y, sample_weight)
            #sample_weight = sample_weight.ravel()

        function = functions.LogisticRegressionL1L2TV(X, y,
                                              self.l2, self.l1, self.tv,
                                              A=self.A,
                                              weights=sample_weight,
                                              penalty_start=self.penalty_start,
                                              mean=self.mean)

        self.algorithm.check_compatibility(function,
                                           self.algorithm.INTERFACES)

        # TODO: Should we use a seed here so that we get deterministic results?
        if beta is None:
            beta = self.start_vector.get_vector(X.shape[1])

        if self.mu is None:
            self.mu = function.estimate_mu(beta)
        else:
            self.mu = float(self.mu)

        function.set_params(mu=self.mu)
        self.beta = self.algorithm.run(function, beta)

        return self


class LogisticRegressionL1L2GL(LogisticRegressionEstimator):
    """Logistic regression (re-weighted log-likelihood aka. cross-entropy)
    with L1, L2 and Group Lasso penalties:

        f(beta) = - loglik/n_samples
                  + l1 * ||beta||_1
                  + (l2 / 2 * n) * ||beta||²_2
                  + gl * GL(beta)
    where
        loglik = Sum wi * (yi * log(pi) + (1 − yi) * log(1 − pi)),

        pi = p(y=1|xi, beta) = 1 / (1 + exp(-xi'*beta)),

        wi = weight of sample i.

    Parameters
    ----------
    l1 : Non-negative float. The L1 regularization parameter.

    l2 : Non-negative float. The L2 regularization parameter.

    gl : Non-negative float. The group lasso regularization parameter.

    A : Numpy or (usually) scipy.sparse array. The linear operator for the
            smoothed total variation Nesterov function. A must be given.

    mu : Non-negative float. The regularisation constant for the Nesterov
            smoothing.

    algorithm : ExplicitAlgorithm. The algorithm that should be applied.
            Should be one of:
                1. StaticCONESTA(...)
                2. FISTA(...)
                3. ISTA(...)

            Default is StaticCONESTA(...).

    algorithm_params : A dict. The dictionary algorithm_params contains
            parameters that should be set in the algorithm. Passing
            algorithm=StaticCONESTA(**params) is equivalent to passing
            algorithm=StaticCONESTA() and algorithm_params=params. Default is
            an empty dictionary.

    class_weight : Dict, 'auto' or None. If 'auto', class weights will be
            given inverse proportional to the frequency of the class in
            the data. If a dictionary is given, keys are classes and values
            are corresponding class weights. If None is given, the class
            weights will be uniform.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first
            index to be penalised. Default is 0, all columns are included.

    mean : Boolean. Whether to compute the squared loss or the mean squared
            loss. Default is True, the mean squared loss.

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.estimators as estimators
    >>> import parsimony.algorithms.explicit as explicit
    >>> import parsimony.functions.nesterov.gl as group_lasso
    >>>
    >>> np.random.seed(42)
    >>>
    >>> n, p = 10, 16
    >>> groups = [range(0, p / 2), range(p / 2, p)]
    >>> weights = [1.5, 0.5]
    >>> A = group_lasso.A_from_groups(p, groups=groups, weights=weights)
    >>>
    >>> X = np.random.rand(n, p)
    >>> y = np.random.randint(0, 2, (n, 1))
    >>> l1 = 0.1  # L1 coefficient
    >>> l2 = 0.9  # Ridge coefficient
    >>> tv = 1.0  # TV coefficient
    >>> lr = estimators.LogisticRegressionL1L2GL(l1, l2, tv, A=A,
    ...                        algorithm=explicit.StaticCONESTA(max_iter=1000),
    ...                        mean=False)
    >>> res = lr.fit(X, y)
    >>> error = lr.score(X, y)
    >>> print "error = ", error
    error =  0.7
    >>> lr = estimators.LogisticRegressionL1L2GL(l1, l2, tv, A=A,
    ...                                algorithm=explicit.FISTA(max_iter=1000),
    ...                                mean=False)
    >>> lr = lr.fit(X, y)
    >>> error = lr.score(X, y)
    >>> print "error = ", error
    error =  0.7
    >>> lr = estimators.LogisticRegressionL1L2GL(l1, l2, tv, A,
    ...                                 algorithm=explicit.ISTA(max_iter=1000),
    ...                                 mean=False)
    >>> lr = lr.fit(X, y)
    >>> error = lr.score(X, y)
    >>> print "error = ", error
    error =  0.7
    """
    def __init__(self, l1, l2, gl,
                 A=None, mu=consts.TOLERANCE,
                 weigths=None,
                 algorithm=None, algorithm_params=dict(),
                 class_weight=None,
                 penalty_start=0,
                 mean=True):

        if algorithm is None:
            algorithm = explicit.StaticCONESTA(**algorithm_params)
        else:
            algorithm.set_params(**algorithm_params)

        super(LogisticRegressionL1L2GL, self).__init__(algorithm=algorithm,
                                                     class_weight=class_weight)

        self.l1 = float(l1)
        self.l2 = float(l2)
        self.gl = float(gl)

        if isinstance(algorithm, explicit.CONESTA) \
                and self.gl < consts.TOLERANCE:
            warnings.warn("The GL parameter should be positive.")

        if A is None:
            raise TypeError("A may not be None.")
        self.A = A

        try:
            self.mu = float(mu)
        except (ValueError, TypeError):
            self.mu = None

        self.penalty_start = int(penalty_start)
        self.mean = bool(mean)

    def get_params(self):
        """Returns a dictionary containing all the estimator's parameters.
        """
        return {"l1": self.l1, "l2": self.l2, "gl": self.gl,
                "A": self.A, "mu": self.mu, "class_weight": self.class_weight,
                "penalty_start": self.penalty_start, "mean": self.mean}

    def fit(self, X, y, beta=None, sample_weight=None):
        """Fit the estimator to the data.
        """
        X, y = check_arrays(X, check_labels(y))
        if sample_weight is None:
            sample_weight = class_weight_to_sample_weight(self.class_weight, y)
        y, sample_weight = check_arrays(y, sample_weight)
        function = functions.LogisticRegressionL1L2GL(X, y,
                                              self.l2, self.l1, self.gl,
                                              A=self.A,
                                              weights=sample_weight,
                                              penalty_start=self.penalty_start,
                                              mean=self.mean)

        self.algorithm.check_compatibility(function,
                                           self.algorithm.INTERFACES)

        # TODO: Should we use a seed here so that we get deterministic results?
        if beta is None:
            beta = self.start_vector.get_vector(X.shape[1])

        if self.mu is None:
            self.mu = function.estimate_mu(beta)
        else:
            self.mu = float(self.mu)

        function.set_params(mu=self.mu)
        self.beta = self.algorithm.run(function, beta)

        return self


class LinearRegressionL2SmoothedL1TV(RegressionEstimator):
    """Linear regression with L2 and simultaneously smoothed L1 and TV
    penalties:

        f(beta, X, y) = (1 / 2 * n) * ||Xbeta - y||²_2
                        + (l2 / 2) * ||beta||²_2
                        + L1TV(beta),

    where L1TV is l1 * L1(beta) + tv * TV(beta) smoothed together by Nesterov's
    smoothing.

    Parameters
    ----------
    l1 : Non-negative float. The L1 regularization parameter.

    l2 : Non-negative float. The L2 regularization parameter.

    tv : Non-negative float. The total variation regularization parameter.

    Atv : Numpy or (usually) scipy.sparse array. The linear operator for the
            smoothed total variation Nesterov function. A must be given.

    Al1 : Numpy or (usually) scipy.sparse array. The linear operator for the
            smoothed L1 Nesterov function. A must be given.

    mu : Non-negative float. The regularisation constant for the smoothing.

    algorithm : ExplicitAlgorithm. The algorithm that should be applied.
            Should be one of:
                1. ExcessiveGapMethod(...)

            Default is ExcessiveGapMethod(...).

    algorithm_params : A dict. The dictionary algorithm_params contains
            parameters that should be set in the algorithm. Passing
            algorithm=ExcessiveGapMethod(**params) is equivalent to passing
            algorithm=ExcessiveGapMethod() and algorithm_params=params. Default
            is an empty dictionary.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first
            index to be penalised. Default is 0, all columns are included.

    mean : Boolean. Whether to compute the squared loss or the mean squared
            loss. Default is True, the mean squared loss.

    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.estimators as estimators
    >>> import parsimony.algorithms.explicit as explicit
    >>> import parsimony.functions.nesterov.l1tv as l1tv
    >>> shape = (1, 4, 4)
    >>> n = 10
    >>> p = shape[0] * shape[1] * shape[2]
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.rand(n, p)
    >>> y = np.random.rand(n, 1)
    >>> l1 = 0.1  # L1 coefficient
    >>> l2 = 0.9  # Ridge coefficient
    >>> tv = 1.0  # TV coefficient
    >>> Atv, Al1 = l1tv.A_from_shape(shape, p, penalty_start=0)
    >>> lr = estimators.LinearRegressionL2SmoothedL1TV(l1, l2, tv,
    ...                 Atv=Atv, Al1=Al1,
    ...                 algorithm=explicit.ExcessiveGapMethod(),
    ...                 algorithm_params=dict(max_iter=1000))
    >>> res = lr.fit(X, y)
    >>> error = lr.score(X, y)
    >>> print "error = ", error
    error =  0.101661632451
    """
    def __init__(self, l1, l2, tv,
                 Atv=None, Al1=None,
                 algorithm=None, algorithm_params=dict(),
                 penalty_start=0,
                 mean=True):

        if algorithm is None:
            algorithm = explicit.ExcessiveGapMethod(**algorithm_params)
        else:
            algorithm.set_params(**algorithm_params)

        super(LinearRegressionL2SmoothedL1TV, self).__init__(
                                                     algorithm=algorithm)

        self.l1 = float(l1)
        self.l2 = float(l2)
        self.tv = float(tv)

        if self.l2 < consts.TOLERANCE:
            warnings.warn("The ridge parameter should be non-zero.")

        if Atv is None:
            raise TypeError("Atv may not be None.")
        self.Atv = Atv
        if Al1 is None:
            raise TypeError("Al1 may not be None.")
        self.Al1 = Al1

        self.penalty_start = int(penalty_start)
        self.mean = bool(mean)

    def get_params(self):
        """Returns a dictionary containing all the estimator's parameters.
        """
        return {"l1": self.l1, "l2": self.l2, "tv": self.tv,
                "Atv": self.Atv, "Al1": self.Al1,
                "penalty_start": self.penalty_start, "mean": self.mean}

    def fit(self, X, y, beta=None):
        """Fit the estimator to the data.
        """
        X, y = check_arrays(X, y)
        function = functions.LinearRegressionL2SmoothedL1TV(X, y,
                                              self.l2, self.l1, self.tv,
                                              Atv=self.Atv, Al1=self.Al1,
                                              penalty_start=self.penalty_start,
                                              mean=self.mean)

        self.algorithm.check_compatibility(function,
                                           self.algorithm.INTERFACES)

        self.beta = self.algorithm.run(function, beta)

        return self

    def score(self, X, y):
        """Return the mean squared error of the estimator.
        """
        n, p = X.shape
        y_hat = np.dot(X, self.beta)
        return np.sum((y_hat - y) ** 2.0) / float(n)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
