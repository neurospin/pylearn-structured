# -*- coding: utf-8 -*-
"""Estimators encapsulates a loss function with penalties and a corresponding
algorithm.

The naming convention is such that an underscore, '_', implies plus, such that
if

    f(x) = a + (b + c),

where "(b + c)" means that b and c are encapsulated as a single function, then
the name of that estimator should be a_bc.

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
import parsimony.functions.nesterov.tv as tv
import parsimony.algorithms.explicit as explicit
import parsimony.start_vectors as start_vectors
from parsimony.utils import check_arrays
from parsimony.utils import class_weight_to_sample_weight, check_labels

__all__ = ["BaseEstimator", "RegressionEstimator",

           "LinearRegression_L1_L2_TV",
           "RidgeRegression_L1_TV", "RidgeLogisticRegression_L1_TV",

           "RidgeRegression_SmoothedL1TV"]

# TODO: Add penalty_start and mean to all of these!


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
        """Fit the estimator to the data.
        """
        raise NotImplementedError('Abstract method "fit" must be '
                                  'specialised!')

#        self.function.set_params(X=X, y=y)
#        # TODO: Should we use a seed here so that we get deterministic
#        # results?
#        beta = self.start_vector.get_vector((X.shape[1], 1))
#        if self.output:
#            self.beta, self.output = self.algorithm(X, y, self.function, beta)
#        else:
#            self.beta = self.algorithm(X, y, self.function, beta)

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

    output : Boolean. Whether or not to return extra output information.

    start_vector : Numpy array. Generates the start vector that will be used.

    class_weight : {dict, "auto"}, optional. Set the parameter weight of
            sample belonging to class i to class_weight[i]. If not given, all
            classes are supposed to have weight one. The "auto" mode uses the
            values of y to automatically adjust weights inversely proportional
            to class frequencies.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, algorithm, output=False,
                 start_vector=start_vectors.RandomStartVector(),
                 class_weight=None):

        self.output = output
        self.start_vector = start_vector
        self.class_weight = class_weight

        super(LogisticRegressionEstimator, self).__init__(algorithm=algorithm)

    @abc.abstractmethod
    def fit(self, X, y):
        """Fit the model to the data.
        """
        raise NotImplementedError('Abstract method "fit" must be '
                                  'specialised!')

#        self.function.set_params(X=X, y=y)
#        # TODO: Should we use a seed here so that we get deterministic
#        # results?
#        beta = self.start_vector.get_vector((X.shape[1], 1))
#        if self.output:
#            self.beta, self.output = self.algorithm(X, y, self.function, beta)
#        else:
#            self.beta = self.algorithm(X, y, self.function, beta)

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
        prob = 1. / (1. + np.exp(-logit))

        return prob

    def score(self, X, y):
        """Rate of correct classification.
        """
        yhat = self.predict(X)
        rate = np.mean(y == yhat)

        return rate


class LinearRegression_L1_L2_TV(RegressionEstimator):
    """
    Examples
    --------
    >>> import numpy as np
    >>> import parsimony.estimators as estimators
    >>> import parsimony.algorithms.explicit as explicit
    >>> import parsimony.functions.nesterov.tv as tv
    >>> shape = (1, 4, 4)
    >>> num_samples = 10
    >>> num_ft = shape[0] * shape[1] * shape[2]
    >>>
    >>> np.random.seed(42)
    >>> X = np.random.random((num_samples, num_ft))
    >>> y = np.random.randint(0, 2, (num_samples, 1))
    >>> l = 0.1  # L1 coefficient
    >>> k = 0.9  # Ridge coefficient
    >>> g = 1.0  # TV coefficient
    >>> A, n_compacts = tv.A_from_shape(shape)
    >>> lr = estimators.LinearRegression_L1_L2_TV(l, k, g, A,
    ...                               algorithm=explicit.FISTA(max_iter=1000))
    >>> lr = lr.fit(X, y)
    >>> error = lr.score(X, y)
    >>> print "error = ", error
    error =  0.87958672586
    >>> lr = estimators.LinearRegression_L1_L2_TV(l, k, g, A,
    ...                                algorithm=explicit.ISTA(max_iter=1000))
    >>> lr = lr.fit(X, y)
    >>> error = lr.score(X, y)
    >>> print "error = ", error
    error =  1.07391299463
    """
    def __init__(self, l, k, g, A, mu=consts.TOLERANCE, output=False,
                 algorithm=explicit.StaticCONESTA(),
#                 algorithm=algorithms.DynamicCONESTA()):
#                 algorithm=algorithms.FISTA()):
                 penalty_start=0, mean=True):
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

        algorithm : ExplicitAlgorithm. The algorithm that should be applied.
                Should be one of:
                    3. algorithms.FISTA()
                    4. algorithms.ISTA()

        penalty_start : Non-negative integer. The number of columns, variables
                etc., to be exempt from penalisation. Equivalently, the first
                index to be penalised. Default is 0, all columns are included.

        mean : Boolean. Whether to compute the squared loss or the mean squared
                loss. Default is True, the mean squared loss.
        """
        self.l = float(l)
        self.k = float(k)
        self.g = float(g)
        self.A = A
        self.mu = float(mu)
        self.penalty_start = int(penalty_start)
        self.mean = bool(mean)

        super(LinearRegression_L1_L2_TV, self).__init__(algorithm=algorithm,
                                                        output=output)

    def get_params(self):
        """Return a dictionary containing all the estimator's parameters
        """
        return {"k": self.k, "l": self.l, "g": self.g,
                "A": self.A, "mu": self.mu,
                "penalty_start": self.penalty_start, "mean": self.mean}

    def fit(self, X, y, beta=None):
        """Fit the estimator to the data.
        """
        X, y = check_arrays(X, y)

        function = functions.RR_L1_TV(X, y, self.k, self.l, self.g,
                                           A=self.A,
                                           penalty_start=self.penalty_start,
                                           mean=self.mean)
        self.algorithm.check_compatibility(function,
                                           self.algorithm.INTERFACES)

        # TODO: Should we use a seed here so that we get deterministic results?
        if beta is None:
            beta = self.start_vector.get_vector((X.shape[1], 1))

#        function.set_params(mu=self.mu)
        self.algorithm.set_params(output=self.output)

        if self.output:
            (self.beta, self.info) = self.algorithm.run(function, beta)
        else:
            self.beta = self.algorithm.run(function, beta)

        return self

    def score(self, X, y):
        """Return the mean squared error of the estimator.
        """
        X, y = check_arrays(X, y)
        n, p = X.shape
        y_hat = np.dot(X, self.beta)
        return np.sum((y_hat - y) ** 2.0) / float(n)


class RidgeRegression_L1_TV(RegressionEstimator):
    """
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
    error =  4.70079219833
    >>> ridge_l1_tv = estimators.RidgeRegression_L1_TV(k, l, g, A,
    ...                     algorithm=explicit.DynamicCONESTA(max_iter=1000))
    >>> res = ridge_l1_tv.fit(X, y)
    >>> error = np.sum(np.abs(np.dot(X, ridge_l1_tv.beta) - y))
    >>> print "error = ", error
    error =  4.70069909262
    >>> ridge_l1_tv = estimators.RidgeRegression_L1_TV(k, l, g, A,
    ...                     algorithm=explicit.FISTA(max_iter=1000))
    >>> res = ridge_l1_tv.fit(X, y)
    >>> error = np.sum(np.abs(np.dot(X, ridge_l1_tv.beta) - y))
    >>> print "error = ", error
    error =  3.9603518627
    """
    def __init__(self, k, l, g, A, mu=None, output=False,
                 algorithm=explicit.StaticCONESTA(),
#                 algorithm=algorithms.DynamicCONESTA()):
#                 algorithm=algorithms.FISTA()):
                 penalty_start=0, mean=True):
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
                    4. algorithms.ISTA()

        penalty_start : Non-negative integer. The number of columns, variables
                etc., to be exempt from penalisation. Equivalently, the first
                index to be penalised. Default is 0, all columns are included.

        mean : Boolean. Whether to compute the squared loss or the mean
                squared loss. Default is True, the mean squared loss.
        """
        self.k = float(k)
        self.l = float(l)
        self.g = float(g)
        self.A = A
        try:
            self.mu = float(mu)
        except (ValueError, TypeError):
            self.mu = None
        self.penalty_start = int(penalty_start)
        self.mean = bool(mean)

        super(RidgeRegression_L1_TV, self).__init__(algorithm=algorithm,
                                                    output=output)

    def get_params(self):
        """Return a dictionary containing all the estimator's parameters
        """
        return {"k": self.k, "l": self.l, "g": self.g,
                "A": self.A, "mu": self.mu,
                "penalty_start": self.penalty_start}

    def fit(self, X, y, beta=None):
        """Fit the estimator to the data.
        """
        X, y = check_arrays(X, y)
        function = functions.RR_L1_TV(X, y, self.k, self.l, self.g,
                                           A=self.A,
                                           penalty_start=self.penalty_start,
                                           mean=self.mean)
        self.algorithm.check_compatibility(function,
                                           self.algorithm.INTERFACES)

        # TODO: Should we use a seed here so that we get deterministic results?
        if beta is None:
            beta = self.start_vector.get_vector((X.shape[1], 1))

        if self.mu is None:
            self.mu = function.estimate_mu(beta)
        else:
            self.mu = float(self.mu)

        function.set_params(mu=self.mu)
        self.algorithm.set_params(output=self.output)

        if self.output:
            (self.beta, self.info) = self.algorithm.run(function, beta)
        else:
            self.beta = self.algorithm.run(function, beta)

        return self

    def score(self, X, y):
        """Return the mean squared error of the estimator.
        """
        X, y = check_arrays(X, y)
        n, p = X.shape
        y_hat = np.dot(X, self.beta)
        return np.sum((y_hat - y) ** 2.0) / float(n)


class RidgeRegression_L1_GL(RegressionEstimator):
    """
    Examples
    --------
#    >>> import numpy as np
#    >>> import parsimony.estimators as estimators
#    >>> import parsimony.algorithms.explicit as explicit
#    >>> import parsimony.functions.nesterov.tv as tv
#    >>> shape = (1, 4, 4)
#    >>> num_samples = 10
#    >>> num_ft = shape[0] * shape[1] * shape[2]
#    >>> np.random.seed(seed=1)
#    >>> X = np.random.random((num_samples, num_ft))
#    >>> y = np.random.randint(0, 2, (num_samples, 1))
#    >>> k = 0.9  # ridge regression coefficient
#    >>> l = 0.1  # l1 coefficient
#    >>> g = 1.0  # tv coefficient
#    >>> A, n_compacts = tv.A_from_shape(shape)
#    >>> ridge_l1_tv = estimators.RidgeRegression_L1_TV(k, l, g, A,
#    ...                     algorithm=explicit.StaticCONESTA(max_iter=1000))
#    >>> res = ridge_l1_tv.fit(X, y)
#    >>> error = np.sum(np.abs(np.dot(X, ridge_l1_tv.beta) - y))
#    >>> print "error = ", error
#    error =  4.70079220678
#    >>> ridge_l1_tv = estimators.RidgeRegression_L1_TV(k, l, g, A,
#    ...                     algorithm=explicit.DynamicCONESTA(max_iter=1000))
#    >>> res = ridge_l1_tv.fit(X, y)
#    >>> error = np.sum(np.abs(np.dot(X, ridge_l1_tv.beta) - y))
#    >>> print "error = ", error
#    error =  4.70096544168
#    >>> ridge_l1_tv = estimators.RidgeRegression_L1_TV(k, l, g, A,
#    ...                     algorithm=explicit.FISTA(max_iter=1000))
#    >>> res = ridge_l1_tv.fit(X, y)
#    >>> error = np.sum(np.abs(np.dot(X, ridge_l1_tv.beta) - y))
#    >>> print "error = ", error
#    error =  4.24400179809
    """
    def __init__(self, k, l, g, A, mu=None, output=False,
                 algorithm=explicit.StaticCONESTA(),
#                 algorithm=algorithms.DynamicCONESTA()):
#                 algorithm=algorithms.FISTA()):
                 penalty_start=0, mean=True):
        """
        Parameters
        ----------
        l : Non-negative float. The L1 regularisation parameter.

        k : Non-negative float. The L2 regularisation parameter.

        g : Non-negative float. The Group lasso regularisation parameter.

        A : Numpy or (usually) scipy.sparse array. The linear operator for the
                smoothed group lasso Nesterov function.

        mu : Non-negative float. The regularisation constant for the smoothing.

        output : Boolean. Whether or not to return extra output information.

        algorithm : ExplicitAlgorithm. The algorithm that should be applied.
                Should be one of:
                    1. algorithms.StaticCONESTA()
                    2. algorithms.DynamicCONESTA()
                    3. algorithms.FISTA()
                    4. algorithms.ISTA()

        penalty_start : Non-negative integer. The number of columns, variables
                etc., to be exempt from penalisation. Equivalently, the first
                index to be penalised. Default is 0, all columns are included.

        mean : Boolean. Whether to compute the squared loss or the mean
                squared loss. Default is True, the mean squared loss.
        """
        self.k = float(k)
        self.l = float(l)
        self.g = float(g)
        self.A = A
        try:
            self.mu = float(mu)
        except (ValueError, TypeError):
            self.mu = None
        self.penalty_start = int(penalty_start)
        self.mean = bool(mean)

        super(RidgeRegression_L1_GL, self).__init__(algorithm=algorithm,
                                                    output=output)

    def get_params(self):
        """Return a dictionary containing all the estimator's parameters.
        """
        return {"k": self.k, "l": self.l, "g": self.g,
                "A": self.A, "mu": self.mu,
                "penalty_start": self.penalty_start,
                "mean": self.mean}

    def fit(self, X, y, beta=None):
        """Fit the estimator to the data
        """
        X, y = check_arrays(X, y)
        self.function = functions.RR_L1_GL(X, y, self.k, self.l, self.g,
                                           A=self.A,
                                           penalty_start=self.penalty_start,
                                           mean=self.mean)
        self.algorithm.check_compatibility(self.function,
                                           self.algorithm.INTERFACES)

        # TODO: Should we use a seed here so that we get deterministic results?
        if beta is None:
            beta = self.start_vector.get_vector((X.shape[1], 1))

        if self.mu is None:
            self.mu = self.function.estimate_mu(beta)
        else:
            self.mu = float(self.mu)

        self.function.set_params(mu=self.mu)
        self.algorithm.set_params(output=self.output)

        if self.output:
            (self.beta, self.info) = self.algorithm.run(self.function, beta)
        else:
            self.beta = self.algorithm.run(self.function, beta)

        return self

    def score(self, X, y):
        """Return the mean squared error of the estimator.
        """
        X, y = check_arrays(X, y)
        n, p = X.shape
        y_hat = np.dot(X, self.beta)
        return np.sum((y_hat - y) ** 2.0) / float(n)


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

    Examples
    --------
    """
    def __init__(self, k, l, g, A, mu=None, output=False,
                 algorithm=explicit.StaticCONESTA(),
#                 algorithm=algorithms.DynamicCONESTA()):
#                 algorithm=algorithms.FISTA()):
                 class_weight=None, penalty_start=0, mean=True):
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

        algorithm : ExplicitAlgorithm. The algorithm that will be run. Should
                be one of:
                    1. algorithms.StaticCONESTA()
                    2. algorithms.DynamicCONESTA()
                    3. algorithms.FISTA()
                    4. algorithms.ISTA()

        class_weight : Dict, 'auto' or None. If 'auto', class weights will be
                given inverse proportional to the frequency of the class in
                the data. If a dictionary is given, keys are classes and values
                are corresponding class weights. If None is given, the class
                weights will be uniform.

        penalty_start : Non-negative integer. The number of columns, variables
                etc., to be exempt from penalisation. Equivalently, the first
                index to be penalised. Default is 0, all columns are included.

        mean : Boolean. Whether to compute the squared loss or the mean
                squared loss. Default is True, the mean squared loss.
        """
        self.k = float(k)
        self.l = float(l)
        self.g = float(g)
        if isinstance(algorithm, explicit.CONESTA) \
                and self.g < consts.TOLERANCE:
            warnings.warn("The GL parameter should be positive.")
        self.A = A
        #self.weigths = weigths
        try:
            self.mu = float(mu)
        except (ValueError, TypeError):
            self.mu = None

        self.penalty_start = int(penalty_start)
        self.mean = bool(mean)

        super(RidgeLogisticRegression_L1_TV,
              self).__init__(algorithm=algorithm, output=output,
                            class_weight=class_weight)

    def get_params(self):
        """Return a dictionary containing all the estimator's parameters.
        """
        return {"k": self.k, "l": self.l, "g": self.g,
                "A": self.A, "mu": self.mu, "class_weight": self.class_weight,
                "penalty_start": self.penalty_start, "mean": self.mean}

    def fit(self, X, y, sample_weight=None):
        """Fit the estimator to the data.
        """
        X, y = check_arrays(X, check_labels(y))
        if sample_weight is None:
            sample_weight = class_weight_to_sample_weight(self.class_weight, y)
        y, sample_weight = check_arrays(y, sample_weight)
            #sample_weight = sample_weight.ravel()

        function = functions.RLR_L1_TV(X, y,
                                       self.k, self.l, self.g,
                                       A=self.A,
                                       weights=sample_weight,
                                       penalty_start=self.penalty_start,
                                       mean=self.mean)
        self.algorithm.check_compatibility(function,
                                           self.algorithm.INTERFACES)

        # TODO: Should we use a seed here so that we get deterministic results?
        beta = self.start_vector.get_vector((X.shape[1], 1))

        if self.mu is None:
            self.mu = 0.9 * function.estimate_mu(beta)
        else:
            self.mu = float(self.mu)

        function.set_params(mu=self.mu)
        self.algorithm.set_params(output=self.output)

        if self.output:
            (self.beta, self.info) = self.algorithm.run(function, beta)
        else:
            self.beta = self.algorithm.run(function, beta)

        return self


class RidgeLogisticRegression_L1_GL(LogisticRegressionEstimator):
    """Minimises RidgeLogisticRegression (re-weighted log-likelihood aka.
    cross-entropy) with L1 and group lasso penalties:

    Ridge (re-weighted) log-likelihood (cross-entropy):

        f(beta, X, y) = - Sum wi * (yi * log(pi) + (1 − yi) * log(1 − pi))
                        + k/2 * ||beta||^2_2
                        + l * ||beta||_1
                        + g * GL(beta)

    With pi = p(y=1|xi, beta) = 1 / (1 + exp(-xi' beta)) and wi the weight of
    sample i.

    Parameters
    ----------
    l : Non-negative float. The L1 regularization parameter.

    k : Non-negative float. The L2 regularization parameter.

    g : Non-negative float. The total variation regularization parameter.

    A : Numpy or (usually) scipy.sparse array. The linear operator for the
            smoothed total variation Nesterov function.

    class_weight : dict, 'auto' or None
        If 'auto', class weights will be given inverse proportional
        to the frequency of the class in the data.
        If a dictionary is given, keys are classes and values
        are corresponding class weights.
        If None is given, the class weights will be uniform.

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
                 algorithm=explicit.StaticCONESTA(), penalty_start=0,
                 mean=True, class_weight=None):
#                 algorithm=algorithms.DynamicCONESTA()):
#                 algorithm=algorithms.FISTA()):
        self.k = float(k)
        self.l = float(l)
        self.g = float(g)
        if isinstance(algorithm, explicit.CONESTA) \
                and self.g < consts.TOLERANCE:
            warnings.warn("The GL parameter should be positive.")
        self.A = A
        self.class_weight = class_weight
        try:
            self.mu = float(mu)
        except (ValueError, TypeError):
            self.mu = None

        self.penalty_start = int(penalty_start)
        self.mean = bool(mean)

        super(RidgeLogisticRegression_L1_GL,
              self).__init__(algorithm=algorithm, output=output)

    def get_params(self):
        """Returns a dictionary containing all the estimator's own parameters.
        """
        return {"k": self.k, "l": self.l, "g": self.g,
                "A": self.A, "mu": self.mu, "class_weight": self.class_weight}

    def fit(self, X, y, sample_weight=None):
        """Fit the estimator to the data.
        """
        X, y = check_arrays(X, check_labels(y))
        if sample_weight is None:
            sample_weight = class_weight_to_sample_weight(self.class_weight, y)
        y, sample_weight = check_arrays(y, sample_weight)
        function = functions.RLR_L1_GL(X, y, self.k, self.l, self.g,
                                            A=self.A,
                                            weights=sample_weight,
                                            penalty_start=self.penalty_start,
                                            mean=self.mean)
        self.algorithm.check_compatibility(function,
                                           self.algorithm.INTERFACES)

        # TODO: Should we use a seed here so that we get deterministic results?
        beta = self.start_vector.get_vector((X.shape[1], 1))

        if self.mu is None:
            self.mu = 0.9 * function.estimate_mu(beta)
        else:
            self.mu = float(self.mu)

        function.set_params(mu=self.mu)
        self.algorithm.set_params(output=self.output)

        if self.output:
            (self.beta, self.info) = self.algorithm.run(function, beta)
        else:
            self.beta = self.algorithm.run(function, beta)

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
        try:
            self.mu = float(mu)
        except (ValueError, TypeError):
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
        X, y = check_arrays(X, y)
        function = functions.RR_SmoothedL1TV(X, y,
                                                  self.k, self.l, self.g,
                                                  Atv=self.Atv, Al1=self.Al1)

        self.algorithm.check_compatibility(function,
                                           self.algorithm.INTERFACES)

        self.algorithm.set_params(output=self.output)
        if self.output:
            (self.beta, self.info) = self.algorithm.run(function)
        else:
            self.beta = self.algorithm.run(function)

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