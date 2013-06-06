# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:59:03 2013

@author: edouard.duchesnay@cea.fr
@author: fouad.hadjselem@cea.fr
@author: vincent.guillemot@cea.fr
@author: lofstedt.tommy@gmail.com
@author: vincent.frouin@cea.fr
"""
import numpy as np
from abc import abstractmethod
from sklearn.metrics import accuracy_score, r2_score

class ClassifierMixin(object):
    """Mixin class for all classifiers in scikit-learn"""

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training set.

        y : array-like, shape = [n_samples]
            Labels for X.

        Returns
        -------
        z : float

        """

        return accuracy_score(y, self.predict(X))


class RegressorMixin(object):
    """Mixin class for all regression estimators in scikit-learn"""

    def score(self, X, y):
        """Returns the coefficient of determination R^2 of the prediction.

        The coefficient R^2 is defined as (1 - u/v), where u is the regression
        sum of squares ((y_true - y_pred) ** 2).sum() and v is the residual
        sum of squares ((y_true - y_true.mean()) ** 2).sum().
        Best possible score is 1.0, lower values are worse.


        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training set.

        y : array-like, shape = [n_samples]

        Returns
        -------
        z : float
        """

        return r2_score(y, self.predict(X))


class LinearRegressor(RegressorMixin):

    def predict(self, X):
        """Predict using the linear model

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples]
            Returns predicted values.
        """
        return np.dot(X, self.coef_.T) + self.y_mean_


class LinearClassifier(RegressorMixin):

    def predict(self, X):
        """Predict using the linear model

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]

        Returns
        -------
        C : array, shape = [n_samples]
            Returns predicted values.
        """
        raise ValueError("Function not implemented")


