# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 16:44:15 2013

@author: edouard.duchesnay@cea.fr
@author: fouad.hadjselem@cea.fr
@author: vincent.guillemot@cea.fr
@author: lofstedt.tommy@gmail.com
@author: vincent.frouin@cea.fr
"""
# from sklearn.??? import LinearClassifierMixin, LinearModel

import numpy as np
from numpy.linalg import norm, eig

from .base import LinearRegressor, LinearClassifier
from structured.optimizers.ista import ISTA


def mse_l1(X, y, beta, lambd):
    return norm(y - np.dot(X, beta)) ** 2 + lambd * norm(beta, 1)


def grad_mse(X, y, beta):
    return 2 * np.dot(X.T, np.dot(X, beta) - y)


def prox_l1(beta, alpha):
    return (np.abs(beta) > alpha) * (beta - alpha * np.sign(beta - alpha))


class Lasso(LinearRegressor):
    """Lasso regression"""
    def __init__(self, lambd, optimizer="ista"):
        self.lambd = lambd
        self.optimizer = optimizer

    def fit(self, X, y):
        self.y_mean_ = np.mean(y)
        y = y - self.y_mean_
        D, V = eig(np.dot(X.T, X))
        t = 0.95 / np.max(D.real)
        if self.optimizer is "ista":
            ista = ISTA(f=mse_l1, grad_g=grad_mse, prox_h=prox_l1,
                        lambd=self.lambd)
            ista.optimize(X, y, t)
        self.iteration = ista.iteration
        self.coef_ = ista.beta


class LassoLogistic(LinearClassifier):
    def __init__(self, lambd, optimizer=None):
        self.lambd = lambd

    def fit(self, X, y):
        pass