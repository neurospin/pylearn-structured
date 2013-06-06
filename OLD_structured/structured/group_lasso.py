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

from .base import LinearRegressor, LinearClassifier

class GLasso(LinearRegressor):
    
    def __init__(self, lambd, optimizer=None):
        self.lambd = lambd

    def fit(self, X, y):
        pass

class GLassoLogistic(LinearClassifier):
    
    def __init__(self, lambd, optimizer=None):
        self.lambd = lambd

    def fit(self, X, y):
        pass
