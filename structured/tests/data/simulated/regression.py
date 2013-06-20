# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 09:22:40 2013

@author: Tommy Löfstedt
@email: tommy.loefstedt@cea.fr
"""

import correlation_matrices

__all__ = ['load', 'labels']

import numpy as np


def load(size=[100, 100], rho=0.05, delta=0.1, eps=0.01, locally_smooth=False):

    if not isinstance(rho, (list, tuple)):
        size = [size]
        rho = [rho]

    K = len(rho)

    p = [0] * K
    n = None
    for k in xrange(K):
        if n != None and size[k][0] != n:
            raise ValueError("The groups must have the same number of samples")
        n = size[k][0]
        pk = size[k][1]
        p[k] = pk

    if locally_smooth:
        S = correlation_matrices.ToeplitzCorrelation(p, rho, eps)
    else:
        S = correlation_matrices.ConstantCorrelation(p, rho, delta, eps)

    mean = np.zeros(sum(p))
    X = np.random.multivariate_normal(mean, S, n)

    return X, S


def labels():
    pass