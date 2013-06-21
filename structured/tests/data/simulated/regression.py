# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 09:22:40 2013

@author: Tommy Löfstedt
@email: tommy.loefstedt@cea.fr
"""

import correlation_matrices

__all__ = ['load', 'labels']

import numpy as np
import random


def load(size=[100, 100], rho=0.05, delta=0.1, eps=None, sparsity=0.5,
         snr=100.0, locally_smooth=False):

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

    if eps == None:
        eps = np.sqrt(10.0 / float(n))  # Set variance to 1 / n.

    if locally_smooth:
        S = correlation_matrices.ToeplitzCorrelation(p, rho, eps)
    else:
        S = correlation_matrices.ConstantCorrelation(p, rho, delta, eps)

    p = sum(p)

    # Create X matrix using the generated correlation matrix
    mean = np.zeros(p)
    X = np.random.multivariate_normal(mean, S, n)

    # Enforce sparsity
    b = (np.random.rand(p, 1) - 0.5) * 2.0
    ind = range(p)
    random.shuffle(ind)
    ind = ind[:int(round(len(ind) * sparsity))]
    b[ind] = 0

    # Compute pure y
    y = np.dot(X, b)

    # Add noise from N(0, (1/snr)*||Xb||² / (n-1))
    var = (np.sum(y ** 2.0) / float(n - 1)) / float(snr)
    e = np.random.randn(p, 1)
    e *= np.sqrt(var)

    y += e

    return X, y, b, e


def labels():
    pass


if __name__ == "__main__":

    n = 100
    p = 100
    # Var(S) ~= (eps * (1 - max(rho))) ** 2.0 / 10
    # Var(uu) = 1 / n => eps = np.sqrt(10.0 / n)
#    X, S = load(size=[10, 10], rho=0.0, delta=0.0, eps=np.sqrt(10.0 / n))
#    XX = np.dot(X.T, X) / (float(n) - 1.0)
    X, S = load()
    XX = np.dot(X.T, X) / (float(n) - 1.0)