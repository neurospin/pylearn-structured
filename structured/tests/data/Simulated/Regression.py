# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 09:22:40 2013

@author: Tommy Löfstedt
@email: tommy.loefstedt@cea.fr
"""

__all__ = ['load', 'labels']

import numpy as np


def load(size=[200, 200], rho=0.05, eps=0.1, delta=0.01, K=1):

    if K == 1:
        size = [size]
        rho = [rho]

    M = 100  # Dim. of noise space. Higher is slower but may improve randomness
    N = 0
    rho_min = min(rho)
    rho_max = max(rho)
    eps = eps * (1.0 - rho_max)
    delta = delta * rho_min
    print eps
    print delta
    print delta + eps * 0.5

    for k in xrange(K):
        N += size[k][1]  # g_k

    u = np.random.rand(M, N)
    u /= np.sqrt(np.sum(u ** 2.0, axis=0))
    uu = np.dot(u.T, u)

    print np.mean(uu)

    S = delta + eps * uu

    Nk = 0
    for k in xrange(K):
        n = size[k][0]
        p = size[k][1]  # g_k
        Nk += p

#        uk = u[:, Nk - p:Nk]
#        uuk = np.dot(uk.T, uk)
        uuk = uu[Nk - p:Nk, Nk - p:Nk]
        Sk = rho[k] + eps * uuk
        Sk -= np.diag(np.diag(Sk)) - np.eye(*Sk.shape)

        S[Nk - p:Nk, Nk - p:Nk] = Sk

    k = (N * (1.0 + eps) + 1) / (1.0 - rho_max - eps)

    mean = np.zeros(N)

    X = np.random.multivariate_normal(mean, S, n)

    return X, S


def labels():
    pass