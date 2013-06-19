# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 13:56:24 2013

@author: Tommy Löfstedt
@email: tommy.loefstedt@cea.fr
"""
__all__ = ['ConstantCorrelation']

import numpy as np


def ConstantCorrelation(p=[100], rho=[0.05], delta=0.10, eps=0.01):
    """ Returns a positive definite matrix, corresponding to a block covariance
    matrix. Each block has the structure:

              [1, ..., rho_k]
        S_k = [...,  1,  ...],
              [rho_k, ..., 1]

    i.e. 1 on the diagonal and rho_k (on average) outside the diagonal. S then
    has the structure:

            [S_1, delta, delta]
        S = [delta, S_i, delta],
            [delta, delta, S_N]

    i.e. with the groups-correlation matrices on the diagonal and delta (on
    average) outside.

    Parameters
    ----------
    p    : A scalar or a list of scalars with the numbers of variables for each
           group.

    rho  : A scalar or a list of the average correlation between off-diagonal
           elements of S.

    delta: Baseline noise between groups. (Percent of rho.) Only used if K > 1.

    eps  : Maximum entry-wise random noise. (Pecent of 1-rho.)

    Returns
    -------
    S    : The correlation matrix.
    """
    if not isinstance(rho, (list, tuple)):
        p = [p]
        rho = [rho]

    K = len(rho)

    M = 10  # Dim. of noise space. Higher is slower but may improve randomness?
    N = 0
    rho_min = min(rho)
    rho_max = max(rho)
    eps = eps * (1.0 - rho_max)
    delta = delta * rho_min

    for k in xrange(K):
        N += p[k]

    u = (np.random.rand(M, N) - 0.5) * 2.0  # [-1,1]
    u /= np.sqrt(np.sum(u ** 2.0, axis=0))  # Normailse
    uu = np.dot(u.T, u)

    S = delta + eps * uu

    Nk = 0
    for k in xrange(K):
        pk = p[k]
        Nk += pk

        uuk = uu[Nk - pk:Nk, Nk - pk:Nk]
        Sk = rho[k] + eps * uuk
        Sk -= np.diag(np.diag(Sk)) - np.eye(*Sk.shape)

        S[Nk - pk:Nk, Nk - pk:Nk] = Sk

#    k = (N * (1.0 + eps) + 1) / (1.0 - rho_max - eps)
#    print "cond(S)  = ", np.linalg.cond(S)
#    print "cond(S) <= ", k

    return S


def ToeplitzCorrelation(p=[100], rho=[0.05], delta=0.10, eps=0.01):
    """ Returns a positive definite matrix, corresponding to a block covariance
    matrix. Each block has the structure:

              [1, ..., rho_k]
        S_k = [...,  1,  ...],
              [rho_k, ..., 1]

    i.e. 1 on the diagonal and rho_k (on average) outside the diagonal. S then
    has the structure:

            [S_1, delta, delta]
        S = [delta, S_i, delta],
            [delta, delta, S_N]

    i.e. with the groups-correlation matrices on the diagonal and delta (on
    average) outside.

    Parameters
    ----------
    p    : A scalar or a list of scalars with the numbers of variables for each
           group.

    rho  : A scalar or a list of the average correlation between off-diagonal
           elements of S.

    delta: Baseline noise between groups. (Percent of rho.) Only used if K > 1.

    eps  : Maximum entry-wise random noise. (Pecent of 1-rho.)

    Returns
    -------
    S    : The correlation matrix.
    """
    if not isinstance(rho, (list, tuple)):
        p = [p]
        rho = [rho]

    K = len(rho)

    M = 10  # Dim. of noise space. Higher is slower but may improve randomness?
    N = sum(p)
    rho_max = max(rho)
    eps = eps * (1.0 - rho_max) / (1.0 + rho_max)

    u = (np.random.rand(M, N) - 0.5) * 2.0  # [-1,1]
    u /= np.sqrt(np.sum(u ** 2.0, axis=0))  # Normailse
    uu = np.dot(u.T, u)

    Sigma = np.zeros(N, N)
    for k in xrange(K):
        pk = p[k]
        rhok = rho[k]
        v = [0] * (pk - 1)
        for i in xrange(0, pk - 1):
            v[i] = rhok ** i

        Sigmak = np.eye(pk, pk)
        for i in xrange(0, pk):
            Sigmak[i]

        uuk = uu[Nk - pk:Nk, Nk - pk:Nk]
        Sk = rho[k] + eps * uuk
        Sk -= np.diag(np.diag(Sk)) - np.eye(*Sk.shape)

        S[Nk - pk:Nk, Nk - pk:Nk] = Sk

    S = Sigma + eps * (uu - np.eye(*uu.shape))

#    k = (N * (1.0 + eps) + 1) / (1.0 - rho_max - eps)
#    print "cond(S)  = ", np.linalg.cond(S)
#    print "cond(S) <= ", k

    return S