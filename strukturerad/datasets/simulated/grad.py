# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 12:06:07 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: TBD.
"""

import numpy as np
from utils import TOLERANCE
from utils import U
from utils import norm2

__all__ = ['grad_L1', 'grad_L2', 'grad_norm2', 'grad_TV', 'grad_TVmu']


def grad_L1(beta):

    p = beta.shape[0]

    grad = np.zeros((p, 1))
    for i in range(p):
        if beta[i, 0] > TOLERANCE:
            grad[i, 0] = 1.0
        elif beta[i, 0] < -TOLERANCE:
            grad[i, 0] = -1.0
        else:
            grad[i, 0] = U(-1, 1)

    return grad


def grad_L2(beta):

    return beta


def grad_norm2(beta):

    norm_beta = norm2(beta)
    if norm_beta > TOLERANCE:
        return beta / norm_beta
    else:
        D = beta.shape[0]
        u = (np.random.rand(D, 1) * 2.0) - 1.0  # [-1, 1]^D
        norm_u = norm2(u)
        a = np.random.rand()  # [0, 1]
        return u * (a / norm_u)


def grad_TV(beta, shape):

    p = np.prod(shape)
    D = len(shape)

    grad = np.zeros((p, 1))
    for i in range(p):
        Ai = np.zeros((D, p))
        for d in range(D):
            if i < p:
                b = np.prod([shape[-j] for j in range(1, d + 1)])
                if b + i < p:
                    Ai[d, i] = -1
                    Ai[d, b + i] = 1

        gradnorm2 = grad_norm2(np.dot(Ai, beta))
        grad += np.dot(Ai.T, gradnorm2)

    return grad


def grad_TVmu(beta, shape, mu):

    p = np.prod(shape)
    D = len(shape)

    grad = np.zeros((p, 1))
    for i in range(p):
        Ai = np.zeros((D, p))
        for d in range(D):
            if i < p:
                b = np.prod([shape[-j] for j in range(1, d + 1)])
                if b + i < p:
                    Ai[d, i] = -1
                    Ai[d, b + i] = 1

        alphai = np.dot(Ai, beta) / mu
        anorm = np.sqrt(np.sum(alphai ** 2.0))
        if anorm > 1.0:
            alphai /= anorm

        grad += np.dot(Ai.T, alphai)

    return grad