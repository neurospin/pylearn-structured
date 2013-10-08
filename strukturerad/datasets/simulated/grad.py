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

__all__ = ['grad_L1', 'grad_L1mu', 'grad_L2', 'grad_norm2',
           'grad_TV', 'grad_TVmu']


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


def grad_L1mu(beta, mu):

    alpha = (1.0 / mu) * beta
    asnorm = np.abs(alpha)
    i = asnorm > 1.0
    alpha[i] = np.divide(alpha[i], asnorm[i])

    return alpha


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


def _generate_A(shape):

    D = len(shape)
    p = np.prod(shape)
    dims = shape + (1,)
    A = [0] * D
    for i in xrange(D - 1, -1, -1):
        shift = np.prod(dims[i + 1:])
        A[D - i - 1] = np.eye(p, p, shift) - np.eye(p, p)

    # TODO: Only works for up to 3 dimensions ...
    ind = np.reshape(xrange(p), shape)
    xind = ind[:, :, -1].flatten().tolist()
    yind = ind[:, -1, :].flatten().tolist()
    zind = ind[-1, :, :].flatten().tolist()

    for i in xind:
        A[0][i, :] = 0
    for i in yind:
        A[1][i, :] = 0
    for i in zind:
        A[2][i, :] = 0

    return A


def _generate_Ai(i, A, shape):

    D = len(shape)
    v = []
    for k in xrange(D):
        v.append(A[k][i, :])

    Ai = np.vstack(v)

#    Ai = np.zeros((D, p))
#    for d in xrange(D - 1, -1, -1):
#        if d == D - 1:
#            x = i % shape[d]
#            if x + 1 < shape[d]:
#                Ai[D - d - 1, i] = -1
#                Ai[D - d - 1, i + 1] = 1
#        else:
#            b = np.prod(shape[d + 1:])
#            y = int(i / b)
#            if y + 1 < shape[d]:
#                Ai[D - d - 1, i] = -1
#                Ai[D - d - 1, i + b] = 1

    return Ai


def grad_TV(beta, shape):

    p = np.prod(shape)
#    D = len(shape)

    A = _generate_A(shape)

    grad = np.zeros((p, 1))
    for i in range(p):
#        Ai = np.zeros((D, p))
#        for d in range(D):
#            if i < p:
#                b = np.prod([shape[-j] for j in range(1, d + 1)])
#                if b + i < p:
#                    Ai[d, i] = -1
#                    Ai[d, b + i] = 1

        Ai = _generate_Ai(i, A, shape)

#        print "i:", i, "Ai:\n", np.reshape(Ai[0, :] + Ai[1, :] + Ai[2, :],
#                                           shape)
#        print "i:", i, "Ai:\n", np.reshape(Ai[0, :], shape)
#        print np.reshape(Ai[1, :], shape)
#        print np.reshape(Ai[2, :], shape)

        gradnorm2 = grad_norm2(np.dot(Ai, beta))
        grad += np.dot(Ai.T, gradnorm2)

    return grad


def grad_TVmu(beta, shape, mu):

    p = np.prod(shape)
#    D = len(shape)

    A = _generate_A(shape)

    grad = np.zeros((p, 1))
    for i in range(p):

        Ai = _generate_Ai(i, A, shape)

        alphai = np.dot(Ai, beta) / mu
        anorm = np.sqrt(np.sum(alphai ** 2.0))
        if anorm > 1.0:
            alphai /= anorm

        grad += np.dot(Ai.T, alphai)

    return grad