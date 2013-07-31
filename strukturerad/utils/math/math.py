# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 20:55:58 2013

@author: Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: TBD
"""

import strukturerad.utils as utils
import numpy as np

__all__ = ['norm', 'norm1', 'norm0', 'sign', 'cov', 'corr']

norm = np.linalg.norm


def norm1(x):
    return norm(x, ord=1)


def norm0(x):
    return np.count_nonzero(np.absolute(x))


def sign(v):
    if v < 0.0:
        return -1.0
    elif v > 0.0:
        return 1.0
    else:
        return 0.0


def corr(a, b):
    ma = np.mean(a)
    mb = np.mean(b)

    a_ = a - ma
    b_ = b - mb

    norma = np.sqrt(np.sum(a_ ** 2.0, axis=0))
    normb = np.sqrt(np.sum(b_ ** 2.0, axis=0))

    norma[norma < utils.TOLERANCE] = 1.0
    normb[normb < utils.TOLERANCE] = 1.0

    a_ /= norma
    b_ /= normb

    ip = np.dot(a_.T, b_)

    if ip.shape == (1, 1):
        return ip[0, 0]
    else:
        return ip


def cov(a, b):
    ma = np.mean(a)
    mb = np.mean(b)

    a_ = a - ma
    b_ = b - mb

    ip = np.dot(a_.T, b_) / (a_.shape[0] - 1.0)

    if ip.shape == (1, 1):
        return ip[0, 0]
    else:
        return ip