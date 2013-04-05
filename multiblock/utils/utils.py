# -*- coding: utf-8 -*-
"""
The :mod:`multiblock.utils` module includes common functions and constants.

Please add anything you need throughout the whole package to this module.
(As opposed to having several commong definitions scattered all over the source)

Created on Thu Feb 8 09:22:00 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD Style
"""

import numpy as _np
from numpy import dot
from numpy.linalg import norm
from numpy.random import rand
from numpy import sqrt
from copy import copy

__all__ = ['dot', 'norm', 'norm1', 'norm0', 'normI', 'make_list', 'sign',
           'cov', 'corr', 'TOLERANCE', 'MAX_ITER', 'copy', 'sstot', 'ssvar',
           'sqrt', 'rand', 'zeros', 'direct', '_DEBUG', 'debug']

_DEBUG = True

# Settings
TOLERANCE = 5e-8
MAX_ITER = 500


def norm1(x):
    return norm(x, ord=1)


def norm0(x):
    return _np.count_nonzero(_np.absolute(x))


def normI(x):
    """Identity norm. Used in e.g. prox_op when the norm of a vector should not
    be changed after division by its norm.

    Always return 1, as the norm of all vectors.
    """
    return 1


def make_list(a, n, default=None):
    # If a list, but empty
    if isinstance(a, (tuple, list)) and len(a) == 0:
        a = None
    # If only one value supplied, create a list with that value
    if a != None:
        if not isinstance(a, (tuple, list)):
#            a = [a for i in xrange(n)]
            a = [a] * n
    else:  # None or empty list supplied, create a list with the default value
#        a = [default for i in xrange(n)]
        a = [default] * n
    return a


def sign(v):
    if v < 0:
        return -1
    elif v > 0:
        return 1
    else:
        return 0


def corr(a, b):
    ma = _np.mean(a)
    mb = _np.mean(b)

    a_ = a - ma
    b_ = b - mb

    norma = norm(a_)
    normb = norm(b_)

    if norma < TOLERANCE or normb < TOLERANCE:
        return _np.zeros((1, 1))

    ip = dot(a_.T, b_)
    return ip / (norma * normb)


def cov(a, b):
    ma = _np.mean(a)
    mb = _np.mean(b)

    a_ = a - ma
    b_ = b - mb

    ip = dot(a_.T, b_)

    return ip[0, 0] / (a_.shape[0] - 1)


def sstot(a):
    a = _np.asarray(a)
    return _np.sum(a ** 2)


def ssvar(a):
    a = _np.asarray(a)
    return _np.sum(a ** 2, axis=0)


def zeros(*shape, **kwargs):
    """ Wrapper for numpy.zeros that accepts a variable number of arguments as
    the shape. I.e. zeros(2,3) and zeros((2,3)) give the same output.

    The other parameters are the same, i.e. pass dtype and/or order if you need
    to. No other arguments are processed but simply passed on to numpy.zeros.

    Caution, though, the the second and third keyword arguments must be named
    if the shape is not passed as a tuple or list, i.e. zeros(2, int) will
    raise a ValueError. Instead you will have to call zeros(2, dtype=int).
    """
    if isinstance(shape[0], (tuple, list)):
        shape = tuple(shape[0])
    return _np.zeros(shape, **kwargs)


def direct(W, T=None, P=None, compare=False):
    if compare and T == None:
        raise ValueError("In order to compare you need to supply two arrays")

    for j in xrange(W.shape[1]):
        w = W[:, [j]]
        if compare:
            t = T[:, [j]]
            cov = dot(w.T, t)
            if P != None:
                p = P[:, [j]]
                cov2 = dot(w.T, p)
        else:
            cov = dot(w.T, _np.ones(w.shape))
        if cov < 0:
            if not compare:
                w *= -1
                if T != None:
                    t = T[:, [j]]
                    t *= -1
                    T[:, j] = t.ravel()
                if P != None:
                    p = P[:, [j]]
                    p *= -1
                    P[:, j] = p.ravel()
            else:
                t = T[:, [j]]
                t *= -1
                T[:, j] = t.ravel()

            W[:, j] = w.ravel()

        if compare and P != None and cov2 < 0:
            p = P[:, [j]]
            p *= -1
            P[:, j] = p.ravel()

    if T != None and P != None:
        return W, T, P
    elif T != None and P == None:
        return W, T
    elif T == None and P != None:
        return W, P
    else:
        return W


def debug(string):
    if _DEBUG:
        print string