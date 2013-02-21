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
from copy import copy

__all__ = ['dot', 'norm', 'norm1', 'norm0', 'make_list', 'sign',
           'cov', 'corr', 'TOLERANCE', 'MAX_ITER', 'copy']

# Settings
TOLERANCE = 5e-8
MAX_ITER  = 500

def norm1(x):
    return norm(x, ord = 1)

def norm0(x):
    return _np.count_nonzero(_np.absolute(x))

def make_list(a, n, default = None):
    # If a list, but empty
    if isinstance(a, (tuple, list)) and len(a) == 0:
        a = None
    # If only one value supplied, create a list with that value
    if a != None:
        if not isinstance(a, (tuple, list)):
            a = [a for i in xrange(n)]
    else: # None or empty list supplied, create a list with the default value
        a = [default for i in xrange(n)]
    return a

def sign(v):
    if v < 0:
        return -1
    elif v > 0:
        return 1
    else:
        return 0

def corr(a,b):
    ma = _np.mean(a)
    mb = _np.mean(b)

    a_ = a - ma
    b_ = b - mb

    norma = norm(a_)
    normb = norm(b_)

    if norma < TOLERANCE or normb < TOLERANCE:
        return 0

    ip = dot(a_.T, b_)
    return ip / (norma * normb)

def cov(a,b):
    ma = _np.mean(a)
    mb = _np.mean(b)

    a_ = a - ma
    b_ = b - mb

    ip = dot(a_.T, b_)

    return ip[0,0] / (a_.shape[0] - 1)
