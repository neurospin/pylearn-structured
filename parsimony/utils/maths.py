# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 20:55:58 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: TBD.
"""

import numpy as np
#import parsimony.utils as utils
from parsimony.utils.consts import TOLERANCE

__all__ = ['norm', 'norm1', 'norm0', 'normInf', 'sign', 'cov', 'corr']


def norm(x):
    return np.linalg.norm(x)


def norm1(x):
    return np.linalg.norm(x, ord=1)


def norm0(x):
    return np.count_nonzero(np.absolute(x))


def normInf(x):
    return np.linalg.norm(x, ord=float('inf'))


def sign(v):
    if v < 0.0:
        return -1.0
    elif v > 0.0:
        return 1.0
    else:
        return 0.0


def corr(a, b):
    """
    Example
    -------
    >>> import numpy as np
    >>> from parsimony.utils.maths import corr
    >>> v1 = np.asarray([[1., 2., 3.], [1., 2., 3.]])
    >>> v2 = np.asarray([[1., 2., 3.], [1., 2., 3.]])
    >>> print corr(v1, v2)
    [[ 1.  0. -1.]
     [ 0.  0.  0.]
     [-1.  0.  1.]]
    """
    ma = np.mean(a)
    mb = np.mean(b)

    a_ = a - ma
    b_ = b - mb

    norma = np.sqrt(np.sum(a_ ** 2.0, axis=0))
    normb = np.sqrt(np.sum(b_ ** 2.0, axis=0))

    norma[norma < TOLERANCE] = 1.0
    normb[normb < TOLERANCE] = 1.0

    a_ /= norma
    b_ /= normb

    ip = np.dot(a_.T, b_)

    if ip.shape == (1, 1):
        return ip[0, 0]
    else:
        return ip


def cov(a, b):
    """
    Example
    -------
    >>> import numpy as np
    >>> from parsimony.utils.maths import corr
    >>> v1 = np.asarray([[1., 2., 3.], [1., 2., 3.]])
    >>> v2 = np.asarray([[1., 2., 3.], [1., 2., 3.]])
    >>> print cov(v1, v2)
    [[ 1.  1.  1.  1.]
     [ 1.  1.  1.  1.]
     [ 1.  1.  1.  1.]
     [ 1.  1.  1.  1.]]
    """
    ma = np.mean(a)
    mb = np.mean(b)

    a_ = a - ma
    b_ = b - mb

    ip = np.dot(a_.T, b_) / (a_.shape[0] - 1.0)

    if ip.shape == (1, 1):
        return ip[0, 0]
    else:
        return ip

if __name__ == "__main__":
    import doctest
    doctest.testmod()