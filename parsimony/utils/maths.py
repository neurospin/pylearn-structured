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

__all__ = ['norm', 'norm1', 'norm0', 'norm_inf', 'sign', 'corr']


def norm(x):
    '''Return the 2-norm for vectors, the Frobenius norm for matrices

    Examples
    --------
    >>> from parsimony.utils.maths import norm
    >>> matrix = np.array([[0.2, 1.0, 0.4], [2.0, 1.5, 0.1]])
    >>> norm(matrix)
    2.7313000567495327
    >>> vector = np.array([[0.2], [1.0], [0.4]])
    >>> norm(vector)
    1.0954451150103324
    '''
    return np.linalg.norm(x)


def norm1(x):
    '''Return the 1-norm

    For vectors : sum(abs(x)**2)**(1./2)
    For matrices : max(sum(abs(x), axis=0))

    Examples
    --------
    >>> from parsimony.utils.maths import norm1
    >>> matrix = np.array([[0.2, 1.0, 0.4], [2.0, 1.5, 0.1]])
    >>> norm1(matrix)
    2.5
    >>> vector = np.array([[0.2], [1.0], [0.4]])
    >>> norm1(vector)
    1.6000000000000001
    '''
    return np.linalg.norm(x, ord=1)


def norm0(x):
    '''Return the number of non-zero elements

    Examples
    --------
    >>> from parsimony.utils.maths import norm0
    >>> matrix = np.array([[0.2, 1.0, 0.4], [2.0, 1.5, 0.1]])
    >>> norm0(matrix)
    6
    >>> vector = np.array([[0.2], [1.0], [0.4]])
    >>> norm0(vector)
    3
    '''
    return np.count_nonzero(np.absolute(x))


def norm_inf(x):
    '''Return the max of the absolute sum for each column of the matrix

    For vectors : max(abs(x))
    For matrices : max(sum(abs(x), axis=1))

    Examples
    --------
    >>> from parsimony.utils.maths import norm_inf
    >>> matrix = np.array([[0.2, 1.0, 0.4], [2.0, 1.5, 0.1]])
    >>> norm_inf(matrix)
    3.6000000000000001
    >>> vector = np.array([[0.2], [1.0], [0.4]])
    >>> norm_inf(vector)
    1.0
    '''
    return np.linalg.norm(x, ord=float('inf'))

# TODO Remove this function and use np.sign
def sign(v):
    """Return the sign of v, or 0 if v is null
    """
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

# TODO: remove commented code
#def cov(a, b):
#    """
#    Example
#    -------
#    >>> import numpy as np
#    >>> from parsimony.utils.maths import cov
#    >>> v1 = np.asarray([[1., 2., 3.], [1., 2., 3.]])
#    >>> v2 = np.asarray([[1., 2., 3.], [1., 2., 3.]])
#    >>> print cov(v1, v2)
#    [[ 1.  1.  1.  1.]
#     [ 1.  1.  1.  1.]
#     [ 1.  1.  1.  1.]
#     [ 1.  1.  1.  1.]]
#    """
#    ma = np.mean(a)
#    mb = np.mean(b)
#
#    a_ = a - ma
#    b_ = b - mb
#
#    ip = np.dot(a_.T, b_) / (a_.shape[0] - 1.0)
#
#    if ip.shape == (1, 1):
#        return ip[0, 0]
#    else:
#        return ip

if __name__ == "__main__":
    import doctest
    doctest.testmod()