# -*- coding: utf-8 -*-
"""
The :mod:`multiblock.utils` module includes common functions and constants.

Please add anything you need throughout the whole package to this module.
(As opposed to having several commong definitions scattered all over the
source).

Created on Thu Feb 8 09:22:00 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD Style
"""

import scipy
import numpy as np
from numpy.linalg import norm
from numpy.random import rand
from numpy import sqrt
from copy import copy
import traceback

__all__ = ['norm', 'norm1', 'norm0', 'normI', 'make_list', 'sign',
           'cov', 'corr', 'TOLERANCE', 'MAX_ITER', 'copy', 'sstot', 'ssvar',
           'sqrt', 'rand', 'zeros', 'direct', '_DEBUG', 'debug', 'warning',
           'optimal_shrinkage', 'delete_sparse_csr_row', 'AnonymousClass']

_DEBUG = True

# Settings
TOLERANCE = 5e-8
MAX_ITER = 1000


def norm1(x):
    return norm(x, ord=1)


def norm0(x):
    return np.count_nonzero(np.absolute(x))


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
    ma = np.mean(a)
    mb = np.mean(b)

    a_ = a - ma
    b_ = b - mb

    norma = norm(a_)
    normb = norm(b_)

    if norma < TOLERANCE or normb < TOLERANCE:
        return np.zeros((1, 1))

    ip = np.dot(a_.T, b_)

    return ip[0, 0] / (norma * normb)


def cov(a, b):
    ma = np.mean(a)
    mb = np.mean(b)

    a_ = a - ma
    b_ = b - mb

    ip = np.dot(a_.T, b_)

    return ip[0, 0] / (a_.shape[0] - 1.0)


def sstot(a):
    a = np.asarray(a)
    return np.sum(a ** 2)


def ssvar(a):
    a = np.asarray(a)
    return np.sum(a ** 2, axis=0)


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
    return np.zeros(shape, **kwargs)


def direct(W, T=None, P=None, compare=False):
    if compare and T == None:
        raise ValueError("In order to compare you need to supply two arrays")

    for j in xrange(W.shape[1]):
        w = W[:, [j]]
        if compare:
            t = T[:, [j]]
            cov = np.dot(w.T, t)
            if P != None:
                p = P[:, [j]]
                cov2 = np.dot(w.T, p)
        else:
            cov = np.dot(w.T, np.ones(w.shape))
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


def debug(*args):
    if _DEBUG:
        s = ""
        for a in args:
            s = s + str(a)
        print s


def warning(*args):
    if _DEBUG:
        s = ""
        for a in args:
            s = s + str(a)
#        traceback.print_stack()
        print "WARNING:", s


def optimal_shrinkage(*X, **kwargs):

    tau = []

    T = kwargs.pop('T', None)

    if T == None:
        T = [T] * len(X)
    if len(X) != len(T):
        if T == None:
            T = [T] * len(X)
        else:
            T = [T[0]] * len(X)

    for i in xrange(len(X)):
        Xi = X[i]
        Ti = T[i]

        M, N = Xi.shape
        Si = np.cov(Xi.T)
        if Ti == None:
            Ti = np.diag(np.diag(Si))

#        R = _np.corrcoef(X.T)
        Wm = Si * ((M - 1.0) / M)  # 1 / N instead of 1 / N - 1

        Var_sij = 0
        for i in xrange(N):
            for j in xrange(N):
                wij = np.multiply(Xi[:, [i]], Xi[:, [j]]) - Wm[i, j]
                Var_sij += np.dot(wij.T, wij)
        Var_sij = Var_sij[0, 0] * (M / ((M - 1.0) ** 3.0))

#        diag = _np.diag(C)
#        SS_sij = _np.sum((C - _np.diag(diag)) ** 2.0)
#        SS_sij += _np.sum((diag - 1.0) ** 2.0)

        d = (Ti - Si) ** 2.0

        l = Var_sij / np.sum(d)
        l = max(0, min(1, l))

        tau.append(l)

    return tau


def delete_sparse_csr_row(mat, i):
    """Delete row i in-place from sparse matrix mat (CSR format).

    Implementation from:

        http://stackoverflow.com/questions/13077527/is-there-a-numpy-delete-equivalent-for-sparse-matrices
    """
    if not isinstance(mat, scipy.sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    n = mat.indptr[i + 1] - mat.indptr[i]
    if n > 0:
        mat.data[mat.indptr[i]:-n] = mat.data[mat.indptr[i + 1]:]
        mat.data = mat.data[:-n]
        mat.indices[mat.indptr[i]:-n] = mat.indices[mat.indptr[i + 1]:]
        mat.indices = mat.indices[:-n]
    mat.indptr[i:-1] = mat.indptr[i + 1:]
    mat.indptr[i:] -= n
    mat.indptr = mat.indptr[:-1]
    mat._shape = (mat._shape[0] - 1, mat._shape[1])


class AnonymousClass:
    """Used to create anonymous classes.

    Usage: anonymous_class = AnonymousClass(field=value, method=function)
    """
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __neq__(self, other):
        return self.__dict__ != other.__dict__


#class Enum(object):
#    def __init__(self, *sequential, **named):
#        enums = dict(zip(sequential, range(len(sequential))), **named)
#        for k, v in enums.items():
#            setattr(self, k, v)
#
#    def __setattr__(self, name, value): # Read-only
#        raise TypeError("Enum attributes are read-only.")
#
#    def __str__(self):
#        return "Enum: "+str(self.__dict__)