# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:35:26 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD Style
"""

__all__ = ['BaseStartVector', 'RandomStartVector', 'OnesStartVector',
           'ZerosStartVector', 'LargestStartVector', 'GaussianCurveVector']

import abc
import numpy as np
import numpy.linalg as la
from multiblock.utils import norm


class BaseStartVector(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, shape=None, normalise=True):
        super(BaseStartVector, self).__init__()

        self.shape = shape
        self.normalise = normalise

    @abc.abstractmethod
    def get_vector(self, X=None, shape=None):
        raise NotImplementedError('Abstract method "getVector" must be '\
                                  'specialised!')


class IdentityStartVector(BaseStartVector):
    def __init__(self, vector, **kwargs):
        super(IdentityStartVector, self).__init__(**kwargs)

        self.vector = vector

    def get_vector(self, *args, **kwargs):
        return self.vector


class RandomStartVector(BaseStartVector):
    def __init__(self, normalise=True, **kwargs):
        super(RandomStartVector, self).__init__(normalise=normalise, **kwargs)

    def get_vector(self, X=None, shape=None, axis=1):
        if X == None and shape == None:
            raise ValueError('A matrix X or a shape must be must be given.')
        if X != None:
            shape = (X.shape[axis], 1)

        w = np.random.rand(*shape)  # Random start vector

        if self.normalise:
            return w / norm(w)
        else:
            return w


class OnesStartVector(BaseStartVector):
    def __init__(self, normalise=True, **kwargs):
        super(OnesStartVector, self).__init__(normalise=normalise, **kwargs)

    def get_vector(self, X=None, shape=None, axis=1):
        if X == None and shape == None:
            raise ValueError('A matrix X or a shape must be must be given.')
        if X != None:
            shape = (X.shape[axis], 1)

        w = np.ones(shape)  # Using a vector of ones

        if self.normalise:
            return w / norm(w)
        else:
            return w


class ZerosStartVector(BaseStartVector):
    """A start vector of zeros.

    Use with care! Be aware that using this in algorithms that are not aware
    may result in division by zero since the norm of this start vector is 0.
    """
    def __init__(self, **kwargs):
        super(ZerosStartVector, self).__init__(normalise=False, **kwargs)

    def get_vector(self, X=None, shape=None, axis=1):
        if X == None and shape == None:
            raise ValueError('A matrix X or a shape must be must be given.')
        if X != None:
            shape = (X.shape[axis], 1)

        w = np.zeros(shape)  # Using a vector of zeros

        return w


class LargestStartVector(BaseStartVector):

    def __init__(self, normalise=True, **kwargs):
        super(LargestStartVector, self).__init__(normalise=normalise, **kwargs)

    def get_vector(self, X, axis=1):
        if X == None:
            raise ValueError('A matrix X must be must be given.')

        idx = np.argmax(np.sum(X ** 2.0, axis=axis))
        if axis == 0:
            w = X[:, [idx]]  # Using column with largest sum of squares
        else:
            w = X[[idx], :].T  # Using row with largest sum of squares

        if self.normalise:
            return w / norm(w)
        else:
            return w


class GaussianCurveVector(BaseStartVector):

    def __init__(self, normalise=True, **kwargs):
        super(GaussianCurveVector, self).__init__(normalise=normalise,
                                                  **kwargs)

    def get_vector(self, X=None, shape=None, mean=None, cov=None, axis=1):
        if X == None and shape == None:
            raise ValueError('A matrix X or a shape must be must be given.')
        if X != None:
            shape = (X.shape[axis], 1)
        if not isinstance(shape, tuple):
            shape = tuple(shape)

        n = len(shape)
        if mean == None:
            mean = [float(s) / 2.0 for s in shape]
        if cov == None:
            S = np.eye(n) / 4.0
#            S = np.diag([s / 2.0 for s in shape])
            invS = S
#            detS = 1
        else:
            S = np.diag(np.diag(cov))
            invS = la.pinv(S)
#            detS = la.det(S)
#            if detS < TOLERANCE:
#                detS = TOLERANCE

#        k = 1.0 / math.sqrt(detS * ((2.0 * math.pi) ** n))

        s = []
        X = 0
        for i in xrange(n):
            x = np.arange(shape[i]) - mean[i]
            X = X + invS[i, i] * (np.reshape(x, [shape[i]] + s) ** 2)
            s.append(1)

        X = np.exp(-0.5 * X)
        X /= np.sum(X)

        print invS
        print np.max(X)

        return X

#        G = np.exp(((x-x0)**2 + (y-y0)**2))

#        if self.normalise:
#            return w / norm(w)
#        else:
#            return w