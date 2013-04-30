# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:35:26 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD Style
"""

__all__ = ['BaseStartVector', 'RandomStartVector', 'OnesStartVector',
           'LargestStartVector', 'GaussianCurveVector']

import abc
import numpy as np
import numpy.linalg as la
import math
from multiblock.utils import norm, TOLERANCE


class BaseStartVector(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, normalise=True):
        super(BaseStartVector, self).__init__()

        self.normalise = normalise

    @abc.abstractmethod
    def get_vector(self, X=None, shape=None):
        raise NotImplementedError('Abstract method "getVector" must be '\
                                  'specialised!')


class IdentityStartVector(BaseStartVector):
    def __init__(self, vector):
        super(IdentityStartVector, self).__init__()

        self.vector = vector

    def get_vector(self, *args, **kwargs):
        return self.vector


class RandomStartVector(BaseStartVector):
    def __init__(self, normalise=True):
        super(RandomStartVector, self).__init__(normalise=normalise)

    def get_vector(self, X=None, shape=None):
        if X == None and shape == None:
            raise ValueError('Either a matrix X or the shape must be given')
        if X != None:
            shape = (X.shape[1], 1)

        w = np.random.rand(*shape)  # Random start vector

        if self.normalise:
            return w / norm(w)
        else:
            return w


class OnesStartVector(BaseStartVector):
    def __init__(self, normalise=True):
        super(OnesStartVector, self).__init__(normalise=normalise)

    def get_vector(self, X=None, shape=None):
        if X == None and shape == None:
            raise ValueError('Either a matrix X or the shape must be given')
        if X != None:
            shape = (X.shape[1], 1)

        w = np.ones(shape)  # Using a vector of ones

        if self.normalise:
            return w / norm(w)
        else:
            return w


#class ZerosStartVector(BaseStartVector):
#    def __init__(self):
#        super(ZerosStartVector, self).__init__()
#
#    def get_vector(self, X=None, shape=None):
#        if X == None and shape == None:
#            raise ValueError('Either a matrix X or the shape must be given')
#        if X != None:
#            shape = (X.shape[1], 1)
#
#        w = np.zeros(shape)  # Using a vector of zeros
#
##        if self.normalise:
##            return w / norm(w)
##        else:
#        return w


class LargestStartVector(BaseStartVector):

    def __init__(self, axis=1, normalise=True):
#        BaseStartVector.__init__(self, size=None, normalise=normalise)
        super(LargestStartVector, self).__init__(normalise=normalise)
        self.axis = axis

    def get_vector(self, X):
        idx = np.argmax(np.sum(X ** 2, axis=self.axis))
        if self.axis == 0:
            w = X[:, [idx]]  # Using column with largest sum of squares
        else:
            w = X[[idx], :].T  # Using row with largest sum of squares

        if self.normalise:
            return w / norm(w)
        else:
            return w


class GaussianCurveVector(BaseStartVector):

    def __init__(self, normalise=True):

        super(GaussianCurveVector, self).__init__(normalise=normalise)

    def get_vector(self, shape, mean=None, cov=None):
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