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
from utils import norm


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
    """A start vector with the shape of a Gaussian curve.

    The gaussian is computed with respect to the numbers of dimension in a
    supposed image. The output is thus a reshaped vector corresponsing to a 1-,
    2-, 3- or higher-dimensional Gaussian curve.
    """

    def __init__(self, normalise=True, **kwargs):
        super(GaussianCurveVector, self).__init__(normalise=normalise,
                                                  **kwargs)

    def get_vector(self, X=None, shape=None, size=None, mean=None, cov=None,
                   axis=1, dims=2):
        """ Computes a Gaussian curve-shaped starting vector.

        Parameters:
        X     : The matrix for which we need a start vector. Used in
                conjunction with axis to determine the shape of the start
                vector.

        shape : The shape of the start vector.

        size  : The size of the supposed image. Must have the form (Z, Y, X).

        mean  : The mean vector of the Gaussian. Default is zero.

        cov   : The covariance matrix of the Gaussian. Default is identity.

        axis  : The axis along X which the shape is taken.

        dims  : The number of dimensions of the output image. Default is 2.
        """
        if size != None:
            p = 1
            for i in xrange(dims):
                p *= size[i]
            if axis == 1:
                shape = (p, 1)
            else:
                shape = (1, p)
        else:
            if X != None:
                p = X.shape[axis]
                shape = (p, 1)
            else:  # Assumes shape != None
                p = shape[0] * shape[1]

            size = [0] * dims
            for i in xrange(dims):  # Split in equal-sized hypercube
                size[i] = round(float(p) ** (1.0 / float(dims)))

        if mean == None:
            mean = [float(s - 1.0) / 2.0 for s in size]
        if cov == None:
            S = np.diag([s ** (1.0 / dims) for s in size])
            invS = np.linalg.pinv(S)
        else:
#            S = np.diag(np.diag(cov))
            S = np.asarray(cov)
            invS = np.linalg.pinv(S)

        a = np.arange(size[0])
        ans = np.reshape(a, (a.shape[0], 1)).tolist()
        for i in xrange(1, dims):
            b = np.arange(size[i]).tolist()
            ans = [y + [x] for x in b for y in ans]

        X = np.zeros((size))
        for x in ans:
            i = tuple(x)
            x = np.array([x]) - np.array(mean)
            v = np.dot(x, np.dot(invS, x.T))
            X[i] = v[0, 0]

        X = np.exp(-0.5 * X)
        X /= np.sum(X)

#        s = []
#        X = 0
#        for i in xrange(dims):
#            x = np.arange(size[i]) - mean[i]
#            x = np.reshape(x, [size[i]] + s)
#            X = X + invS[i, i] * (x ** 2.0)
#            s.append(1)

        w = np.reshape(X, (p, 1))

        if self.normalise:
            return w / norm(w)
        else:
            return w