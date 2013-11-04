# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 15:35:26 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: TBD.
"""

import abc
import numpy as np

import parsimony.utils as utils

__all__ = ['BaseStartVector', 'IdentityStartVector', 'RandomStartVector',
           'OnesStartVector', 'ZerosStartVector']


class BaseStartVector(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, normalise=True):

        super(BaseStartVector, self).__init__()

        self.normalise = normalise

    @abc.abstractmethod
    def get_vector(self, shape):

        raise NotImplementedError('Abstract method "get_vector" must be '\
                                  'specialised!')


class IdentityStartVector(BaseStartVector):
    """A pre-determined start vector.
    """
    def __init__(self, vector, **kwargs):

        super(IdentityStartVector, self).__init__(**kwargs)

        self.vector = vector

    def get_vector(self, *args, **kwargs):

        return self.vector


class RandomStartVector(BaseStartVector):
    """A start vector of uniformly distributed random values.
    """
    def __init__(self, **kwargs):

        super(RandomStartVector, self).__init__(**kwargs)

    def get_vector(self, shape):

        vector = np.random.rand(*shape)  # Random start vector

        if self.normalise:
            return vector / utils.math.norm(vector)
        else:
            return vector


class OnesStartVector(BaseStartVector):
    """A start vector of zeros.
    """
    def __init__(self, **kwargs):

        super(OnesStartVector, self).__init__(**kwargs)

    def get_vector(self, shape):

        vector = np.ones(shape)  # Using a vector of ones.

        if self.normalise:
            return vector / utils.math.norm(vector)
        else:
            return vector


class ZerosStartVector(BaseStartVector):
    """A start vector of zeros.

    Use with care! Be aware that using this in algorithms that are not aware
    may result in division by zero since the norm of this start vector is 0.
    """
    def __init__(self, **kwargs):

        kwargs.pop('normalise', False)  # We do not care about this argument.

        super(ZerosStartVector, self).__init__(normalise=False, **kwargs)

    def get_vector(self, shape):

        w = np.zeros(shape)  # Using a vector of zeros.

        return w


#class LargestStartVector(BaseStartVector):
#
#    def __init__(self, normalise=True, **kwargs):
#
#        super(LargestStartVector, self).__init__(normalise=normalise, **kwargs)
#
#    def get_vector(self, X, axis=1):
#        if X == None:
#            raise ValueError('A matrix X must be must be given.')
#
#        idx = np.argmax(np.sum(X ** 2.0, axis=axis))
#        if axis == 0:
#            w = X[:, [idx]]  # Using column with largest sum of squares
#        else:
#            w = X[[idx], :].T  # Using row with largest sum of squares
#
#        if self.normalise:
#            return w / norm(w)
#        else:
#            return w


#class GaussianCurveVector(BaseStartVector):
#    """A start vector with the shape of a Gaussian curve.
#
#    The gaussian is computed with respect to the numbers of dimension in a
#    supposed image. The output is thus a reshaped vector corresponsing to a 1-,
#    2-, 3- or higher-dimensional Gaussian curve.
#    """
#
#    def __init__(self, **kwargs):
#
#        super(GaussianCurveVector, self).__init__(**kwargs)
#
#    def get_vector(self, shape=None, size=None, mean=None, cov=None, dims=2):
#        """ Computes a Gaussian curve-shaped starting vector.
#
#        Parameters:
#        shape : A tuple. The shape of the start vector.
#
#        size : A tuple. The size of the supposed image. Must have the form (Z,
#                Y, X).
#
#        mean : A numpy array. The mean vector of the Gaussian. Default is zero.
#
#        cov : A numpy array. The covariance matrix of the Gaussian. Default is
#                the identity.
#
#        dims : A scalar. The number of dimensions of the output image. Default
#                is 2.
#        """
#        if size != None:
#            p = 1
#            for i in xrange(dims):
#                p *= size[i]
#            if axis == 1:
#                shape = (p, 1)
#            else:
#                shape = (1, p)
#        else:
#            if X != None:
#                p = X.shape[axis]
#                shape = (p, 1)
#            else:  # Assumes shape != None
#                p = shape[0] * shape[1]
#
#            size = [0] * dims
#            for i in xrange(dims):  # Split in equal-sized hypercube
#                size[i] = round(float(p) ** (1.0 / float(dims)))
#
#        if mean == None:
#            mean = [float(s - 1.0) / 2.0 for s in size]
#        if cov == None:
#            S = np.diag([s ** (1.0 / dims) for s in size])
#            invS = np.linalg.pinv(S)
#        else:
##            S = np.diag(np.diag(cov))
#            S = np.asarray(cov)
#            invS = np.linalg.pinv(S)
#
#        a = np.arange(size[0])
#        ans = np.reshape(a, (a.shape[0], 1)).tolist()
#        for i in xrange(1, dims):
#            b = np.arange(size[i]).tolist()
#            ans = [y + [x] for x in b for y in ans]
#
#        X = np.zeros((size))
#        for x in ans:
#            i = tuple(x)
#            x = np.array([x]) - np.array(mean)
#            v = np.dot(x, np.dot(invS, x.T))
#            X[i] = v[0, 0]
#
#        X = np.exp(-0.5 * X)
#        X /= np.sum(X)
#
##        s = []
##        X = 0
##        for i in xrange(dims):
##            x = np.arange(size[i]) - mean[i]
##            x = np.reshape(x, [size[i]] + s)
##            X = X + invS[i, i] * (x ** 2.0)
##            s.append(1)
#
#        w = np.reshape(X, (p, 1))
#
#        if self.normalise:
#            return w / norm(w)
#        else:
#            return w
#
#
#class GaussianCurveVectors(BaseStartVector):
#    """A start vector with multibple Gaussian curve shapes.
#
#    The gaussians are in an imagined 1D or 2D image. The output is a reshaped
#    vector corresponsing to a 1- or 2-dimensional image.
#    """
#
#    def __init__(self, num_points=3, normalise=True, **kwargs):
#        super(GaussianCurveVectors, self).__init__(normalise=normalise,
#                                                  **kwargs)
#
#        self.num_points = num_points
#
#    def get_vector(self, X=None, axis=1, shape=None, size=None,
#                   mean=None, cov=None, dims=2):
#        """ Computes a starting vector with set of Gaussian curve-shapes.
#
#        Parameters:
#        X     : The matrix for which we need a start vector. Used in
#                conjunction with axis to determine the shape of the start
#                vector.
#
#        axis  : The axis along X which the shape is taken.
#
#        shape : The shape of the start vector, may be passed instead of X.
#
#        size  : The size of the supposed image. Must have the form (Z, Y, X).
#                May be passed instead of X or shape.
#
#        means : The mean vectors of the Gaussians. Default is random.
#
#        covs  : The covariance matrices of the Gaussians. Default is random.
#
#        dims  : The number of dimensions of the output image. Default is 2.
#        """
#        if size != None:
#            p = 1
#            for i in xrange(dims):
#                p *= size[i]
#            if axis == 1:
#                shape = (p, 1)
#            else:
#                shape = (1, p)
#        else:
#            if X != None:
#                p = X.shape[axis]
#                shape = (p, 1)
#            else:  # Assumes shape != None
#                p = shape[0] * shape[1]
#
#            size = [0] * dims
#            for i in xrange(dims):  # Split in equal-sized hypercube
#                size[i] = round(float(p) ** (1.0 / float(dims)))
#
#        means = np.random.rand(1, 2)
#        for i in xrange(1, self.num_points):
#            dist = 0.0
#            p_best = 0
#            for j in xrange(20):
#                p = np.random.rand(1, 2)
#                dist_curr = np.min(np.sqrt(np.sum((means - p) ** 2.0, axis=1)))
#                if dist_curr > dist:
#                    p_best = p
#                    dist = dist_curr
#                if dist_curr > 0.3:
#                    break
#            means = np.vstack((means, p_best))
#
#        means[means < 0.05] = 0.05
#        means[means > 0.95] = 0.95
#        means[:, 0] *= size[0]
#        means[:, 1] *= size[1]
#        means = means.tolist()
#
#        covs = [0] * self.num_points
#        for i in xrange(self.num_points):
#            S1 = np.diag((np.abs(np.diag(np.random.rand(2, 2))) * 0.5) + 0.5)
#
#            S2 = np.random.rand(2, 2)
#            S2 = (((S2 + S2.T) / 2.0) - 0.5) * 0.9  # [0, 0.45]
#            S2 = S2 - np.diag(np.diag(S2))
#
#            S = S1 + S2
#
#            S /= np.max(S)
#
#            S *= float(min(size))
#
#            covs[i] = S.tolist()
#
#        vector = GaussianCurveVector(normalise=False)
#
#        X = np.zeros(shape)
#        for i in xrange(self.num_points):
#            X = X + vector.get_vector(size=size, dims=dims,
#                                      mean=means[i], cov=covs[i])
#
#        w = np.reshape(X, size)
#
#        if self.normalise:
#            return w / norm(w)
#        else:
#            return w