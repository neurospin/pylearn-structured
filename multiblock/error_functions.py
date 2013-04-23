# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 10:54:29 2013

@author: tl236864
"""

__all__ = ['ErrorFunction', 'ConvexErrorFunction',
           'DifferentiableErrorFunction', 'NonDifferentiableErrorFunction']

import abc
import numpy as np

from utils import norm, norm1, TOLERANCE
import prox_ops
import warnings


class ErrorFunction(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        super(ErrorFunction, self).__init__()


class ConvexErrorFunction(ErrorFunction):

    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        super(ConvexErrorFunction, self).__init__(**kwargs)

    @abc.abstractmethod
    def f(self, *args, **kwargs):
        raise NotImplementedError('Abstract method "f" must be specialised!')


class ProximalOperatorErrorFunction(ErrorFunction):

    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        super(ProximalOperatorErrorFunction, self).__init__(**kwargs)

    @abc.abstractmethod
    def f(self, *args, **kwargs):
        raise NotImplementedError('Abstract method "f" must be specialised!')

    @abc.abstractmethod
    def prox(self, x):
        raise NotImplementedError('Abstract method "prox" must be ' \
                                  'specialised!')


class DifferentiableErrorFunction(ErrorFunction):

    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        super(DifferentiableErrorFunction, self).__init__(**kwargs)

    @abc.abstractmethod
    def f(self, *args, **kwargs):
        raise NotImplementedError('Abstract method "f" must be specialised!')

    @abc.abstractmethod
    def grad(self, *args, **kwargs):
        raise NotImplementedError('Abstract method "grad" must be specialised!')


class NonDifferentiableErrorFunction(ErrorFunction):

    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        super(NonDifferentiableErrorFunction, self).__init__(**kwargs)

    @abc.abstractmethod
    def f(self, *args, **kwargs):
        raise NotImplementedError('Abstract method "f" must be specialised!')


class ZeroErrorFunction(ConvexErrorFunction, ProximalOperatorErrorFunction):

    def __init__(self):
        super(ZeroErrorFunction, self).__init__()

    def f(self, *args, **kwargs):
        return 0

    def prox(self, beta, *args, **kwargs):
        return beta


class MeanSquareRegressionError(DifferentiableErrorFunction,
                                ConvexErrorFunction):

    def __init__(self, X, y):
        super(MeanSquareRegressionError, self).__init__()

        self.X = X
        self.y = y

    def f(self, beta):
        return norm(self.y - np.dot(self.X, beta)) ** 2

    def grad(self, beta):
        return 2 * np.dot(self.X.T, np.dot(self.X, beta) - self.y)


class L1(ProximalOperatorErrorFunction, ConvexErrorFunction):

    def __init__(self, l):
        super(L1, self).__init__()

        self.l = l

#        if prox_op == None:
#            self.prox_op = prox_ops.L1(self.l)
#        else:
#            self.prox_op = prox_op

    def f(self, beta):
        return self.l * norm1(beta)

#    def prox(self, beta):
#        return self.prox_op.prox(beta)

    def prox(self, x, factor=1, allow_empty=False):

        xorig = x.copy()
        lorig = factor * self.l
        l = lorig

        warn = False
        while True:
            x = xorig

            sign = np.sign(x)
            np.absolute(x, x)
            x -= l
            x[x < 0] = 0
            x = np.multiply(sign, x)

            if norm(x) > TOLERANCE or allow_empty:
                break
            else:
                warn = True
                # TODO: Improved this!
                l *= 0.95  # Reduce by 5 % until at least one significant

        if warn:
            warnings.warn('Soft threshold was too large (all variables ' \
                          'purged). Threshold reset to %f (was %f)'
                          % (l, lorig))
        return x