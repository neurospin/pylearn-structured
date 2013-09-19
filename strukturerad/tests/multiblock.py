# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 14:58:02 2013

@author:  Tommy Löfstedt <tommy.loefstedt@cea.fr>
@email:   tommy.loefstedt@cea.fr
@license: TBD
"""

import numpy as np
import abc

import matplotlib.pyplot as plot

import strukturerad.algorithms as algorithms
import strukturerad.utils.math as math


class LossFunction(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):

        super(LossFunction, self).__init__(**kwargs)

    @abc.abstractmethod
    def f(self, *args, **kwargs):

        raise NotImplementedError('Abstract method "f" must be specialised!')


class ProximalOperator(LossFunction):

    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):

        super(ProximalOperator, self).__init__(**kwargs)

    @abc.abstractmethod
    def prox(self, x):

        raise NotImplementedError('Abstract method "prox" must be ' \
                                  'specialised!')


class LinearRegressionError(LossFunction):
    """Loss function for linear regression. Represents the function:

        f(b) = (1 / 2) * ||X*b - y||²,

    where ||.||² is the L2 norm.
    """
    def __init__(self, X, y, **kwargs):

        super(LinearRegressionError, self).__init__(**kwargs)

        self.X = X
        self.y = y

        self._lipschitz = None

    def f(self, beta, **kwargs):

        return 0.5 * np.sum((self.y - np.dot(self.X, beta)) ** 2.0)

    def grad(self, beta, *args, **kwargs):

        return np.dot((np.dot(self.X, beta) - self.y).T, self.X).T

    def Lipschitz(self, *args, **kwargs):

        if self._lipschitz == None:  # Squared largest singular value
            v = algorithms.FastSVD(max_iter=100).run(self.X)
            us = np.dot(self.X, v)
            self._lipschitz = np.sum(us ** 2.0)

        return self._lipschitz


class CovarianceError(LossFunction):
    """Loss function for covariance. Represents the function

        f(w, c) = w'X'Yc.
    """
    def __init__(self, X, Y, **kwargs):

        super(CovarianceError, self).__init__(**kwargs)

        self.X = X
        self.Y = Y
        self.XY = np.dot(X.T, Y)

    def f(self, w, c):
        return -np.dot(np.dot(self.X, w).T, np.dot(self.Y, c))

    def grad(self, c):
        return -np.dot(np.dot(self.Y, c).T, self.X).T

    def step(self, w, c, a=0.35, b=0.5):
        dx = -self.grad(c)
        grad = -dx
        t = 1.0
        while True:
            fval = self.f(w, c)
            ftdx = self.f(w + t * dx, c)
            atgraddx = a * t * np.dot(grad.T, dx)

            print "ftdx:", ftdx
            print "f + atgraddx:", (fval + atgraddx)

            if (ftdx <= fval + atgraddx):
                break

            t *= b

        return t


class L1(ProximalOperator):
    """The proximal operator of the L1 loss function

        f(x) = l * ||x||_1,

    where ||x||_1 is the L1 loss function.
    """
    def __init__(self, l, **kwargs):

        super(L1, self).__init__(**kwargs)

        self.l = l

    def f(self, beta):

        return self.l * math.norm1(beta)

    def prox(self, x, factor=1.0):

        l = factor * self.l
        return (np.abs(x) > l) * (x - l * np.sign(x - l))


np.random.seed(42)

p = 100
n = 60
X = np.random.randn(n, p)
betastar = np.concatenate((np.zeros((p / 2, 1)),
                           np.random.randn(p / 2, 1)))
betastar = np.sort(np.abs(betastar), axis=0)
Y = np.dot(X, betastar)

eps = 0.01
maxit = 1000


def f(X, Y, w, c, l, k):
    return -0.5 * np.dot(np.dot(X, w).T, np.dot(Y, c))[0, 0] \
            + 0.5 * l * (np.dot(w.T, w) - 1.0)[0, 0] \
            + 0.5 * k * (np.dot(c.T, c) - 1.0)[0, 0]


def grad(X, Y, w, c, l):
    return -np.dot(X.T, np.dot(Y, c)) + l * w


def step(f, grad, X, Y, w, c, l, k, a=0.35, b=0.5):
    dx = -grad(X, Y, w, c, l)
    grad = -dx
    t = 1.0
    while True:
        fval = f(X, Y, w, c, l, k)
        ftdx = f(X, Y, w + t * dx, c, l, k)
        atgraddx = a * t * np.dot(grad.T, dx)[0, 0]

        print "ftdx:", ftdx
        print "f + atgraddx:", (fval + atgraddx)

        if (ftdx <= fval + atgraddx):
            break

        t *= b

    return t

w = np.random.rand(X.shape[1], 1)
w = w / np.linalg.norm(w)
c = np.random.rand(Y.shape[1], 1)
c = c / np.linalg.norm(c)
l = k = 1.0

for i in range(3):
    print f(X, Y, w, c, l, k)
    grad_x = grad(X, Y, w, c, l)
    print np.linalg.norm(grad_x)
    t_ = step(f, grad, X, Y, w, c, l, k)
    w_ = w - t_ * grad_x
    print "norm(w):", np.linalg.norm(w_)
    w_ = w_ / np.linalg.norm(w_)
    print "t:", t_, ", f:", f(X, Y, w_, c, l, k)

    print

    print "f:", f(X, Y, w, c, l, k)
    grad_y = grad(Y, X, c, w, k)
    print "grad:", np.linalg.norm(grad_y)
    t_ = step(f, grad, Y, X, c, w, k, l)
    c_ = c - t_ * grad_y
    print "norm(c):", np.linalg.norm(c_)
    c_ = c_ / np.linalg.norm(c_)
    print "t:", t_, ", f:", f(X, Y, w, c_, l, k)

    w = w_
    c = c_

    print
    print "f:", f(X, Y, w, c, l, k)

w = np.random.rand(X.shape[1], 1)
w = w / np.linalg.norm(w)
c = np.random.rand(Y.shape[1], 1)
c = c / np.linalg.norm(c)

t = np.dot(X, w)
c = np.dot(Y.T, t)
c = c / np.linalg.norm(c)
u = np.dot(Y, c)
w = np.dot(X.T, u)
w = w / np.linalg.norm(w)
t = np.dot(X, w)

print "cov:", -0.5 * np.dot(t.T, u)[0, 0]

#t = step(f, grad, X, Y, w, c)
#print "t:", t
#w = w - t * grad_x
#grad_y = grad(Y, X, c, w)
#t = step(f, grad, Y, X, c, w)
#print "t:", t
#c = c - t * grad_y
#print f(X, Y, w, c)
#
#print
#
#print err_x.f(w, c)
#grad_x = err_x.grad(c)
#print np.linalg.norm(grad_x)
#t = err_x.step(w, c)
#w = w - t * grad_x
#grad_y = err_y.grad(w)
#t = err_y.step(c, w)
#c = c - t * grad_y
#print err_x.f(w, c)

#f = []
#for i in range(1):
#    f.append(err_x.f(w, c))
#    print "f:", err_x.f(w, c)
#    grad = err_x.grad(c)
#    t = err_x.step(w, c)
#    w = w - t * grad
#    grad_x = err_y.grad(w)
#    t = err_y.step(c, w)
#    c = c - t * grad_x
#    print "f + dx:", err_x.f(w - t * grad_x, c)
#
#plot.plot(f)
#plot.show()

#lr = LinearRegressionError(X, y)
#
#L = [[],
#     []]