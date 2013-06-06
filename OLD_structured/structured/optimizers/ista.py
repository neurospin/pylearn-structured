# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:55:08 2013

@author: edouard.duchesnay@cea.fr
@author: fouad.hadjselem@cea.fr
@author: vincent.guillemot@cea.fr
@author: lofstedt.tommy@gmail.com
@author: vincent.frouin@cea.fr
"""
from math import log
import numpy as np
from numpy.linalg import norm, eig, svd
from structured.optimizers.base import Optimizer


class ISTA(Optimizer):
    """ISTA algorithm. Minimize a function f = g + h. Where g is a smooth
    (differentiable) and and h is a convex (possibly non smooth) function.

    Parameters
    ----------

    f: function
       function to be minimized

    grad_g: function
       Gradiant of g the smooth part.

    prox_h: function
       proximal operator of the convex (possibly non smooth) part.

    t: float
      Time size.

    epsilon: float
       Accuracy value.

    kmax: int
        Max iteration number.
    """
    def __init__(self, f, grad_g, prox_h, lambd=1, epsilon=1e-4, kmax=100000):
        self.f = f
        self.grad_g = grad_g
        self.prox_h = prox_h
        self.lambd = lambd
        self.epsilon = epsilon
        self.kmax = kmax

    def optimize(self, X, y, t):
        n, p = X.shape
        beta_old = np.zeros((p, 1))
        f_old = self.f(X, y, beta_old, lambd)
        self.f_beta_k = [f_old]
        for k in xrange(1, self.kmax + 1):
            #print "ite",k
            beta_new = self.prox_h(beta_old - t * self.grad_g(X, y, beta_old),
                                t * self.lambd)
            #f1 = self.f(X, y, beta_old, lambd)
            f_new = self.f(X, y, beta_new, lambd)
            self.f_beta_k.append(f_new)
            #if norm(beta_new - beta_old, 2) / t < self.epsilon:
            if abs(f_new - f_old) / f_old < self.epsilon:
                break
            beta_old = beta_new
            f_old = f_new
        self.beta = beta_new
        self.iterations = k


def ista(f, grad_g, prox_h, t, X, y, epsilon=1e-10, kmax=100000):
    ista = ISTA(f, grad_g, prox_h, t)
    ista.optimize(X, y, lambd)
    return ista.beta

if __name__ == "__main__":
    import time
    n = 250
    p = int(1E04)
    lambd = 1
    X = np.random.randn(n, p)
    betastar = np.concatenate((np.zeros((p / 2, 1)),
                               np.random.randn(p / 2, 1)))
    y = np.dot(X, betastar)
    print "Before SVD"
    start = time.time()
    U, s, Vh = svd(X)
    print "SVD time ellapsed:", (time.time() - start)
    t = 1 / np.max(s**2)
    from structured.lasso import mse_l1, grad_mse, prox_l1
    start = time.time()
    ista = ISTA(f=mse_l1, grad_g=grad_mse, prox_h=prox_l1, lambd=1)
    ista.optimize(X, y, t)
    print "Time ellapsed:", (time.time() - start), "Nb ite:", ista.iterations,\
        "Norm to true sol:", norm(ista.beta - betastar)
    #ista(f, grad_g, prox_h, t, X, y, t)
    import pylab
    pylab.plot(betastar[:, 0], '-', ista.beta[:, 0], '*')
    pylab.title("the iteration number is equal to " + str(ista.iterations))
    xi = [log(n) for n in range(1, (len(ista.f_beta_k) + 1))]
    pylab.show()
    pylab.plot(np.log(xi), ista.f_beta_k, '-')
    pylab.show()
 #xf = [log(n) for n in range(1, (len(fista.crit) + 1))]
#xi = [log(n) for n in range(1, (len(ista.crit) + 1))]
    #xfm = [log(n) for n in range(1, (len(fistam.crit) + 1))]
    #pylab.plot(xf, fista.crit, '--r', xi, ista.crit, '-b', xfm, fistam.crit, ':k')
    #pylab.show()