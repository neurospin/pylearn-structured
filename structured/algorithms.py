# -*- coding: utf-8 -*-
"""
The :mod:`structured.algorithms` module includes several algorithms used
throughout the package.

Algorithms may not be stateful. I.e., if they are classes, do not keep
references to objects with state in the algorithm objects. It should be
possible to copy and share algorithms between models, and thus they should not
depend on any state.

Created on Fri Feb  8 17:24:11 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: TBD.
"""

import numpy as np

import structured.utils as utils
from structured.utils import math
from time import time

__all = ['FastSVD', 'FastSparseSVD', 'FISTA', 'CONESTA', 'ExcessiveGapMethod']


def FastSVD(X, max_iter=100, start_vector=None):
    """A kernel SVD implementation.

    Performs SVD of given matrix. This is always faster than np.linalg.svd.
    Particularly, this is a lot faster than np.linalg.svd when M << N or
    M >> N, for an M-by-N matrix.

    Arguments:
    ---------
    X : The matrix to decompose

    Returns:
    -------
    v : The right singular vector.
    """
    M, N = X.shape
    if M < 80 and N < 80:  # Very arbitrary threshold for my computer ;-)
        _, _, V = np.linalg.svd(X, full_matrices=True)
        v = V[[0], :].T
    elif M < N:
        Xt = X.T
        K = np.dot(X, Xt)
        # TODO: Use module for this!
        t = np.random.rand(X.shape[0], 1)
#        t = start_vectors.RandomStartVector().get_vector(Xt)
        for it in xrange(max_iter):
            t_ = t
            t = np.dot(K, t_)
            t /= np.sqrt(np.sum(t_ ** 2.0))

            if np.sqrt(np.sum((t_ - t) ** 2.0)) < utils.TOLERANCE:
                break

        v = np.dot(Xt, t)
        v /= np.sqrt(np.sum(v ** 2.0))

    else:
        K = np.dot(X.T, X)
        # TODO: Use module for this!
        v = np.random.rand(X.shape[1], 1)
#        v = start_vectors.RandomStartVector().get_vector(X)
        for it in xrange(max_iter):
            v_ = v
            v = np.dot(K, v_)
            v /= np.sqrt(np.sum(v ** 2.0))

            if np.sqrt(np.sum((v_ - v) ** 2.0)) < utils.TOLERANCE:
                break

    return v


def FastSparseSVD(X, max_iter=100, start_vector=None):
    """A kernel SVD implementation for sparse CSR matrices.

    This is usually faster than np.linalg.svd when density < 20% and when
    M << N or N << M (at least one order of magnitude). When M = N >= 10000 it
    is faster when the density < 1% and always faster regardless of density
    when M = N < 10000.

    These are ballpark estimates that may differ on your computer.

    Arguments:
    ---------
    X : The matrix to decompose

    Returns:
    -------
    v : The right singular vector.
    """
    M, N = X.shape
    # TODO: Use module for this!
    p = np.random.rand(X.shape[1], 1)
#    p = start_vectors.RandomStartVector().get_vector(X)
    if M < N:
        Xt = X.T
        K = X.dot(Xt)
        t = X.dot(p)
        for it in xrange(max_iter):
            t_ = t
            t = K.dot(t_)
            t /= np.sqrt(np.sum(t_ ** 2.0))

            if np.sum((t_ - t) ** 2.0) < utils.TOLERANCE:
                break

        p = Xt.dot(t)
        normp = np.sqrt(np.sum(p ** 2.0))
        # Is the solution significantly different from zero (or TOLERANCE)?
        if normp >= utils.TOLERANCE:
            p /= normp
        else:
            p = np.zeros(p.shape) / np.sqrt(p.shape[0])

    else:
        K = X.T.dot(X)
        for it in xrange(max_iter):
            p_ = p
            p = K.dot(p_)
            normp = np.sqrt(np.sum(p ** 2.0))
            if normp > utils.TOLERANCE:
                p /= normp
            else:
                p = np.zeros(p.shape) / np.sqrt(p.shape[0])

            if np.sum((p_ - p) ** 2.0) < utils.TOLERANCE:
                break

    return p


def FISTA(X, y, function, beta, step, mu,
          eps=utils.TOLERANCE, max_iter=utils.MAX_ITER):
    """ The fast iterative shrinkage threshold algorithm.
    """
    z = betanew = betaold = beta

    t = []
    f = []
    for i in xrange(1, max_iter + 1):
        tm = time()

        z = betanew + ((i - 2.0) / (i + 1.0)) * (betanew - betaold)
        betaold = betanew
        betanew = function.prox(z - step * function.grad(X, y, z, mu), step)

        t.append(time() - tm)
#        f.append(function.f(X, y, betanew, mu=mu))
        f.append(function.f(X, y, betanew, mu=0.0))

        if math.norm(betanew - z) < eps * step:
            break

    return (betanew, f, t)


def CONESTA(X, y, function, beta, mu_start=None, mumin=utils.TOLERANCE,
            sigma=1.0, tau=0.5, dynamic=True,
            eps=utils.TOLERANCE, conts=50, max_iter=1000):

    if mu_start != None:
        mu = [mu_start]
    else:
        mu = [0.9 * function.mu(beta)]
    print "mu:", mu[0]
    eps0 = function.eps_opt(mu[0], X)
#    print "eps:", eps0

    tmin = 1.0 / function.Lipschitz(X, mumin, max_iter=100)

    t = []
    f = []
    Gval = []
#    beta_hat = beta

    i = 0
    while True:
        step = 1.0 / function.Lipschitz(X, mu[i], max_iter=100)
#        eps_plus = max(eps, function.eps_opt(mu[i], X))
        eps_plus = function.eps_opt(mu[i], X)
        print "eps_plus: ", eps_plus
        (beta, crit, tm) = FISTA(X, y, function, beta, step, mu[i],
                                 eps=eps_plus, max_iter=max_iter)
        print "Fista did iterations=", len(crit)
        t = t + tm
        f = f + crit

        mumin = min(mumin, mu[i])
        tmin = min(tmin, step)
        beta_tilde = function.prox(beta - tmin * function.grad(X, y,
                                                               beta, mumin),
                                   tmin)

        if (1.0 / tmin) * math.norm(beta - beta_tilde) < eps or i >= conts:
            print "%f < %f" % ((1. / tmin) * math.norm(beta - beta_tilde), eps)
            print "%d >= %d" % (i, conts)
            break

        if dynamic:
#            G = function.gap(X, y, beta, mumin,
            G = function.gap(X, y, beta, mu[i],
                                       eps=eps_plus, max_iter=max_iter)
#            G, beta_hat = function.gap(X, y, beta, beta_hat, mumin,
#                                       eps=eps_plus, max_iter=max_iter)
        else:
            G = eps0 * tau ** (i + 1)

        G = abs(G) / sigma
        print "Gap:", G

        if G <= utils.TOLERANCE and mu[i] <= utils.TOLERANCE:
            break

        Gval.append(G)
        mu_new = min(mu[i], function.mu_opt(G, X))
        print "mu_opt: ", mu_new
        mu.append(max(mumin, mu_new))
#        mu.append(mu_new)

#        print "mu:", mu[i + 1]

        i = i + 1

    return (beta, f, t, mu, Gval)


def ExcessiveGapMethod(X, y, function, eps=utils.TOLERANCE,
                       max_iter=utils.MAX_ITER):
    """ The excessive gap method for strongly convex functions.

    Parameters
    ----------
    function : The function to minimise. It contains two parts, function.g is
            the strongly convex part and function.h is the smoothed part of the
            function.
    """
    A = function.h.A()

    u = [0] * len(A)
    for i in xrange(len(A)):
        u[i] = np.zeros((A[i].shape[0], 1))

    L = function.Lipschitz(X, 1.0, max_iter=1000)
    mu = [L]
    beta0 = function.betahat(X, y, u)  # u is zero here
    beta = beta0
    alpha = function.V(u, beta, L)  # u is zero here

    k = 0

    t = []
    f = []  # function.f(X, y, beta[0], mu[0])]
    # mu[0] * function.h.D()
    ulim = [4.0 * L * function.h.M() / ((k + 1.0) * (k + 2.0))]

    while True:
        tm = time()

        tau = 2.0 / (float(k) + 3.0)

        alpha_hat = function.h.alpha(beta, mu[k])
        for i in xrange(len(alpha_hat)):
            u[i] = (1.0 - tau) * alpha[i] + tau * alpha_hat[i]

        mu.append((1.0 - tau) * mu[k])
        betahat = function.betahat(X, y, u)
        beta = ((1.0 - tau) * beta + tau * betahat)
        alpha = function.V(u, betahat, L)

        t.append(time() - tm)
        f.append(function.f(X, y, beta, mu=0.0))  # utils.TOLERANCE))
#        ulim.append(mu[k + 1] * function.h.D())
        ulim.append(4.0 * L * function.h.M() / ((k + 1.0) * (k + 2.0)))

        if mu[k] * function.h.M() < eps / 15.0 or k >= max_iter:
            break

        k = k + 1

    return (beta, f, t, mu, ulim, beta0)