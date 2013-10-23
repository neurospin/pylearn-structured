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
from time import time, clock

#time_func = time
time_func = clock

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
        K = np.dot(X, X.T)
        # TODO: Use module for this!
        t = np.random.rand(X.shape[0], 1)
#        t = start_vectors.RandomStartVector().get_vector(Xt)
        for it in xrange(max_iter):
            t_ = t
            t = np.dot(K, t_)
            t /= np.sqrt(np.sum(t ** 2.0))

            if np.sqrt(np.sum((t_ - t) ** 2.0)) < utils.TOLERANCE:
                break

        v = np.dot(X.T, t)
        v /= np.sqrt(np.sum(v ** 2.0))

    else:
        K = np.dot(X.T, X)
        # TODO: Use module for this!
        v = np.random.rand(X.shape[1], 1)
        v /= math.norm(v)
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
    if M < N:
        K = X.dot(X.T)
#        t = X.dot(p)
        # TODO: Use module for this!
        t = np.random.rand(X.shape[0], 1)
        for it in xrange(max_iter):
            t_ = t
            t = K.dot(t_)
            t /= np.sqrt(np.sum(t ** 2.0))

            a = float(np.sqrt(np.sum((t_ - t) ** 2.0)))
            if a < utils.TOLERANCE / 1000.0:
                break

        v = X.T.dot(t)
        v /= np.sqrt(np.sum(v ** 2.0))

    else:
        K = X.T.dot(X)
        # TODO: Use module for this!
        v = np.random.rand(X.shape[1], 1)
        v /= math.norm(v)
#        v = start_vectors.RandomStartVector().get_vector(X)
        for it in xrange(max_iter):
            v_ = v
            v = K.dot(v_)
            v /= np.sqrt(np.sum(v ** 2.0))

            a = float(np.sqrt(np.sum((v_ - v) ** 2.0)))
            if a < utils.TOLERANCE / 1000.0:
                break

    return v


def FISTA(X, y, function, beta, step, mu,
          eps=utils.TOLERANCE,
          max_iter=utils.MAX_ITER, min_iter=1):
    """ The fast iterative shrinkage threshold algorithm.
    """
    z = betanew = betaold = beta

    t = []
    f = []
    for i in xrange(1, max_iter + 1):
        tm = time_func()

        z = betanew + ((i - 2.0) / (i + 1.0)) * (betanew - betaold)
        betaold = betanew
        betanew = function.prox(z - step * function.grad(X, y, z, mu), step)

        t.append(time_func() - tm)
        f.append(function.f(X, y, betanew, mu=0.0))

        if math.norm(betanew - z) < eps * step and i >= min_iter:
            break

    return (betanew, f, t)


def CONESTA(X, y, function, beta, mu_start=None, mumin=utils.TOLERANCE,
            sigma=1.0, tau=0.5, dynamic=True,
            eps=utils.TOLERANCE,
            conts=50, max_iter=1000, min_iter=1, init_iter=1000):

    if mu_start != None:
        mu = [mu_start]
    else:
        mu = [0.9 * function.mu(beta)]
    print "mu0:", mu[0]
    max_eps = 10000.0
    G = eps0 = min(max_eps, function.eps_opt(mu[0], X))

    tmin = 1.0 / function.Lipschitz(X, mumin, max_iter=1000)

    beta_hat = None

    t = []
    f = []
    Gval = []

    i = 0
    while True:
        stop = False

        step = 1.0 / function.Lipschitz(X, mu[i], max_iter=100)
        eps_plus = min(max_eps, G)  # function.eps_opt(mu[i], X))
#        print "eps_plus: ", eps_plus
        (beta, fval, tval) = FISTA(X, y, function, beta, step, mu[i],
                                 eps=eps_plus,
                                 max_iter=max_iter,
#                                 max_iter=int(float(max_iter) / np.sqrt(i + 1)),
                                 min_iter=1)
        print "FISTA did iterations =", len(fval)

        fista_its = len(fval)

        mumin = min(mumin, mu[i])
        tmin = min(tmin, step)
        beta_tilde = function.prox(beta - tmin * function.grad(X, y,
                                                               beta, mumin),
                                   tmin)

        if (1.0 / tmin) * math.norm(beta - beta_tilde) < eps or i >= conts:
            print "%f < %f" % ((1. / tmin) * math.norm(beta - beta_tilde), eps)
            print "%d >= %d" % (i, conts)
            stop = True

        gap_time = time_func()
        if dynamic:
            G_new, beta_hat = function.gap(X, y, beta, beta_hat,
                                       mu[i],
                                       eps=eps_plus,
                                       max_iter=max(1, int(0.1 * fista_its)),
                                       min_iter=1,
                                       init_iter=init_iter)
        else:
            G_new = eps0 * tau ** (i + 1)

        G = min(G, abs(G_new)) / sigma
        gap_time = time_func() - gap_time
        print "Gap:", G
        Gval.append(G)

        tval[-1] += gap_time
        t = t + tval
        f = f + fval

        if (G <= utils.TOLERANCE and mu[i] <= utils.TOLERANCE) or stop:
            break

        mu_new = min(mu[i], function.mu_opt(G, X))
#        print "mu_opt: ", mu_new
        mu.append(max(mumin, mu_new))

        i = i + 1

    return (beta, f, t, mu, Gval)


def ExcessiveGapMethod(X, y, function, eps=utils.TOLERANCE,
                       max_iter=utils.MAX_ITER, f_star=0.0):
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

    # L = lambda_max(A'A) / (lambda_min(X'X) + k)
#    L = function.Lipschitz(X, max_iter=1000000)
    L = function.Lipschitz(X, max_iter=1000000)
    print "L:", L

    mu = [2.0 * L]
    beta0 = function.betahat(X, y, u)  # u is zero here
    beta = beta0
    alpha = function.V(u, beta, L)  # u is zero here

#    print "f  :", function.g.f(X, y, beta) + function.h.f(beta, mu[0])
#    print "phi:", function.g.f(X, y, beta) + function.h.phi(beta, alpha, mu=0.0)

    t = []
    f = []
    ulim = []

    k = 0

#    _f = []
#    _phi = []

    while True:
        tm = time_func()

        tau = 2.0 / (float(k) + 3.0)

        alpha_hat = function.h.alpha(beta, mu[k])
        for i in xrange(len(alpha_hat)):
            u[i] = (1.0 - tau) * alpha[i] + tau * alpha_hat[i]

        mu.append((1.0 - tau) * mu[k])
        betahat = function.betahat(X, y, u)
        beta = (1.0 - tau) * beta + tau * betahat
        alpha = function.V(u, betahat, L)

#        _f.append(function.g.f(X, y, beta) + function.h.f(beta, mu[k+1]))
#        _phi.append(function.g.f(X, y, betahat) + function.h.phi(betahat, alpha, mu=0.0))

        t.append(time_func() - tm)
        f.append(function.f(X, y, beta, mu=0.0))
        ulim.append(2.0 * function.h.M() * mu[0] / ((float(k) + 1.0) * (float(k) + 2.0)))
#        ulim.append(mu[k + 1] * function.h.M())

        if ulim[-1] < eps or k >= max_iter:
            break

        k = k + 1

    print "L:", L
    print "mu[-1]:", mu[-1]

#    import matplotlib.pyplot as plot
#    plot.plot([_f[i] - f_star for i in xrange(len(_f))], 'r')
#    plot.plot([_phi[i] - f_star for i in xrange(len(_phi))], 'g')
#    plot.plot([f[i] - f_star for i in xrange(len(f))])
#    plot.plot(ulim, '-.r')
#    plot.show()

    return (beta, f, t, mu, ulim, beta0)