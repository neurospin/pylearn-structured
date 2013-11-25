# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.algorithms` module includes several algorithms used
throughout the package.

Algorithms may not store states. I.e., if they are classes, do not keep
references to objects with state in the algorithm objects. It should be
possible to copy and share algorithms between estimators, and thus they should
not depend on any state.

Created on Fri Feb  8 17:24:11 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: TBD.
"""
import abc
import numpy as np

import parsimony.utils.consts as consts
import parsimony.utils.maths as maths
import parsimony.functions as functions

from time import time, clock

#TODO: This depends on the OS. We should try to be clever here ...
#time_func = time
time_func = clock

__all__ = ['BaseAlgorithm',
           'ImplicitAlgorithm',
           'FastSVD', 'FastSparseSVD',

           'ExplicitAlgorithm',
           'FISTA', 'CONESTA', 'StaticCONESTA', 'DynamicCONESTA',
           'ExcessiveGapMethod']


class BaseAlgorithm(object):

    def check_compatability(self, function, interfaces):

        for interface in interfaces:
            if not isinstance(function, interface):
                raise ValueError("%s does not implement interface %s" % \
                                (str(function), str(interface)))

    def set_params(self, **kwargs):

        for k in kwargs:
            self.__setattr__(k, kwargs[k])


class ImplicitAlgorithm(BaseAlgorithm):
    """Implicit algorithms are algorithms that do not use a loss function, but
    instead minimise or maximise some underlying function implicitly, from the
    data.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(X, **kwargs):
        raise NotImplementedError('Abstract method "__call__" must be ' \
                                  'specialised!')


class ExplicitAlgorithm(BaseAlgorithm):
    """Explicit algorithms are algorithms that minimises a given function
    explicitly from properties of the function.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(function, beta, **kwargs):
        """Call this object to obtain the variable that gives the minimum.

        Arguments:
        =========
        X : The data.

        function : The function to minimise.

        beta : A start vector.
        """
        raise NotImplementedError('Abstract method "__call__" must be ' \
                                  'specialised!')


class FastSVD(ImplicitAlgorithm):

    def __call__(self, X, max_iter=100, start_vector=None):
        """A kernel SVD implementation.

        Performs SVD of given matrix. This is always faster than np.linalg.svd.
        Particularly, this is a lot faster than np.linalg.svd when M << N or
        M >> N, for an M-by-N matrix.

        Arguments:
        ---------
        X : The matrix to decompose.

        Returns:
        -------
        v : The right singular vector of X that corresponds to the largest
                singular value.
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

                if np.sqrt(np.sum((t_ - t) ** 2.0)) < consts.TOLERANCE:
                    break

            v = np.dot(X.T, t)
            v /= np.sqrt(np.sum(v ** 2.0))

        else:
            K = np.dot(X.T, X)
            # TODO: Use module for this!
            v = np.random.rand(X.shape[1], 1)
            v /= maths.norm(v)
    #        v = start_vectors.RandomStartVector().get_vector(X)
            for it in xrange(max_iter):
                v_ = v
                v = np.dot(K, v_)
                v /= np.sqrt(np.sum(v ** 2.0))

                if np.sqrt(np.sum((v_ - v) ** 2.0)) < consts.TOLERANCE:
                    break

        return v


class FastSparseSVD(ImplicitAlgorithm):

    def __call__(self, X, max_iter=100, start_vector=None):
        """A kernel SVD implementation for sparse CSR matrices.

        This is usually faster than np.linalg.svd when density < 20% and when
        M << N or N << M (at least one order of magnitude). When M = N >= 10000
        it is faster when the density < 1% and always faster regardless of
        density when M = N < 10000.

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
                if a < consts.TOLERANCE:
                    break

            v = X.T.dot(t)
            v /= np.sqrt(np.sum(v ** 2.0))

        else:
            K = X.T.dot(X)
            # TODO: Use module for this!
            v = np.random.rand(X.shape[1], 1)
            v /= maths.norm(v)
    #        v = start_vectors.RandomStartVector().get_vector(X)
            for it in xrange(max_iter):
                v_ = v
                v = K.dot(v_)
                v /= np.sqrt(np.sum(v ** 2.0))

                a = float(np.sqrt(np.sum((v_ - v) ** 2.0)))
                if a < consts.TOLERANCE:
                    break

        return v


class FISTA(ExplicitAlgorithm):
    """ The fast iterative shrinkage threshold algorithm.
    """
    INTERFACES = [functions.Gradient,
                  functions.LipschitzContinuousGradient,
                  functions.ProximalOperator,
                  functions.NesterovFunction
                 ]

    def __init__(self, step=None, output=False,
                 eps=consts.TOLERANCE,
                 max_iter=consts.MAX_ITER, min_iter=1):

        self.step = step
        self.output = output
        self.eps = eps
        self.max_iter = max_iter
        self.min_iter = min_iter

    def __call__(self, function, beta):

        self.check_compatability(function, self.INTERFACES)

        z = betanew = betaold = beta

        if self.step == None:
            self.step = 1.0 / function.L()

        if self.output:
            t = []
            f = []
        for i in xrange(1, self.max_iter + 1):
            if self.output:
                tm = time_func()

            z = betanew + ((i - 2.0) / (i + 1.0)) * (betanew - betaold)
            betaold = betanew
            betanew = function.prox(z - self.step * function.grad(z),
                                    self.step)

            if self.output:
                t.append(time_func() - tm)
                f.append(function.f(betanew))

            if (1.0 / self.step) * maths.norm(betanew - z) < self.eps \
                    and i >= self.min_iter:
                break

        if self.output:
            output = {"t": t, "f": f}
            return betanew, output
        else:
            return betanew


class CONESTA(ExplicitAlgorithm):
    """COntinuation with NEsterov smoothing in a Soft-Thresholding Algorithm,
    or CONESTA for short.
    """
    INTERFACES = [functions.NesterovFunction,
                  functions.Continuation,
                  functions.LipschitzContinuousGradient,
                  functions.ProximalOperator,
                  functions.Gradient,
                  functions.DualFunction
                 ]

    def __init__(self, mu_start=None, mu_min=consts.TOLERANCE, tau=0.5,
                 dynamic=True, continuations=30,

                 output=False,
                 eps=consts.TOLERANCE,
                 max_iter=consts.MAX_ITER, min_iter=1):

        self.mu_start = mu_start
        self.mu_min = mu_min
        self.tau = tau
        self.dynamic = dynamic
        self.continuations = continuations

        self.output = output
        self.eps = eps
        self.max_iter = max_iter
        self.min_iter = min_iter

        self.FISTA = FISTA(output=self.output,
                           eps=self.eps,
                           max_iter=self.max_iter, min_iter=self.min_iter)

    def __call__(self, function, beta):

        self.check_compatability(function, self.INTERFACES)

        if self.mu_start != None:
            mu = [self.mu_start]
        else:
            mu = [0.9 * function.estimate_mu(beta)]

#        old_mu = function.get_mu()
        function.set_mu(self.mu_min)
        tmin = 1.0 / function.L()
        function.set_mu(mu[0])

        max_eps = function.eps_max(mu[0])

#        G = eps0 = min(max_eps, function.eps_opt(mu[0]))
        G = min(max_eps, function.eps_opt(mu[0]))

        if self.output:
            t = []
            f = []
            Gval = []

        i = 0
        while True:
            stop = False

            tnew = 1.0 / function.L()
            eps_plus = min(max_eps, function.eps_opt(mu[-1]))
            self.FISTA.set_params(step=tnew, eps=eps_plus)
            if self.output:
                (beta, info) = self.FISTA(function, beta)
                fval = info["f"]
                tval = info["t"]
            else:
                beta = self.FISTA(function, beta)

            self.mu_min = min(self.mu_min, mu[-1])
            tmin = min(tmin, tnew)
            function.set_mu(self.mu_min)
            # Take one ISTA step to use in the stopping criterion.
            beta_tilde = function.prox(beta - tmin * function.grad(beta),
                                       tmin)
            function.set_mu(mu[-1])

            if (1.0 / tmin) * maths.norm(beta - beta_tilde) < self.eps \
                    or i >= self.continuations:
                print "%f < %f" % ((1. / tmin) \
                                * maths.norm(beta - beta_tilde), self.eps)
                print "%d >= %d" % (i, self.continuations)
                stop = True

            if self.output:
                gap_time = time_func()
            if self.dynamic:
                G_new = function.gap(beta)
                # TODO: Warn if G_new < 0.
                G_new = abs(G_new)  # Just in case ...

                if G_new < G:
                    G = G_new
                else:
                    G = self.tau * G

            else:  # Static

    #            G_new = eps0 * tau ** (i + 1)
                G = self.tau * G

            print "Gap:", G
            if self.output:
                gap_time = time_func() - gap_time
                Gval.append(G)

                f = f + fval
                tval[-1] += gap_time
                t = t + tval

            if (G <= consts.TOLERANCE and mu[-1] <= consts.TOLERANCE) or stop:
                break

            mu_new = min(mu[-1], function.mu_opt(G))
            self.mu_min = min(self.mu_min, mu_new)
            if self.output:
                mu = mu + [max(self.mu_min, mu_new)] * len(fval)
            else:
                mu.append(max(self.mu_min, mu_new))
            function.set_mu(mu_new)

            i = i + 1

        if self.output:
            info = {"t": t, "f": f, "mu": mu, "gap": Gval}
            return (beta, info)
        else:
            return beta


class StaticCONESTA(CONESTA):
    """COntinuation with NEsterov smoothing in a Soft-Thresholding Algorithm,
    or CONESTA for short, with a statically decreasing \mu.
    """
    def __init__(self, **kwargs):

        kwargs.pop("dynamic")  # We ignore this if it was given ...

        super(StaticCONESTA, self).__init__(dynamic=False, **kwargs)


class DynamicCONESTA(CONESTA):
    """COntinuation with NEsterov smoothing in a Soft-Thresholding Algorithm,
    or CONESTA for short, with a dynamically decreasing \mu.
    """
    def __init__(self, **kwargs):

        kwargs.pop("dynamic")  # We ignore this if it was given ...

        super(StaticCONESTA, self).__init__(dynamic=True, **kwargs)


class ExcessiveGapMethod(ExplicitAlgorithm):
    """Nesterov's excessive gap method for strongly convex functions.
    """
    INTERFACES = [functions.NesterovFunction,
#                  functions.Continuation,
                  functions.LipschitzContinuousGradient,
#                  functions.ProximalOperator,
#                  functions.Gradient,
                  functions.GradientMap,
                  functions.DualFunction
                 ]

    def __init__(self, output=False,
                 eps=consts.TOLERANCE,
                 max_iter=consts.MAX_ITER, min_iter=1):

        self.output = output
        self.eps = eps
        self.max_iter = max_iter
        self.min_iter = min_iter

    def __call__(self, function, beta=None):
        """The excessive gap method for strongly convex functions.

        Parameters
        ----------
        function : The function to minimise. It contains two parts, function.g
                is the strongly convex part and function.h is the smoothed part
                of the function.
        """
        A = function.h.A()

        u = [0] * len(A)
        for i in xrange(len(A)):
            u[i] = np.zeros((A[i].shape[0], 1))

        # L = lambda_max(A'A) / (lambda_min(X'X) + k)
        L = function.L()
        print "L:", L

        mu = [2.0 * L]
        function.h.set_mu(mu)
        beta0 = function.betahat(u)  # u is zero here
        beta = beta0
        alpha = function.V(u, beta, L)  # u is zero here

        t = []
        f = []
        ubound = []

        k = 0

        while True:
            if self.output:
                tm = time_func()

            tau = 2.0 / (float(k) + 3.0)

            function.h.set_mu(mu[k])
            alpha_hat = function.h.alpha(beta)
            for i in xrange(len(alpha_hat)):
                u[i] = (1.0 - tau) * alpha[i] + tau * alpha_hat[i]

            mu.append((1.0 - tau) * mu[k])
            betahat = function.betahat(u)
            beta = (1.0 - tau) * beta + tau * betahat
            alpha = function.V(u, betahat, L)

            ulim = mu[k + 1] * function.h.M()
            if self.output:
                t.append(time_func() - tm)
                mu_old = function.h.get_mu()
                function.h.set_mu(0.0)
                f.append(function.f(beta))
                function.h.set_mu(mu_old)

#                ulim.append(2.0 * function.h.M() * mu[0] / ((float(k) + 1.0) * (float(k) + 2.0)))
                ubound.append(ulim)

            if ulim < self.eps or k >= self.max_iter:
                break

            k = k + 1

        if self.output:
            output = {"t": t, "f": f, "mu": mu, "upper_bound": ubound,
                      "beta_start": beta0}
            return (beta, output)
        else:
            return beta