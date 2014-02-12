# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.algorithms` module includes several algorithms used
throughout the package.

Algorithms may not store states. I.e., if they are classes, do not keep
references to objects with state in the algorithm objects. It should be
possible to copy and share algorithms between estimators, and thus they should
not depend on any state.

There are currently two types of algorithms: implicit and explicit. The
difference is whether they run directly on the data (implicit) or if they have
an actual loss function than is minimised (explicit). Implicit algorithms take
the data as input, and then run on the data. Explicit algorithms take a loss
function and a start vector as input, and then minimise the function value
starting from the point of the start vector.

Created on Fri Feb  8 17:24:11 2013

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import abc
import copy
from time import time, clock

import numpy as np

import parsimony.functions.penalties as penalties
import parsimony.functions.interfaces as interfaces
from parsimony.functions.nesterov.interfaces import NesterovFunction
from parsimony.functions.multiblock.interfaces import MultiblockFunction
from parsimony.functions.multiblock.interfaces import MultiblockGradient
from parsimony.functions.multiblock.interfaces import MultiblockProjectionOperator
import parsimony.start_vectors as start_vectors
import parsimony.utils.consts as consts
import parsimony.utils.maths as maths

#TODO: This depends on the OS. We should try to be clever here ...
time_func = clock
#time_func = time

__all__ = ["BaseAlgorithm",
           "ImplicitAlgorithm",
           "FastSVD", "FastSparseSVD", "FastSVDProduct",

           "ExplicitAlgorithm",
           "ISTA", "FISTA", "CONESTA", "StaticCONESTA", "DynamicCONESTA",
           "ExcessiveGapMethod"]


class BaseAlgorithm(object):

    def check_compatibility(self, function, interfaces):
        """Check if the function considered implements the given interfaces
        """
        for interface in interfaces:
            if not isinstance(function, interface):
                raise ValueError("%s does not implement interface %s" %
                                (str(function), str(interface)))

    def set_params(self, **kwargs):

        for k in kwargs:
            self.__setattr__(k, kwargs[k])


class ImplicitAlgorithm(BaseAlgorithm):
    """Implicit algorithms are algorithms that do not use a loss function, but
    instead minimise or maximise some underlying function implicitly, from the
    data.

    Parameters
    ----------
    X : Regressor
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

        Parameters
        ----------
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

        Parameters
        ----------
        X : The matrix to decompose.

        max_iter : maximum allowed number of iterations

        start_vector : a start vector

        Returns
        -------
        v : The right singular vector of X that corresponds to the largest
                singular value.

        Example
        -------
        >>> import numpy as np
        >>> from parsimony.algorithms import FastSVD
        >>> np.random.seed(0)
        >>> X = np.random.random((10,10))
        >>> fast_svd = FastSVD()
        >>> fast_svd(X)
        array([[-0.3522974 ],
               [-0.35647707],
               [-0.35190104],
               [-0.34715338],
               [-0.19594198],
               [-0.24103104],
               [-0.25578904],
               [-0.29501092],
               [-0.42311297],
               [-0.27656382]])

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

        Parameters
        ----------
        X : The matrix to decompose

        max_iter : maximum allowed number of iterations

        start_vector : a start vector

        Returns
        -------
        v : The right singular vector.

        Example
        -------
        >>> import numpy as np
        >>> from parsimony.algorithms import FastSparseSVD
        >>> np.random.seed(0)
        >>> X = np.random.random((10,10))
        >>> fast_sparse_svd = FastSparseSVD()
        >>> fast_sparse_svd(X)
        array([[ 0.3522974 ],
               [ 0.35647707],
               [ 0.35190103],
               [ 0.34715338],
               [ 0.19594198],
               [ 0.24103104],
               [ 0.25578904],
               [ 0.29501092],
               [ 0.42311297],
               [ 0.27656382]])


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


class FastSVDProduct(ImplicitAlgorithm):

    def __call__(self, X, Y, start_vector=None,
                 eps=consts.TOLERANCE, max_iter=100, min_iter=1):
        """A kernel SVD implementation of a product of two matrices, X and Y.
        I.e. the SVD of np.dot(X, Y), but the SVD is computed without actually
        computing the matrix product.

        Performs SVD of a given matrix. This is always faster than
        np.linalg.svd when extracting only one, or a few, vectors.

        Parameters
        ----------
        X : The first matrix of the product.

        Y : The second matrix of the product.

        start_vector : The start vector.

        eps : Float. Tolerance.

        max_iter : Maximum number of iterations.

        min_iter : Minimum number of iterations.

        Returns
        -------
        v : The right singular vector of np.dot(X, Y) that corresponds to the
                largest singular value of np.dot(X, Y).

        Example
        -------
        >>> import numpy as np
        >>> from parsimony.algorithms import FastSVDProduct
        >>> np.random.seed(0)
        >>> X = np.random.random((15,10))
        >>> Y = np.random.random((10,5))
        >>> fast_svd = FastSVDProduct()
        >>> fast_svd(X, Y)
        array([[ 0.47169804],
               [ 0.38956366],
               [ 0.41397845],
               [ 0.52493576],
               [ 0.42285389]])
        """
        M, N = X.shape

        if start_vector is None:
            start_vector = start_vectors.RandomStartVector(normalise=True)
        v = start_vector.get_vector((Y.shape[1], 1))

        for it in xrange(1, max_iter + 1):
            v_ = v
            v = np.dot(X, np.dot(Y, v_))
            v = np.dot(Y.T, np.dot(X.T, v))
            v /= np.sqrt(np.sum(v ** 2.0))

            if np.sqrt(np.sum((v_ - v) ** 2.0)) < eps \
                    and it >= min_iter:
                break

        return v


class ISTA(ExplicitAlgorithm):
    """ The iterative shrinkage threshold algorithm.
    """
    INTERFACES = [interfaces.Gradient,
                  # TODO: We should use a step size here instead of the
                  # Lipschitz constant. All functions don't have L, but will
                  # still run in FISTA with a small enough step size.
                  # Updated: Use StepSize instead!!
                  interfaces.LipschitzContinuousGradient,
                  interfaces.ProximalOperator,
                 ]

    def __init__(self, step=None, output=False,
                 eps=consts.TOLERANCE,
                 max_iter=consts.MAX_ITER, min_iter=1):
        """
        Parameters
        ----------
        step : Step for each iteration.

        output : Boolean. Get output information.

        eps : Float. Tolerance.

        max_iter : Integer. Maximum allowed number of iterations.

        min_iter : Integer. Minimum allowed number of iterations.
        """
        self.step = step
        self.output = output
        self.eps = eps
        self.max_iter = max_iter
        self.min_iter = min_iter

    def __call__(self, function, beta):
        """Call this object to obtain the variable that gives the minimum.

        Parameters
        ----------
        function : The function to minimise.

        beta : The start vector.
        """
        self.check_compatibility(function, self.INTERFACES)

        betanew = betaold = beta

        # TODO: Change the functions so that we can use the StepSize API here.
        if self.step is None:
            self.step = 1.0 / function.L()

        if self.output:
            t = []
            f = []
        for i in xrange(1, self.max_iter + 1):
            if self.output:
                tm = time_func()

            betaold = betanew
            betanew = function.prox(betaold -
                                    self.step * function.grad(betaold),
                                    self.step)

            if self.output:
                t.append(time_func() - tm)
                f.append(function.f(betanew))

            if (1.0 / self.step) * maths.norm(betanew - betaold) < self.eps \
                    and i >= self.min_iter:
                break

        if self.output:
            output = {"t": t, "f": f}
            return betanew, output
        else:
            return betanew


class FISTA(ExplicitAlgorithm):
    """ The fast iterative shrinkage threshold algorithm.

    Parameters
    ----------
    step : Step for each iteration

    output : Boolean. Get output information

    eps : Float. Tolerance

    max_iter : Maximum allowed number of iterations

    min_iter : Minimum allowed number of iterations

    Example
    -------
    import numpy as np
    import parsimony.estimators as estimators
    import parsimony.algorithms as algorithms
    import parsimony.tv
    from parsimony.functions import OLSL2_L1_TV
    from parsimony.algorithms import fista
    from parsimony.start_vectors import RandomStartVector

    shape = (100, 100, 1)
    num_samples = 500

    num_ft = shape[0] * shape[1] * shape[2]
    X = np.random.random((num_samples, num_ft))
    y = np.random.randint(0, 2, (num_samples, 1))
    random_start_vector = np.random.random((X.shape[1], 1))

    def ratio2coef(alpha, tv_ratio, l1_ratio):
        l2_ratio = 1 - tv_ratio - l1_ratio
        l, k, g = alpha * l1_ratio,  alpha * l2_ratio, alpha * tv_ratio
        return l, k, g

    eps = 0.01
    alpha = 10.

    tv_ratio = .05
    l1_ratio = .9

    l, k, g = ratio2coef(alpha=alpha, tv_ratio=tv_ratio, l1_ratio=l1_ratio)

    Ax, Ay, Az, n_compacts = parsimony.tv.tv_As_from_shape(shape)

    tvl1l2_conesta = estimators.LinearRegressionL1L2TV(k, l, g, [Ax, Ay, Az],
                                        algorithm=algorithms.conesta_static)
    tvl1l2_conesta.fit(X, y)

    tvl1l2_fista = estimators.LinearRegressionL1L2TV(k, l, g, [Ax, Ay, Az],
                                        algorithm=algorithms.fista)
    tvl1l2_fista.fit(X, y)

    residual = np.sum(tvl1l2_fista.beta - tvl1l2_conesta.beta)

    import spams
    spams_X = np.asfortranarray(X)
    spams_Y = np.asfortranarray(y)
    W0 = np.asfortranarray(np.random.random((spams_X.shape[1],
                                             spams_Y.shape[1])))
    spams_X = np.asfortranarray(spams_X - np.tile(np.mean(spams_X, 0),
                                                  (spams_X.shape[0], 1)))
    spams_Y = np.asfortranarray(spams_Y - np.tile(np.mean(spams_Y,0),
                                                         (spams_Y.shape[0],1)))
    param = {'numThreads' : 1,'verbose' : True,
         'lambda1' : 0.05, 'it0' : 10, 'max_it' : 200,
         'L0' : 0.1, 'tol' : 1e-3, 'intercept' : False,
         'pos' : False}
    (W, optim_info) = spams.fistaFlat(spams_Y,
                                      spams_X,
                                      W0,
                                      True,
                                      **param)

#    tvl1l2 = estimators.LinearRegressionL1L2TV(k, l, g, [Ax, Ay, Az],
#                                algorithm=algorithms.conesta_static)
#    tvl1l2.fit(X, y)
#    start_beta_vector = random_start_vector.get_vector([X.shape[1], 1])
#    fista(X, y, olsl2_L1_TV, start_beta_vector)

    """
    INTERFACES = [interfaces.Gradient,
                  # TODO: We should use a step size here instead of the
                  # Lipschitz constant. All functions don't have L, but will
                  # still run in FISTA with a small enough step size.
                  # Updated: Use StepSize instead!!
                  interfaces.LipschitzContinuousGradient,
                  interfaces.ProximalOperator,
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
        """Call this object to obtain the variable that gives the minimum.

        Parameters
        ----------
        function : The function to minimise.

        beta : A start vector.
        """
        self.check_compatibility(function, self.INTERFACES)

        z = betanew = betaold = beta

        # TODO: Change the functions so that we can use the StepSize API here.
        if self.step is None:
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

    Parameters
    ----------
    mu_start :

    mu_min :

    tau :

    dynamic : Boolean. Switch for dynamically or statically decreasing \mu

    continuations : maximum iteration

    """
    INTERFACES = [NesterovFunction,
                  interfaces.Continuation,
                  interfaces.LipschitzContinuousGradient,
                  interfaces.ProximalOperator,
                  interfaces.Gradient,
                  interfaces.DualFunction
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

        self.check_compatibility(function, self.INTERFACES)

        if self.mu_start is not None:
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
#                print "%f < %f" % ((1. / tmin) \
#                                * maths.norm(beta - beta_tilde), self.eps)
#                print "%d >= %d" % (i, self.continuations)
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

#            print "Gap:", G
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

        kwargs["dynamic"] = False

        super(StaticCONESTA, self).__init__(**kwargs)


class DynamicCONESTA(CONESTA):
    """COntinuation with NEsterov smoothing in a Soft-Thresholding Algorithm,
    or CONESTA for short, with a dynamically decreasing \mu.
    """
    def __init__(self, **kwargs):

        kwargs["dynamic"] = True

        super(DynamicCONESTA, self).__init__(**kwargs)


class ExcessiveGapMethod(ExplicitAlgorithm):
    """Nesterov's excessive gap method for strongly convex functions.
    """
    INTERFACES = [NesterovFunction,
                  interfaces.LipschitzContinuousGradient,
                  interfaces.GradientMap,
                  interfaces.DualFunction
                 ]

    def __init__(self, output=False,
                 eps=consts.TOLERANCE,
                 max_iter=consts.MAX_ITER, min_iter=1):
        """
        Parameters
        ----------
        output : Boolean. Get output information

        eps : Float. Tolerance

        max_iter : Maximum allowed number of iterations.

        min_iter : Minimum allowed number of iterations.
        """
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

        beta : The start vector. This is normally not given, but left None.
                The start vector is computed by the algorithm.
        """
        A = function.h.A()

        u = [0] * len(A)
        for i in xrange(len(A)):
            u[i] = np.zeros((A[i].shape[0], 1))

        # L = lambda_max(A'A) / (lambda_min(X'X) + k)
        L = function.L()

        mu = [2.0 * L]
        function.h.set_mu(mu)
        if beta is not None:
            beta0 = beta
        else:
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


class Bisection(ExplicitAlgorithm):
    """Finds a root of the function assumed to be on the line between two
    points.

    Assumes a function f(x) such that |f(x)|_2 < -eps if x is too small,
    |f(x)|_2 > eps if x is too large and |f(x)|_2 <= eps if x is just right.

    Parameters
    ----------
    force_negative : Boolean, default is False. Will try, by running more
            iterations, to make the result negative. It may fail, but it is
            unlikely.

    eps : A positive value such that |f(x)|_2 <= eps. Only guaranteed if
            |f(x)|_2 <= eps in less than maxiter iterations.

    max_iter : The maximum number of iterations.

    min_iter : The minimum number of iterations.
    """
    INTERFACES = [interfaces.Function,
                 ]

    def __init__(self, force_negative=False,
                 parameter_positive=True,
                 parameter_negative=True,
                 parameter_zero=True,
                 eps=consts.TOLERANCE,
                 max_iter=30, min_iter=1):

        self.force_negative = force_negative
        self.parameter_positive = parameter_positive
        self.parameter_negative = parameter_negative
        self.parameter_zero = parameter_zero
        self.eps = eps
        self.max_iter = max_iter
        self.min_iter = min_iter

    def __call__(self, function, x=None):
        """
        Parameters
        ----------
        function: The function for which a root is found. The function must be
                increasing for increasing x, and decresing for decreasing x.

        x: A vector or tuple with two elements. The first element is the lower
                end of the interval for which |f(x[0])|_2 < -eps. The second
                element is the upper end of the interfal for which
                |f(x[1])|_2 > eps. If x is None, these values are found.
                Finding them may be slow, though, if the function is expensive
                to evaluate.
        """
        if x is not None:
            low = x[0]
            high = x[1]
        else:
            if self.parameter_negative:
                low = -1.0
            elif self.parameter_zero:
                low = 0.0
            else:
                low = consts.TOLERANCE

            if self.parameter_positive:
                high = 1.0
            elif self.parameter_zero:
                high = 0.0
            else:
                high = -consts.TOLERANCE

        # Find start values. If the low and high
        # values are feasible this will just break
        for i in xrange(self.max_iter):
            f_low = function.f(low)
            f_high = function.f(high)
#            print "low :", low, ", f:", f_low
#            print "high:", high, ", f:", f_high

            if np.sign(f_low) != np.sign(f_high):
                break
            else:
                if self.parameter_positive \
                        and self.parameter_negative \
                        and self.parameter_zero:

                    low -= abs(low) * 2.0 ** i
                    high += abs(high) * 2.0 ** i

                elif self.parameter_positive \
                        and self.parameter_negative \
                        and not self.parameter_zero:

                    low -= abs(low) * 2.0 ** i
                    high += abs(high) * 2.0 ** i

                    if abs(low) < consts.TOLERANCE:
                        low -= consts.TOLERANCE
                    if abs(high) < consts.TOLERANCE:
                        high += consts.TOLERANCE

                elif self.parameter_positive \
                        and not self.parameter_negative \
                        and self.parameter_zero:

                    low /= 2.0
                    high *= 2.0

                elif self.parameter_positive \
                        and not self.parameter_negative \
                        and not self.parameter_zero:

                    low /= 2.0
                    high *= 2.0

                    if abs(low) < consts.TOLERANCE:
                        low = consts.TOLERANCE
                    if abs(high) < consts.TOLERANCE:
                        high = consts.TOLERANCE

                elif not self.parameter_positive \
                        and self.parameter_negative \
                        and self.parameter_zero:

                    low *= 2.0
                    high /= 2.0

                elif not self.parameter_positive \
                        and self.parameter_negative \
                        and not self.parameter_zero:

                    low *= 2.0
                    high /= 2.0

                    if abs(low) < consts.TOLERANCE:
                        low = -consts.TOLERANCE
                    if abs(high) < consts.TOLERANCE:
                        high = -consts.TOLERANCE

                elif not self.parameter_positive \
                        and not self.parameter_negative \
                        and self.parameter_zero:

                    low = 0.0
                    high = 0.0

                elif not self.parameter_positive \
                        and not self.parameter_negative \
                        and not self.parameter_zero:

                    raise ValueError("Parameter must be allowed to be real!")

        # Use the bisection method to find where |f(x)|_2 <= eps.
        neg_count = 0

        mid = (low + high) / 2.0
        f_mid = function.f(mid)
        for i in xrange(self.max_iter):
            if np.sign(f_mid) == np.sign(f_low):
                low = mid
                f_low = f_mid
            else:
                high = mid
                f_high = f_mid

            mid = (low + high) / 2.0
            f_mid = function.f(mid)
#            print "i:", (i + 1), ", mid: ", mid, ", f_mid:", f_mid

#            if np.sqrt(np.sum((high - low) ** 2.0)) <= self.eps:
            if abs(f_high - f_low) <= self.eps and i + 1 >= self.min_iter:
                if self.force_negative and f_mid > 0.0:
                    if neg_count < self.max_iter:
                        neg_count += 1
                    else:
                        break
                else:
                    break

        return mid


#class GeneralisedMultiblockISTA(ExplicitAlgorithm):
#    """ The iterative shrinkage threshold algorithm in a multiblock setting.
#    """
#    INTERFACES = [functions.MultiblockFunction,
#                  functions.MultiblockGradient,
#                  functions.MultiblockProximalOperator,
#                  functions.StepSize,
#                 ]
#
#    def __init__(self, step=None, output=False,
#                 eps=consts.TOLERANCE,
#                 max_iter=consts.MAX_ITER, min_iter=1):
#
#        self.step = step
#        self.output = output
#        self.eps = eps
#        self.max_iter = max_iter
#        self.min_iter = min_iter
#
#    def __call__(self, function, w):
#
#        self.check_compatability(function, self.INTERFACES)
#
#        for it in xrange(10):  # TODO: Get number of iterations!
#            print "it:", it
#
#            for i in xrange(len(w)):
#                print "  i:", i
#
#                for k in xrange(10000):
#                    print "    k:", k
#
#                    t = function.step(w, i)
#                    w[i] = w[i] - t * function.grad(w, i)
#                    w = function.prox(w, i, t)
##                    = w[:i] + [wi] + w[i+1:]
#
#                    print "    f:", function.f(w)
#
##                w[i] = wi
#
#        return w


class MultiblockProjectedGradientMethod(ExplicitAlgorithm):
    """ The projected gradient algorithm with alternating minimisations in a
    multiblock setting.
    """
    INTERFACES = [MultiblockFunction,
                  MultiblockGradient,
                  MultiblockProjectionOperator,
                  interfaces.StepSize]

    def __init__(self, step=None, output=False,
                 eps=consts.TOLERANCE,
                 outer_iter=25, max_iter=consts.MAX_ITER, min_iter=1):

        self.step = step
        self.output = output
        self.eps = eps
        self.outer_iter = outer_iter
        self.max_iter = max_iter
        self.min_iter = min_iter

    def __call__(self, function, w):

        self.check_compatibility(function, self.INTERFACES)

        print "outer_iter:", self.outer_iter
        print "len(w):", len(w)
        print "max_iter:", self.max_iter

#        z = w_old = w

        if self.output:
            f = [function.f(w)]

        t = [1.0] * len(w)

        for it in xrange(self.outer_iter):  # TODO: Get number of iterations!
            all_converged = True
            for i in xrange(len(w)):
                converged = False
                print "it: %d, i: %d" % (it, i)
                for k in xrange(self.max_iter):
#                    print "it: %d, i: %d, k: %d" % (it, i, k)

#                    z = w[i] + ((k - 2.0) / (k + 1.0)) * (w[i] - w_old[i])

                    w_old = copy.deepcopy(w)

#                    _t = time()
                    t[i] = function.step(w_old, i)
#                    print "t:", t[i]
#                    print "step:", time() - _t

#                    _t = time()
                    grad = function.grad(w_old, i)
                    w[i] = w_old[i] - t[i] * grad

#                    def fun(x):
#                        w_ = [0, 0]
#                        w_[i] = x
#                        w_[1 - i] = w[1 - i]
#                        return function.f(w_)
#                    approx_grad = utils.approx_grad(fun, w[i], eps=1e-6)
#                    diff = float(maths.norm(grad - approx_grad))
#                    print "grad err: %e, lim: %e" % (diff, 5e-5)
#                    if diff > 5e-4:
#                        pass

#                    w[i] = z[i] - t[i] * function.grad(w_old[:i] +
#                                                       [z] +
#                                                       w_old[i + 1:], i)
#                    print "grad:", time() - _t

#                    _t = time()
                    w = function.proj(w, i)
#                    print "proj:", time() - _t

#                    print "l0 :", maths.norm0(w[i]), \
#                        ", l1 :", maths.norm1(w[i]), \
#                        ", l2²:", maths.norm(w[i]) ** 2.0

                    if self.output:
                        f_ = function.f(w)
#                        print "f:", f_
                        improvement = f_ - f[-1]
                        if improvement > 0.0:
                            # If this happens there are two possible reasons:
                            if abs(improvement) <= consts.TOLERANCE:
                                # 1. The function is actually converged, and
                                #         the "increase" is because of
                                #         precision errors. This happens
                                #         sometimes.
                                pass
                            else:
                                # 2. There is an error and the function
                                #         actually increased. Does this
                                #         happen? If so, we need to
                                #         investigate! Possible errors are:
                                #          * The gradient is wrong.
                                #          * The step size is too large.
                                #          * Other reasons?
                                print "ERROR! Function increased!"

                            # Either way, we stop and regroup if it happens.
                            break

                        f.append(f_)

                    err = maths.norm(w_old[i] - w[i])
#                    print "err: %.10f < %.10f * %.10f = %.10f" \
#                        % (err, t[i], self.eps, t[i] * self.eps)
                    if err <= t[i] * self.eps and k + 1 >= self.min_iter:
                        converged = True
                        break

                print "l0 :", maths.norm0(w[i]), \
                    ", l1 :", maths.norm1(w[i]), \
                    ", l2²:", maths.norm(w[i]) ** 2.
                print "f:", f[-1]

                if not converged:
                    all_converged = False

            if all_converged:
                print "All converged!"
                break

        if self.output:
#            output = {"t": t, "f": f}
            output = {"f": f}
            return (w, output)
        else:
            return w


class ProjectionADMM(ExplicitAlgorithm):
    """ The Alternating direction method of multipliers, where the functions
    have projection operators onto the corresponding convex sets.
    """
    INTERFACES = [interfaces.Function,
                  interfaces.ProjectionOperator]

    def __init__(self, output=False,
                 eps=consts.TOLERANCE,
                 max_iter=consts.MAX_ITER, min_iter=1):

        self.output = output
        self.eps = eps
        self.max_iter = max_iter
        self.min_iter = min_iter

    def __call__(self, function, x):
        """Finds the projection onto the intersection of two sets.

        Parameters:
        ----------
        function: List or tuple with two elements. The two functions.

        x: The point that we wish to project.
        """
        self.check_compatibility(function[0], self.INTERFACES)
        self.check_compatibility(function[1], self.INTERFACES)

        z = x
        u = np.zeros(x.shape)
        for i in xrange(1, self.max_iter + 1):
            x = function[0].proj(z - u)
            z = function[1].proj(x + u)
            u = u + x - z

            if maths.norm(z - x) / maths.norm(z) < self.eps \
                    and i >= self.min_iter:
                break

        return z


class DykstrasProjectionAlgorithm(ExplicitAlgorithm):
    """Dykstra's projection algorithm. Computes the projection onto the
    intersection of two convex sets.

    The functions have projection operators onto the corresponding convex sets.
    """
    INTERFACES = [interfaces.Function,
                  interfaces.ProjectionOperator]

    def __init__(self, output=False,
                 eps=consts.TOLERANCE,
                 max_iter=consts.MAX_ITER, min_iter=1):

        self.output = output
        self.eps = eps
        self.max_iter = max_iter
        self.min_iter = min_iter

    def __call__(self, function, r):
        """Finds the projection onto the intersection of two sets.

        Parameters:
        ----------
        function: List or tuple with two elements. The two functions.

        r: The point that we wish to project.
        """
        self.check_compatibility(function[0], self.INTERFACES)
        self.check_compatibility(function[1], self.INTERFACES)

        x_new = r
        p_new = np.zeros(r.shape)
        q_new = np.zeros(r.shape)
        for i in xrange(1, self.max_iter + 1):

            x_old = x_new
            p_old = p_new
            q_old = q_new

            y_old = function[0].proj(x_old + p_old)
            p_new = x_old + p_old - y_old
            x_new = function[1].proj(y_old + q_old)
            q_new = y_old + q_old - x_new

            if maths.norm(x_new - x_old) / maths.norm(x_old) < self.eps \
                    and i >= self.min_iter:
                break

        return x_new


class BacktrackingLineSearch(ExplicitAlgorithm):
    INTERFACES = [interfaces.Function,
                  interfaces.Gradient]

    def __init__(self, condition=None,
                 output=False,
                 max_iter=30, min_iter=1,
                 eps=consts.TOLERANCE):  # Note that tolerance is never used!
        """Finds a step length a that fulfills a given descent criterion.

        Parameters:
        ----------
        condition : The class of the descent condition. If not given, defaults
                to the StrongWolfeCondition.

        output : Boolean. Whether or not to return additional output.

        max_iter : The maximum allowed number of iterations.

        max_iter : The minimum number of iterations that must be made.
        """
        self.condition = condition
        if self.condition is None:
            self.condition = penalties.SufficientDescentCondition
        self.output = output
        self.max_iter = max_iter
        self.min_iter = min_iter

    def __call__(self, function, x, p, rho=0.5, a=1.0, **kwargs):
        """Finds the step length for a descent algorithm.

        Parameters:
        ----------
        function : A Loss function. The function to minimise.

        x : Vector. The current point.

        p : Vector. The descent direction.

        rho : Float. 0 < rho < 1. The rate at which to decrease a in each
                iteration. Smaller will finish faster, but may yield a lesser
                descent.

        a : Float. The upper bound on the step length. Defaults to 1, which is
                suitable for e.g. Newton's method.

        kwargs : Parameters for the descent condition.
        """
        self.check_compatibility(function, self.INTERFACES)

        line_search = self.condition(function, p, **kwargs)
        it = 0
        while True:
            if line_search.feasible(x, a):
                print "Broke after %d iterations of %d iterations." \
                    % (it, self.max_iter)
                return a

            it += 1
            if it >= self.max_iter:
                return 0.0  # If we did not find a feasible point, don't move!

            a = a * rho


if __name__ == "__main__":
    import doctest
    doctest.testmod()