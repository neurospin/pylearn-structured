# -*- coding: utf-8 -*-
"""
The :mod:`strukturerad.models` module includes several different models.

@author:  Tommy Löfstedt <tommy.loefstedt@cea.fr>
@email:   tommy.loefstedt@cea.fr
@license: TBD
"""

__all__ = ['ContinuationRun',
           'Continuation',

           'NesterovProximalGradientMethod',
           'LinearRegression',
           'LinearRegressionL1',
           'LinearRegressionTV',
           'LinearRegressionL1TV',

           'RidgeRegression',
           'RidgeRegressionL1',
           'RidgeRegressionTV',
           'RidgeRegressionL1TV',

           'ExcessiveGapMethod',
           'EGMRidgeRegression',
           'EGMRidgeRegressionL1',
           'EGMRidgeRegressionTV',
           'EGMRidgeRegressionL1TV',
          ]

import abc
import numpy as np

import algorithms
import loss_functions
import start_vectors

import utils


class BaseModel(object):

    __metaclass__ = abc.ABCMeta

    def __init__(self, algorithm):

        super(BaseModel, self).__init__()

        self.algorithm = algorithm

    def get_max_iter(self):

        return self.get_algorithm().get_max_iter()

    def set_max_iter(self, max_iter):

        self.get_algorithm().set_max_iter(max_iter)

    def get_tolerance(self):

        return self.get_algorithm().get_tolerance()

    def set_tolerance(self, tolerance):

        self.get_algorithm().set_tolerance(tolerance)

    def get_start_vector(self):

        return self.get_algorithm().get_start_vector()

    def set_start_vector(self, start_vector):

        self.get_algorithm().set_start_vector(start_vector)

    def get_algorithm(self):

        return self.algorithm

    def set_algorithm(self, algorithm):

        if not isinstance(self.algorithm, algorithms.BaseAlgorithm):

            raise ValueError('The algorithm must be an instance of ' \
                             '"BaseAlgorithm"')

        self.algorithm = algorithm

    @abc.abstractmethod
    def get_transform(self, index=0):

        raise NotImplementedError('Abstract method "get_transform" must be '\
                                  'specialised!')

    @abc.abstractmethod
    def fit(self, *X, **kwargs):

        raise NotImplementedError('Abstract method "fit" must be specialised!')


class ContinuationFixed(BaseModel):

    def __init__(self, model, gaps=None, mus=None, algorithm=None,
                 *args, **kwargs):
        """Performs continuation for the given method. I.e. runs the method
        with sucessively smaller values of mu and uses the output from the
        use of one mu as start vector in the run with the next smaller mu.

        Parameters
        ----------
        model : The NesterovProximalGradient model to perform continuation
                on.

        gaps : A list of successively smaller gap values. The gaps are used as
                terminating condition for the continuation run. Mu is computed
                from this list of gaps. Note that only one of gaps and mus can
                be given.

        mus : A list of successively smaller values of mu, the regularisation
                parameter in the Nesterov smoothing. The gaps are
                computed from this list of mus. Note that only one of mus and
                gaps can be given.

        algorithm : The particular algorithm to use.
        """
        if algorithm == None:
            algorithm = model.get_algorithm()
        else:
            model.set_algorithm(algorithm)

        super(ContinuationFixed, self).__init__(num_comp=1,
                                                algorithm=algorithm,
                                                *args, **kwargs)
        self.model = model
        self.gaps = gaps
        self.mus = mus

    def get_transform(self, index=0):

        return self._beta

    def get_algorithm(self):

        return self.model.get_algorithm()

    def set_algorithm(self, algorithm):

        self.model.set_algorithm(algorithm)

    def fit(self, X, y, **kwargs):

        start_vector = self.model.get_start_vector()
        f = []
        self.model.set_data(X, y)

        if self.mus != None:
            lst = self.mus
        else:
            lst = self.gaps

        beta_new = 0

        for item in lst:
            if self.mus != None:
                self.model.set_mu(item)
#                self.model.set_tolerance(self.model.compute_gap(item))
            else:
#                self.model.set_tolerance(item)
                self.model.set_mu(self.model.compute_mu(item))

            self.model.set_start_vector(start_vector)
            self.model.fit(X, y, **kwargs)

            utils.debug("Continuation with mu = ", self.model.get_mu(), \
                    ", tolerance = ", self.model.get_tolerance(), \
                    ", iterations = ", self.model.get_algorithm().iterations)

            beta_old = beta_new
            beta_new = self.model.get_transform()
            f = f + self.model.get_algorithm().f[1:]  # Skip the first, same

#            if len(f) > 1 and abs(f[-2] - f[-1]) < self.model.get_tolerance():
#                print "Converged in f!!"
#                break

            if utils.norm1(beta_old - beta_new) < self.model.get_tolerance():
                print "Converged in beta!!"
                print utils.norm1(beta_old - beta_new)
                print self.model.get_tolerance()
                break

            start_vector = start_vectors.IdentityStartVector(beta_new)

        self._beta = beta_new
        self.model.get_algorithm().f = f
        self.model.get_algorithm().iterations = len(f)

        return self


class ContinuationGap(BaseModel):

    def __init__(self, model, iterations=100, gap=None, algorithm=None,
                 *args, **kwargs):
        """Performs continuation for the given model. I.e. builds
        NesterovProximalGradientMethod models with sucessively, and optimally,
        smaller values of mu and uses the output from the use of one mu as
        start vector in the fit of model with the next smaller mu.

        Parameters
        ----------
        model : The NesterovProximalGradient model to perform continuation
                on.

        iterations : The number of iterations in each continuation.

        gap : The gap to use in the first continuation. Default is
                mu = max(abs(cov(X,y))) and then
                gap = model.compute_gap(mu).

        algorithm : The particular algorithm to use.
        """
        if algorithm == None:
            algorithm = model.get_algorithm()
        else:
            model.set_algorithm(algorithm)

        super(ContinuationGap, self).__init__(num_comp=1, algorithm=algorithm,
                                              *args, **kwargs)

        self.model = model
        self.iterations = iterations
        self.gap = gap

    def get_transform(self, index=0):

        return self._beta

    def get_algorithm(self):

        return self.model.get_algorithm()

    def set_algorithm(self, algorithm):

        self.model.set_algorithm(algorithm)

    def fit(self, X, y, **kwargs):

        max_iter = self.get_max_iter()
        self.model.set_max_iter(self.iterations)
        self.model.set_data(X, y)
        start_vector_mu = self.model.get_start_vector()
#        start_vector_nomu = self.model.get_start_vector()
        if self.gap == None:
            mu = max(np.max(np.abs(utils.corr(X, y))), 0.01)  # Necessary?
            gap_mu = self.model.compute_gap(mu)
        else:
            gap_mu = self.gap
            mu = self.model.compute_mu(gap_mu)

        gap_nomu = gap_mu
        beta_old = 0.0
        beta_new = 0.0

        tau = 1.1
        eta = 2.0
        mu_zero = 5e-12

        f = []
        for i in xrange(1, max_iter + 1):

#            self.model.set_max_iter(float(self.iterations) / float(i))

            # With computed mu
            self.model.set_mu(mu)
            self.model.set_start_vector(start_vector_mu)
            self.model.fit(X, y, **kwargs)
            f = f + self.model.get_algorithm().f[1:]  # Skip the first, same
            beta_old = beta_new
            beta_new = self.model.get_transform()
            start_vector_mu = start_vectors.IdentityStartVector(beta_new)

            self.model.set_start_vector(start_vector_mu)
            alpha_mu = self.model.get_g().alpha()
            gap_mu = self.model.phi(beta=beta_new, alpha=alpha_mu) \
                        - self.model.phi(beta=None, alpha=alpha_mu)

            utils.debug("With mu: Continuation with mu = ",
                                self.model.get_mu(), \
                    ", tolerance = ", self.model.get_tolerance(), \
                    ", iterations = ", self.model.get_algorithm().iterations, \
                    ", gap = ", gap_mu)

            # With mu "very small"
            self.model.set_mu(min(mu, mu_zero))
            alpha_nomu = self.model.get_g().alpha(beta_new, min(mu, mu_zero))
            gap_nomu = self.model.phi(beta=beta_new, alpha=alpha_nomu) \
                        - self.model.phi(beta=None, alpha=alpha_nomu,
                                         mu=min(mu, mu_zero))

            utils.debug("No mu: Continuation with mu = ",
                                self.model.get_mu(), \
                    ", tolerance = ", self.model.get_tolerance(), \
                    ", iterations = ", self.model.get_algorithm().iterations, \
                    ", gap = ", gap_nomu)

            if gap_nomu < self.model.get_tolerance():
                print "Converged in G!!"
                break

#            if len(f) > 1 and abs(f[-2] - f[-1]) < self.model.get_tolerance():
#                print "Converged in f!!"
#                break

            if utils.norm1(beta_old - beta_new) < self.model.get_tolerance():
                print "Converged in beta!!"
                break

            self.model.set_mu(mu)
            mu = min(mu, self.model.compute_mu(gap_nomu))
            if gap_mu < gap_nomu / (2.0 * tau):
                mu = mu / eta

            mu = max(mu, utils.TOLERANCE)

        self._beta = beta_new

        self.model.get_algorithm().f = f
        self.model.get_algorithm().iterations = len(f)

        self.model.set_max_iter(max_iter)

        return self


class NesterovProximalGradientMethod(BaseModel):

    def __init__(self, algorithm=None, **kwargs):

        if algorithm == None:
#            algorithm = algorithms.ISTARegression()
#            algorithm = algorithms.FISTARegression()
            algorithm = algorithms.MonotoneFISTARegression()

        super(NesterovProximalGradientMethod,
              self).__init__(algorithm=algorithm, **kwargs)

        self.set_g(loss_functions.ZeroErrorFunction())
        self.set_h(loss_functions.ZeroErrorFunction())

    def get_g(self):

        return self.g

    def set_g(self, g):

        self.g = g

    def get_h(self):

        return self.h

    def set_h(self, h):

        self.h = h

    def fit(self, X, y, **kwargs):
        """Fit the model to the given data.

        Parameters
        ----------
        X : The independent variables.

        y : The dependent variable.

        Returns
        -------
        self: The model object.
        """
        X, y = utils.check_arrays(X, y)

        self.set_data(X, y)

        self._beta = self.algorithm.run(X, y, self.get_g(), self.get_h(),
                                        **kwargs)

        self.free_data()

        return self

    def f(self, *args, **kwargs):

        return self.get_g().f(*args, **kwargs) \
                + self.get_h().f(*args, **kwargs)

    def phi(self, beta=None, alpha=None, *args, **kwargs):
        """This function returns the associated loss function value for the
        given alpha and beta.
        """
        return self.get_g().phi(beta, alpha) + self.get_h().f(beta)

    def alpha(self, beta=None, mu=None):
        """Computes the alpha that maximises the smoothed loss function for the
        current computed beta.
        """
        g = self.get_g()

        return g.alpha(beta, mu)

    def beta(self):

        return self._beta

    def get_transform(self, **kwargs):

        return self._beta

    def compute_gap(self, mu, max_iter=100):

        def f(eps):
            return self.compute_mu(eps) - mu

        D = self.get_g().num_compacts() / 2.0
        lb = 2.0 * D * mu
        ub = lb * 100.0
        a = f(lb)
        b = f(ub)

        # Do a binary search for the upper limit. It seems that when mu become
        # very small (e.g. < 1e-8), the lower bound is not correct. Therefore
        # we do a full search for both the lower and upper bounds.
        for i in xrange(max_iter):

            if a > 0.0 and b > 0.0 and a < b:
                ub = lb
                lb /= 2.0
            elif a < 0.0 and b < 0.0 and a < b:
                lb = ub
                ub *= 2.0
            elif a > 0.0 and b > 0.0 and a > b:
                lb = ub
                ub *= 2.0
            elif a < 0.0 and b < 0.0 and a > b:
                ub = lb
                lb /= 2.0
            else:
                break

            a = f(lb)
            b = f(ub)

        bm = algorithms.BisectionMethod(max_iter=max_iter)
        bm.run(utils.AnonymousClass(f=f), lb, ub)

        return bm.x

    def compute_mu(self, eps):

        g = self.get_g()
        D = g.num_compacts() / 2.0

        def f(mu):
            return -(eps - mu * D) / g.Lipschitz(mu)

        gs = algorithms.GoldenSectionSearch()
        gs.run(utils.AnonymousClass(f=f), utils.TOLERANCE, eps / (2.0 * D))

#        ts = algorithms.TernarySearch(utils.AnonymousClass(f=f))
#        ts.run(utils.TOLERANCE, eps / D)

        return gs.x

    def predict(self, X, **kwargs):

        yhat = np.dot(X, self.get_transform())

        return yhat

    def set_mu(self, mu):

        self.get_g().set_mu(mu)

    def get_mu(self):

        return self.get_g().get_mu()

    def set_data(self, X, y):

        self.get_g().set_data(X, y)

    def get_data(self):

        return self.get_g().get_data()

    def free_data(self):

        return self.get_g().free_data()


class LinearRegression(NesterovProximalGradientMethod):
    """Linear regression.

    Optimises the function

        f(b) = (1 / 2).||X.b - y||².
    """
    def __init__(self, **kwargs):

        super(LinearRegression, self).__init__(**kwargs)

        self.set_g(loss_functions.LinearRegressionError())


class LinearRegressionL1(LinearRegression):
    """Linear regression with an L1 constraint.

    Optimises the function

        f(b) = (1 / 2).||X.b - y||² + l.||b||_1,

    where ||.||_1 is the L1 norm constraint.

    Parameters
    ----------
    l    : The L1 parameter.

    mu   : The Nesterov function regularisation parameter.
    """
    def __init__(self, l, **kwargs):

        super(LinearRegressionL1, self).__init__(**kwargs)

        self.set_h(loss_functions.L1(l))


class LinearRegressionTV(NesterovProximalGradientMethod):
    """Linear regression with total variation constraint.

    Optimises the function

        f(b) = (1 / 2).||X.b - y||² + gamma.TV(b),

    where TV(.) is the total variation constraint.

    Parameters
    ----------
    gamma: The TV regularisation parameter.

    shape: The shape of the 3D image. Must be a 3-tuple. If the image is 2D,
           let the Z dimension be 1, and if the "image" is 1D, let the Y and
           Z dimensions be 1. The tuple must be on the form (Z, Y, X).

    mu   : The Nesterov function regularisation parameter.

    mask : A 1-dimensional mask representing the 3D image mask. Must be a
           list of 1s and 0s.
    """
    def __init__(self, gamma, mu=None, shape=None, A=None, mask=None,
                 **kwargs):

        super(LinearRegressionTV, self).__init__(**kwargs)

        lr = loss_functions.LinearRegressionError()
        tv = loss_functions.TotalVariation(gamma, mu=mu, shape=shape, A=A,
                                           mask=mask)

        self.set_g(loss_functions.CombinedNesterovLossFunction(lr, tv))

    def compute_mu(self, eps):

        g = self.get_g()
        lr = g.a
        tv = g.b

        D = tv.num_compacts() / 2.0
        A = tv.Lipschitz(1.0)
        l = lr.Lipschitz()

        return (-2.0 * D * A + np.sqrt((2.0 * D * A) ** 2.0 \
                + 4.0 * D * l * eps * A)) / (2.0 * D * l)

    def compute_gap(self, mu, max_iter=100):

        g = self.get_g()
        lr = g.a
        tv = g.b

        D = tv.num_compacts() / 2.0
        A = tv.Lipschitz(1.0)
        l = lr.Lipschitz()

        return ((2.0 * mu * D * l + 2.0 * D * A) ** 2.0 \
                - (2.0 * D * A) ** 2.0) / (4.0 * D * l * A)


class LinearRegressionL1TV(LinearRegressionTV):
    """Linear regression with total variation and L1 constraints.

    Optimises the function

        f(b) = (1 / 2).||X.b - y||² + l.||b||_1 + gamma.TV(b),

    where ||.||_1 is the L1 norm, ||.||² is the squared L2 norm and TV(.) is
    the total variation constraint.

    Parameters
    ----------
    l    : The L1 parameter.

    gamma: The TV regularisation parameter.

    shape: The shape of the 3D image. Must be a 3-tuple. If the image is 2D,
           let the Z dimension be 1, and if the "image" is 1D, let the Y and
           Z dimensions be 1. The tuple must be on the form (Z, Y, X).

    mu   : The Nesterov function regularisation parameter.

    mask : A 1-dimensional mask representing the 3D image mask. Must be a
           list of 1s and 0s.
    """

    def __init__(self, l, gamma, mu=None, shape=None, A=None, mask=None,
                 **kwargs):

        super(LinearRegressionL1TV, self).__init__(gamma, mu=mu, shape=shape,
                                                   A=A, mask=mask, **kwargs)
        self.set_h(loss_functions.L1(l))


class RidgeRegression(NesterovProximalGradientMethod):
    """Ridge regression.

    Optimises the function

        f(b) = (1 / 2).||X.b - y||² + (l / 2).||b||²

    Parameters
    ----------
    l : The ridge parameter.
    """
    def __init__(self, l, **kwargs):

        super(RidgeRegression, self).__init__(**kwargs)

        self.set_g(loss_functions.RidgeRegression(l))


class RidgeRegressionL1(RidgeRegression):
    """Ridge regression with L1 regularisation, i.e. linear regression with L1
    and L2 constraints.

    Optimises the function

        f(b) = (1 / 2).||X.b - y||² + l.||b||_1 + (k / 2).||b||²,

    where ||.||_1 is the L1 norm and ||.||² is the squared L2 norm.

    Parameters
    ----------
    l : The L1 parameter.

    k : The L2 parameter.
    """
    def __init__(self, l, k, **kwargs):

        super(RidgeRegressionL1, self).__init__(k, **kwargs)

        self.set_h(loss_functions.L1(l))


class RidgeRegressionTV(RidgeRegression):
    """Ridge regression with total variation constraint, i.e. linear regression
    with L2 and TV constraints.

    Optimises the function

        f(b) = (1 / 2).||X.b - y||² + (l / 2).||b||² + gamma.TV(b),

    where ||.||² is the squared L2 norm and TV(.) is the total variation
    constraint.

    Parameters
    ----------
    l: The L1 regularisation parameter.

    gamma: The TV regularisation parameter.

    shape: The shape of the 3D image. Must be a 3-tuple. If the image is 2D,
           let the Z dimension be 1, and if the "image" is 1D, let the Y and
           Z dimensions be 1. The tuple must be on the form (Z, Y, X).

    mu   : The Nesterov function regularisation parameter.

    mask : A 1-dimensional mask representing the 3D image mask. Must be a
           list of 1s and 0s.
    """
    def __init__(self, l, gamma, mu=None, shape=None, A=None, mask=None,
                 compress=True, **kwargs):

        super(RidgeRegressionTV, self).__init__(l, **kwargs)

        tv = loss_functions.TotalVariation(gamma, mu=mu, shape=shape, A=A,
                                           mask=mask, compress=compress,
                                           **kwargs)
        rr = self.get_g()
        self.set_g(loss_functions.CombinedNesterovLossFunction(rr, tv))


class RidgeRegressionL1TV(RidgeRegressionTV):
    """Ridge regression with L1 and Total variation regularisation, i.e. linear
    regression with L1, L2 and TV constraints.

    Optimises the function

        f(b) = (1 / 2).||X.b - y||² + l.||b||_1 + (k / 2).||b||² + gamma.TV(b),

    where ||.||_1 is the L1 norm, ||.||² is the squared L2 norm and TV is the
    total variation function.

    Parameters
    ----------
    l : The L1 regularisation parameter.

    k : The L2 regularisation parameter.

    gamma : The TV regularisation parameter.

    mu : The Nesterov function regularisation parameter.

    mask : A 1-dimensional mask representing the 3D image mask. Must be a
           list of 1s and 0s.
    """
    def __init__(self, l, k, gamma, mu=None, shape=None, A=None, mask=None,
                 compress=True, **kwargs):

        super(RidgeRegressionL1TV, self).__init__(k, gamma, mu=mu, shape=shape,
                                                  A=A, mask=mask,
                                                  compress=compress, **kwargs)
        self.set_h(loss_functions.L1(l))

        self._l1 = loss_functions.SmoothL1(k, num_variables=np.prod(shape),
                                           mu=1e-12, mask=mask)
        # TODO: Reuse the A matrices from self.get_g().b
        self._tv = loss_functions.TotalVariation(gamma, shape=shape, mu=mu,
                                                 mask=mask, compress=False)

    # TODO: Decide if phi(beta, alpha) should be in the general API for all
    # Nesterov functions.
    def phi(self, beta=None, alpha=None, mu=None, *args, **kwargs):
        """This function returns the associated loss function value for the
        given alpha and beta.

        If alpha or beta is not given, they are computed.
        """
        if mu == None:
            mu = self.get_mu()

        if alpha == None:

            alpha = self.get_g().alpha(beta)

        elif beta == None:

            rr = self.get_g().a
            X, y = self.get_data()

            #####
            mu_zero = min(mu, 1e-12)
#            beta_ = self._beta
#            mask_ = tv.get_mask()
#            shape_ = tv.get_shape()

            alpha_l1_ = self._l1.alpha(self._beta, mu=mu_zero)
            alpha_tv_ = alpha  # self._tv.alpha(self._beta, mu=mu)
            Aa_l1_ = self._l1.grad(self._beta, alpha=alpha_l1_, mu=mu_zero)
            Aa_tv_ = self._tv.grad(self._beta, alpha=alpha_tv_, mu=mu)
            Aa_ = Aa_l1_ + Aa_tv_
            if not hasattr(self, '_XtinvXXtlI'):
#                XtX_ = np.dot(X.T, X)
#                self._invXtXlI = np.linalg.inv(XtX_ \
#                                                + rr.l * np.eye(*XtX_.shape))
                invXXtlI = np.linalg.inv(np.dot(X, X.T) \
                                            + rr.l * np.eye(X.shape[0]))
                self._XtinvXXtlI = np.dot(X.T, invXXtlI)
                self._Xty = np.dot(y.T, X).T

            wk_ = (self._Xty - Aa_) / rr.l
            beta = wk_ - np.dot(self._XtinvXXtlI, np.dot(X, wk_))

#            beta_ = np.dot(self._invXtXlI, np.dot(X.T, y) - Aa_)
#            beta = beta_

            return rr.f(beta) \
                    + self._l1.phi(beta, alpha_l1_) \
                    + self._tv.phi(beta, alpha_tv_)
                    # TODO: NOT WORKING!! _tv.phi(beta, alpha_tv_) is negative!
            #####

#            tv = self.get_g().b
#            Aa = tv.grad() <-- WARNING!
#            tv = loss_functions.LinearLossFunction(Aa)
#            dual_model = NesterovProximalGradientMethod()
#            dual_model.set_start_vector(self.get_start_vector())
#            dual_model.set_max_iter(self.get_max_iter())
#            dual_model.set_g(loss_functions.CombinedNesterovLossFunction(rr,
#                                                                         tv))
#            dual_model.set_h(self.get_h())
#
#            dual_model.fit(X, y, early_stopping=False)
#            beta = dual_model._beta

#            print "diff:", np.sum((beta_ - beta) ** 2.0)

        return self.get_g().phi(beta, alpha) + self.get_h().f(beta)

#    def beta(self, alpha=None, mu=None):
#        """Computes the beta that minimises the dual function value for the
#        current computed or given alpha.
#        """
#        raise ValueError("Do not call this function!")
##        return self._rr_l1_tv.beta(alpha=alpha, mu=mu)

#    def alpha(self, beta=None, mu=None):
#        """Computes the alpha that maximises the smoothed loss function for the
#        current computed beta.
#        """
#        raise ValueError("Do not call this function!")
##        return self._rr_l1_tv.alpha(beta=beta, mu=mu)

    def set_data(self, X, y):

        super(RidgeRegressionL1TV, self).set_data(X, y)

#        # We will need to recompute this matrix
#        if hasattr(self, "_XtinvXXtlI"):
#            del self._XtinvXXtlI


class ExcessiveGapMethod(BaseModel):

    __metaclass__ = abc.ABCMeta

    def __init__(self, algorithm=None, **kwargs):

        if algorithm == None:
            algorithm = algorithms.ExcessiveGapRidgeRegression()

        super(ExcessiveGapMethod, self).__init__(num_comp=1,
                                                 algorithm=algorithm,
                                                 **kwargs)

    def fit(self, X, y, **kwargs):
        """Fit the model to the given data.

        Parameters
        ----------
        X : The independent variables.

        y : The dependent variable.

        Returns
        -------
        self: The model object.
        """
        X, y = utils.check_arrays(X, y)

        self.set_data(X, y)

        self._beta = self.algorithm.run(X, y, **kwargs)

        return self

    def get_transform(self, index=0):

        return self._beta

    def predict(self, X, **kwargs):

        yhat = np.dot(X, self.get_transform(**kwargs))

        return yhat

    def get_g(self):

        return self.algorithm.g

    def set_g(self, g):

        self.algorithm.g = g

    def get_h(self):

        return self.algorithm.h

    def set_h(self, h):

        self.algorithm.h = h

    def set_data(self, X, y):

        self.get_g().set_data(X, y)

    def get_data(self):

        return self.get_g().get_data()


class EGMRidgeRegression(ExcessiveGapMethod):
    """Linear regression with L2 regularisation. Uses the excessive gap method.

    Optimises the function

        f(b) = ||y - X.b||² + (k / 2.0).||b||²,

    where ||.||² is the squared L2 norm.

    Parameters
    ----------
    l : The L2 parameter.
    """
    def __init__(self, l, **kwargs):

        super(EGMRidgeRegression, self).__init__(**kwargs)

        self.set_g(loss_functions.RidgeRegression(l))
        self.set_h(loss_functions.SmoothL1(0.0, 1, 0.0))  # We're cheating! ;-)


class EGMRidgeRegressionL1(ExcessiveGapMethod):
    """Linear Regression with L1 and L2 regularisation. Uses the excessive
    gap method.

    Optimises the function

        f(b) = ||y - X.b||² + l.||b||_1 + (k / 2.0).||b||²,

    where ||.||_1 is the L1 norm and ||.||² is the squared L2 norm.

    Parameters
    ----------
    l     : The L1 parameter.

    k     : The L2 parameter.

    p     : The numbers of variables.

    mask  : A 1-dimensional mask representing the 3D image mask. Must be a
            list of 1s and 0s.
    """
    def __init__(self, l, k, p, mask=None, **kwargs):

        super(EGMRidgeRegressionL1, self).__init__(**kwargs)

        self.set_g(loss_functions.RidgeRegression(k))
        self.set_h(loss_functions.SmoothL1(l, p, mask=mask))


class EGMRidgeRegressionTV(ExcessiveGapMethod):
    """Ridge regression with total variation constraint.

    Optimises the function

        f(b) = ||y - X.b||² + l.||b||² + gamma.TV(b),

    where ||.||² is the squared L2 norm and TV(.) is the total variation
    constraint.

    Parameters
    ----------
    l     : The ridge parameter.

    gamma : The TV regularisation parameter.

    shape : The shape of the 3D image. Must be a 3-tuple. If the image is
            2D, let the Z dimension be 1, and if the "image" is 1D, let the
            Y and Z dimensions be 1. The tuple must be on the form
            (Z, Y, X).

    mask  : A 1-dimensional mask representing the 3D image mask. Must be a
           list of 1s and 0s.
    """
    def __init__(self, l, gamma, shape, mask=None, **kwargs):

        super(EGMRidgeRegressionTV, self).__init__(**kwargs)

        self.set_g(loss_functions.RidgeRegression(l))
        self.set_h(loss_functions.TotalVariation(gamma, shape=shape,
                                                 mask=mask))


class EGMRidgeRegressionL1TV(ExcessiveGapMethod):
    """Linear regression with L1, L2 and total variation constraints.

    Optimises the function

        f(b) = ||y - X.b||² + l.||b||_1 + (k / 2.0).||b||² + gamma.TV(b),

    where ||.||_1 is the L1 norm, ||.||² is the squared L2 norm and TV(.) is
    the total variation constraint.

    Parameters
    ----------
    l     : The ridge regularisation parameter. Must be in the interval [0,1].

    k     : The L2 parameter.

    gamma : The TV regularisation parameter.

    shape : The shape of the 3D image. Must be a 3-tuple. If the image is
            2D, let the Z dimension be 1, and if the "image" is 1D, let the
            Y and Z dimensions be 1. The tuple must be on the form
            (Z, Y, X).

    mask  : A 1-dimensional mask representing the 3D image mask. Must be a
           list of 1s and 0s.
    """
    def __init__(self, l, k, gamma, shape, mask=None, **kwargs):

        super(EGMRidgeRegressionL1TV, self).__init__(**kwargs)

        self.set_g(loss_functions.RidgeRegression(k))

        a = loss_functions.SmoothL1(l, np.prod(shape), mask=mask)
        b = loss_functions.TotalVariation(gamma, shape=shape, mask=mask,
                                          compress=False)

        self.set_h(loss_functions.CombinedNesterovLossFunction(a, b))