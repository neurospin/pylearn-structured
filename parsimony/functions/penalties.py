# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.functions.penalties` module contains the penalties used to
constrain the loss functions. These represent mathematical functions and
should thus have properties used by the corresponding algorithms. These
properties are defined in :mod:`parsimony.functions.interfaces`.

Penalties should be stateless. Penalties may be shared and copied and should
therefore not hold anything that cannot be recomputed the next time it is
called.

Created on Mon Apr 22 10:54:29 2013

@author:  Tommy Löfstedt, Vincent Guillemot, Edouard Duchesnay and
          Fouad Hadj-Selem
@email:   lofstedt.tommy@gmail.com, edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
import numpy as np

import parsimony.utils.consts as consts
import parsimony.utils.maths as maths
import interfaces

__all__ = ["L1",
           "QuadraticConstraint", "RGCCAConstraint",
           "SufficientDescentCondition"]


class L1(interfaces.AtomicFunction,
         interfaces.Regularisation,
         interfaces.Constraint,
         interfaces.ProximalOperator,
         interfaces.ProjectionOperator):
    """The proximal operator of the L1 function with regularisation formulation

        f(\beta) = l * (||\beta||_1 - c),

    where ||\beta||_1 is the L1 loss function. The constrained version has the
    form

        ||\beta||_1 <= c.

    Parameters:
    ----------
    l : The Lagrange multiplier, or regularisation constant, of the function.

    c : The limit of the constraint. The function is feasible if
            ||\beta||_1 <= c. The default value is c=0, i.e. the default is a
            regularisation formulation.
    """
    def __init__(self, l=1.0, c=0.0):

        self.l = float(l)
        self.c = float(c)

    def f(self, beta):
        """Function value.
        """
        return self.l * (maths.norm1(beta) - self.c)

    def prox(self, beta, factor=1.0):
        """The corresponding proximal operator.

        From the interface "ProximalOperator".
        """
        l = self.l * factor
        return (np.abs(beta) > l) * (beta - l * np.sign(beta - l))

    def proj(self, beta):
        """The corresponding projection operator.

        From the interface "ProjectionOperator".
        """
        if self.feasible(beta):
            return beta

        from algorithms import Bisection
        bisection = Bisection(force_negative=True, eps=1e-10)

        class F(interfaces.Function):
            def __init__(self, beta, c):
                self.beta = beta
                self.c = c

            def f(self, l):
                beta = (np.abs(self.beta) > l) \
                    * (self.beta - l * np.sign(self.beta - l))

                return maths.norm1(beta) - self.c

        func = F(beta, self.c)
        l = bisection(func, [0.0, np.max(np.abs(beta))])

        return (np.abs(beta) > l) * (beta - l * np.sign(beta - l))

    def feasible(self, beta):
        """Feasibility of the constraint.

        From the interface "Constraint".
        """
        return maths.norm1(beta) <= self.c


class QuadraticConstraint(interfaces.AtomicFunction,
                          interfaces.Gradient,
                          interfaces.Regularisation,
                          interfaces.Constraint):
    """The proximal operator of the quadratic function

        f(x) = l * (x'Mx - c),

    where M is a given symmatric positive definite matrix. The constrained
    version has the form

        x'Mx <= c.

    Parameters:
    ----------
    l : Float. The Lagrange multiplier, or regularisation constant, of the
            function.

    c : Float. The limit of the constraint. The function is feasible if
            x'Mx <= c. The default value is c=0, i.e. the default is a
            regularisation formulation.

    M : Array. The given positive definite matrix
    """
    def __init__(self, l=1.0, c=0.0, M=None):

        self.l = float(l)
        self.c = float(c)
        self.M = M

    def f(self, beta):
        """Function value.
        """
        return self.l * (np.dot(beta.T, np.dot(self.M, beta)) - self.c)

    def grad(self, beta):
        """Gradient of the function.

        From the interface "Gradient".
        """
        return (self.l * 2.0) * np.dot(self.M, beta)

    def feasible(self, beta):
        """Feasibility of the constraint.

        From the interface "Constraint".
        """
        return np.dot(beta.T, np.dot(self.M, beta)) <= self.c


class RGCCAConstraint(QuadraticConstraint,
                      interfaces.ProjectionOperator):
    """The proximal operator of the quadratic function

        f(x) = l * (x'(\tau * I + ((1 - \tau) / n) * X'X)x - c),

    where \tau is a given regularisation constant. The constrained version has
    the form

        x'(\tau * I + ((1 - \tau) / n) * X'X)x <= c.

    Parameters:
    ----------
    l : Float. The Lagrange multiplier, or regularisation constant, of the
            function.

    c : Float. The limit of the constraint. The function is feasible if
            x'(\tau * I + ((1 - \tau) / n) * X'X)x <= c. The default value is
            c=0, i.e. the default is a regularisation formulation.

    tau : Float. Given regularisation constant

    unbiased : Boolean.
    """
    def __init__(self, l=1.0, c=0.0, tau=1.0, X=None, unbiased=True):

        self.l = float(l)
        self.c = float(c)
        self.tau = max(0.0, min(float(tau), 1.0))
        self.X = X
        self.unbiased = unbiased

        self.reset()

    def reset(self):

        self._VU = None

        self._Ip = None
        self._M = None

    def f(self, beta):
        """Function value.
        """
        xtMx = self._compute_value(beta)
        return self.l * (xtMx - self.c)

    def grad(self, beta):
        """Gradient of the function.

        From the interface "Gradient".
        """
        if self.unbiased:
            n = self.X.shape[0] - 1.0
        else:
            n = self.X.shape[0]

        if self.tau < 1.0:
            XtXbeta = np.dot(self.X.T, np.dot(self.X, beta))
            grad = (self.tau * 2.0) * beta \
                 + ((1.0 - self.tau) * 2.0 / float(n)) * XtXbeta
        else:
            grad = (self.tau * 2.0) * beta

#        approx_grad = utils.approx_grad(self.f, beta, eps=1e-4)
#        print maths.norm(grad - approx_grad)

        return grad

    def feasible(self, beta):
        """Feasibility of the constraint.

        From the interface "Constraint".
        """
        xtMx = self._compute_value(beta)
        return xtMx <= self.c

    def proj(self, x):
        """The projection operator corresponding to the function.

        From the interface "ProjectionOperator".
        """
        xtMx = self._compute_value(x)
        if xtMx <= self.c:
            return x

        n, p = self.X.shape
        if p > n:
            In = np.eye(n)     # n-by-n
            U = self.X.T       # p-by-n
#            V = self.X         # n-by-p
            Vx = np.dot(self.X, x)  # n-by-1
            if self._VU is None:
                self._VU = np.dot(self.X, U)  # n-by-n

            def prox(x, l):
                k = 0.5 * l * self.tau + 1.0
                m = 0.5 * l * ((1.0 - self.tau) / float(n - 1))

                invIMx = (x - np.dot(U, np.dot(np.linalg.inv((k / m) * In +
                        self._VU), Vx))) / k

                return invIMx

            from parsimony.algorithms import Bisection
            bisection = Bisection(force_negative=True,
                                  parameter_positive=True,
                                  parameter_negative=False,
                                  parameter_zero=False,
                                  eps=1e-3)

            class F(interfaces.Function):
                def __init__(self, x, c, val):
                    self.x = x
                    self.c = c
                    self.val = val
                    self.y = None

                def f(self, l):

                    # We avoid one evaluation of prox by saving it here.
                    self.y = prox(x, l)

                    return self.val(self.y) - self.c

            func = F(x, self.c, self._compute_value)
            # TODO: Tweak these magic numbers on real data. Or even better,
            # find theoretical bounds. Convergence is faster if these bounds
            # are close to accurate when we start the bisection algorithm.
            if p >= 400000:
                low = p / 90.0
                high = p / 71.0
            elif p >= 200000:
                low = p / 70.0
                high = p / 54.0
            elif p >= 100000:
                low = p / 45.0
                high = p / 40.0
            elif p >= 50000:
                low = p / 37.0
                high = p / 27.0
            elif p >= 10000:
                low = p / 20.0
                high = p / 15.0
            elif p >= 5000:
                low = p / 11.0
                high = p / 9.0
            elif p >= 1000:
                low = p / 8.0
                high = p / 5.0
            else:
                low = p / 4.0
                high = p / 3.0

            print low, ", ", high
            l = bisection(func, [low, high])
            print l

            y = func.y

        else:  # The case when: p <= n

            if self._Ip is None:
                self._Ip = np.eye(p)  # p-by-p

            if self._M is None:
                XtX = np.dot(self.X.T, self.X)
                self._M = self.tau * self._Ip + \
                          ((1.0 - self.tau) / float(n - 1)) * XtX

            def prox2(x, l):

                y = np.dot(np.linalg.inv(self._Ip + (0.5 * l) * self._M), x)

                return y

            from parsimony.algorithms import Bisection
            bisection = Bisection(force_negative=True,
                                  parameter_positive=True,
                                  parameter_negative=False,
                                  parameter_zero=False,
                                  eps=1e-3)

            class F(interfaces.Function):
                def __init__(self, x, c, val):
                    self.x = x
                    self.c = c
                    self.val = val
                    self.y = None

                def f(self, l):

                    # We avoid one evaluation of prox by saving it here.
                    self.y = prox2(x, l)

                    return self.val(self.y) - self.c

            func = F(x, self.c, self._compute_value)
            # TODO: Tweak these magic numbers on real data. Or even better,
            # find theoretical bounds. Convergence is faster if these bounds
            # are close to accurate when we start the bisection algorithm.
            if p >= 950:
                low = p / 5.5555555
                high = (p / 4.56) - np.log10(n)
            elif p >= 850:
                low = p / 5.294
                high = (p / 4.29) - np.log10(n)
            elif p >= 750:
                low = p / 5.0
                high = (p / 4.17) - np.log10(n)
            elif p >= 650:
                low = p / 4.66
                high = (p / 3.85) - np.log10(n)
            elif p >= 550:
                low = p / 4.28
                high = (p / 3.52) - np.log10(n)
            elif p >= 450:
                low = p / 4.0
                high = (p / 3.33) - np.log10(n)
            elif p >= 350:
                low = p / 3.7
                high = (p / 2.96) - np.log10(n)
            elif p >= 250:
                low = p / 3.16
                high = (p / 2.55) - np.log10(n)
            elif p >= 150:
                low = p / 2.667
                high = (p / 2.1) - np.log10(n)
            elif p >= 50:
                low = p / 2.0
                high = (p / 1.5) - np.log10(n)
            else:
                low = p / 1.0
                high = (p / 0.71) - np.log10(n)

            print low, ", ", high
            l = bisection(func, [low, high])
            print l

            y = func.y

        return y

    def _compute_value(self, beta):

        if self.unbiased:
            n = self.X.shape[0] - 1.0
        else:
            n = self.X.shape[0]

        Xbeta = np.dot(self.X, beta)
        val = self.tau * np.dot(beta.T, beta) \
            + ((1.0 - self.tau) / float(n)) * np.dot(Xbeta.T, Xbeta)

        return val[0, 0]


class SufficientDescentCondition(interfaces.Function,
                                 interfaces.Constraint):

    def __init__(self, function, p, c):
        """The sufficient condition

            f(x + a * p) <= f(x) + c * a * grad(f(x))'p

        for descent. This condition is sometimes called the Armijo condition.

        Parameters:
        ----------
        c : Float. 0 < c < 1. A constant for the condition. Should be small.
        """
        self.function = function
        self.p = p
        self.c = c

    def f(self, x, a):

        return self.function.f(x + a * self.p)

    """Feasibility of the constraint at point x.

    From the interface "Constraint".
    """
    def feasible(self, x, a):

        f_x_ap = self.function.f(x + a * self.p)
        f_x = self.function.f(x)
        grad_p = np.dot(self.function.grad(x).T, self.p)[0, 0]
#        print "f_x_ap = %.10f, f_x = %.10f, grad_p = %.10f, feas = %.10f" % (f_x_ap, f_x, grad_p, f_x + self.c * a * grad_p)
#        if grad_p >= 0.0:
#            pass
        feasible = f_x_ap <= f_x + self.c * a * grad_p

        return feasible


#class WolfeCondition(Function, Constraint):
#
#    def __init__(self, function, p, c1=1e-4, c2=0.9):
#        """
#        Parameters:
#        ----------
#        c1 : Float. 0 < c1 < c2 < 1. A constant for the condition. Should be
#                small.
#        c2 : Float. 0 < c1 < c2 < 1. A constant for the condition. Depends on
#                the minimisation algorithms. For Newton or quasi-Newton
#                descent directions, 0.9 is a good choice. 0.1 is appropriate
#                for nonlinear conjugate gradient.
#        """
#        self.function = function
#        self.p = p
#        self.c1 = c1
#        self.c2 = c2
#
#    def f(self, x, a):
#
#        return self.function.f(x + a * self.p)
#
#    """Feasibility of the constraint at point x.
#
#    From the interface "Constraint".
#    """
#    def feasible(self, x, a):
#
#        grad_p = np.dot(self.function.grad(x).T, self.p)[0, 0]
#        cond1 = self.function.f(x + a * self.p) \
#            <= self.function.f(x) + self.c1 * a * grad_p
#        cond2 = np.dot(self.function.grad(x + a * self.p).T, self.p)[0, 0] \
#            >= self.c2 * grad_p
#
#        return cond1 and cond2
#
#
#class StrongWolfeCondition(Function, Constraint):
#
#    def __init__(self, function, p, c1=1e-4, c2=0.9):
#        """
#        Parameters:
#        ----------
#        c1 : Float. 0 < c1 < c2 < 1. A constant for the condition. Should be
#                small.
#        c2 : Float. 0 < c1 < c2 < 1. A constant for the condition. Depends on
#                the minimisation algorithms. For Newton or quasi-Newton
#                descent directions, 0.9 is a good choice. 0.1 is appropriate
#                for nonlinear conjugate gradient.
#        """
#        self.function = function
#        self.p = p
#        self.c1 = c1
#        self.c2 = c2
#
#    def f(self, x, a):
#
#        return self.function.f(x + a * self.p)
#
#    """Feasibility of the constraint at point x.
#
#    From the interface "Constraint".
#    """
#    def feasible(self, x, a):
#
#        grad_p = np.dot(self.function.grad(x).T, self.p)[0, 0]
#        cond1 = self.function.f(x + a * self.p) \
#            <= self.function.f(x) + self.c1 * a * grad_p
#        grad_x_ap = self.function.grad(x + a * self.p)
#        cond2 = abs(np.dot(grad_x_ap.T, self.p)[0, 0]) \
#            <= self.c2 * abs(grad_p)
#
#        return cond1 and cond2