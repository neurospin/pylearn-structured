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

__all__ = ["QuadraticConstraint", "RGCCAConstraint",
           "L1",
           "SufficientDescentCondition"]


class QuadraticConstraint(interfaces.AtomicFunction,
                          interfaces.Gradient,
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

        self._U = None
        self._S = None

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

        if self._U is None:
            if self.unbiased:
                n = self.X.shape[0] - 1.0
            else:
                n = self.X.shape[0]
            const = ((1.0 - self.tau) / float(n))

            num_comp = self.X.shape[1]
            self._U = np.zeros((self.X.shape[1], num_comp))
            self._S = np.diag([0] * num_comp)
            first = True
#            self._U = None
#            self._S = None
            X_ = self.X
            for i in xrange(num_comp):
                u = np.random.rand(X_.shape[1], 1)
                u /= maths.norm(u)
                for k in xrange(10000):
                    u2 = np.dot(X_.T, np.dot(X_, u))
                    u2 *= const
                    u2 += self.tau * u
                    if first:
                        first = False
                    else:  # self._U is not None:
                        u2 -= np.dot(self._U, np.dot(self._S,
                                                     np.dot(self._U.T, u)))
                    norm_u = maths.norm(u2)
                    if norm_u < consts.TOLERANCE:
                        print i
                    u2 /= norm_u
#                    print norm_u

                    if maths.norm(u - u2) < consts.TOLERANCE:
                        u = u2
                        break
                    u = u2

#                t = np.dot(X_, u)
#                p = np.dot(X_.T, t) / np.dot(t.T, t)
#                X_ = X_ - np.dot(t, p.T)
#                X_ = X_ - np.dot(np.dot(X_, np.sqrt(const) * u), u.T)

#                if first:#self._U is None:
#                    self._U = u
#                    self._S = [norm_u]
#                    first = False
#                else:
#                    self._U = np.hstack((self._U, u))
#                    self._S.append(norm_u)
                self._U[:, [i]] = u
                self._S[i, i] = norm_u

#            self._U = np.hstack(self._U)
#            self._S = np.diag(self._S)
#            self._L = np.dot(np.hstack(self._L), np.diag(self._S))
            del X_

#        # The case: n > p
#        L = np.linalg.cholesky(np.dot(A.T, A))

#        L = self._L
        XtX = np.dot(self.X.T, self.X)
        M = self.tau * np.eye(*XtX.shape) \
            + ((1.0 - self.tau) / float(self.X.shape[0] - 1)) * XtX
#        D, U = np.linalg.eig(M)
        U, S, V = np.linalg.svd(M)
#        D = D.real
#        U = U.real
#        L_ = np.dot(U, np.sqrt(np.diag(D)))
        L_ = np.dot(U, np.sqrt(np.diag(S)))
        L__ = np.dot(self._U, np.sqrt(self._S))
        L = np.linalg.cholesky(M)

#        print np.linalg.norm(L_[:, [0]] + L__[:, [0]])
#        print np.linalg.norm(L_[:, [1]] - L__[:, [1]])
#        print np.linalg.norm(L_[:, [2]] + L__[:, [2]])
#        print np.linalg.norm(L_[:, [3]] - L__[:, [3]])
#        print np.dot(L_, L_.T)[1:5,1:5]
#        print np.dot(L__, L__.T)[1:5,1:5]
#        print np.linalg.norm(self._S - np.diag(S))
        print np.dot(self._U, self._U.T)[:10, :10]
        print np.dot(U, U.T)[:10, :10]
#        print np.dot(U, self._U.T)[:10,:10]
#        for i in xrange(10):
#            for j in xrange(10):
#                print np.dot(L__[:, [i]].T, L__[:, [j]]), np.dot(L_[:, [i]].T, L_[:, [j]])

        print L.shape
        print L_.shape
        print L__.shape
        print maths.norm(M - np.dot(L, L.T))
        print maths.norm(M - np.dot(L_, L_.T))
        print maths.norm(M - np.dot(L__, L__.T))

        invL = np.linalg.pinv(L)
        t = 0.99 / (np.max(S) ** 2.0)
#        t = 0.99 / self._S[0]
        y = x
        for i in xrange(1000):
            y = y - t * (np.dot(invL, np.dot(invL.T, y)) - x)
            y /= maths.norm(y)
#        y = np.dot(np.linalg.inv(np.dot(invL, invL.T)), x)
#        y /= maths.norm(y)

        proj_x2 = np.dot(invL.T, y)
#        proj_x2 = (np.sqrt(self.c / self._compute_value(proj_x2))) * proj_x2

        proj_x = (np.sqrt(self.c / xtMx)) * x

        print "norm:", self._compute_value(proj_x2)
        print "norm:", self._compute_value(proj_x)
        print "dist:", maths.norm(proj_x2 - x)
        print "dist:", maths.norm(proj_x - x)
        print

        return proj_x

    def _compute_value(self, beta):

        if self.unbiased:
            n = self.X.shape[0] - 1.0
        else:
            n = self.X.shape[0]

        Xbeta = np.dot(self.X, beta)
        val = self.tau * np.dot(beta.T, beta) \
            + ((1.0 - self.tau) / float(n)) * np.dot(Xbeta.T, Xbeta)

        return val[0, 0]


class L1(interfaces.AtomicFunction,
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