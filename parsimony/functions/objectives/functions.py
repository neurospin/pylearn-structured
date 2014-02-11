# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.functions.functions` module contains rady-made commong
combinations of loss functions and penalties that can be used right away to
analyse real data.

Created on Mon Apr 22 10:54:29 2013

@author:  Tommy Löfstedt, Vincent Guillemot, Edouard Duchesnay and
          Fouad Hadj-Selem
@email:   lofstedt.tommy@gmail.com, edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
import numpy as np

from parsimony.functions.objectives import RidgeRegression
from parsimony.functions.objectives import RidgeLogisticRegression
from parsimony.functions.penalties import L1
from parsimony.functions.nesterov.L1 import L1 as SmoothedL1
from parsimony.functions.nesterov.L1TV import L1TV
from parsimony.functions.nesterov.tv import TotalVariation
from parsimony.functions.nesterov.gl import GroupLassoOverlap
import parsimony.utils.consts as consts
import interfaces

__all__ = ["RR_L1_TV", "RLR_L1_TV", "RR_L1_GL", "RR_SmoothedL1TV"]


class RR_L1_TV(interfaces.CompositeFunction,
               interfaces.Gradient,
               interfaces.LipschitzContinuousGradient,
               interfaces.ProximalOperator,
               interfaces.NesterovFunction,
               interfaces.Continuation,
               interfaces.DualFunction):
    """Combination (sum) of RidgeRegression, L1 and TotalVariation

    Parameters
    ----------
    X : Ridge Regression parameter.

    y : Ridge Regression parameter.

    k : Ridge Regression parameter.

    l : L1 parameter.
            The Lagrange multiplier, or regularisation constant, of the
            function.

    g : Total Variation parameter
            The Lagrange multiplier, or regularisation constant, of the
            function.

    A : Total Variation parameter.
            The linear operator for the Nesterov formulation. May not be None!

    mu : Total Variation parameter.
            The regularisation constant for the smoothing.
    """
    def __init__(self, X, y, k, l, g, A=None, mu=0.0):

        self.X = X
        self.y = y

        self.rr = RidgeRegression(X, y, k)
        self.l1 = L1(l)
        self.tv = TotalVariation(g, A=A, mu=mu)

        self.reset()

    def reset(self):

        self.rr.reset()
        self.l1.reset()
        self.tv.reset()

        self._Xty = None
        self._invXXkI = None
        self._XtinvXXtkI = None

    def set_params(self, **kwargs):

        mu = kwargs.pop("mu", self.get_mu())
        self.set_mu(mu)

        super(RR_L1_TV, self).set_params(**kwargs)

    def get_mu(self):
        """Returns the regularisation constant for the smoothing.

        From the interface "NesterovFunction".
        """
        return self.tv.get_mu()

    def set_mu(self, mu):
        """Sets the regularisation constant for the smoothing.

        From the interface "NesterovFunction".

        Parameters
        ----------
        mu: The regularisation constant for the smoothing to use from now on.

        Returns
        -------
        old_mu: The old regularisation constant for the smoothing that was
                overwritten and is no longer used.
        """
        return self.tv.set_mu(mu)

    def f(self, beta):
        """Function value.
        """
        return self.rr.f(beta) \
             + self.l1.f(beta) \
             + self.tv.f(beta)

    def fmu(self, beta, mu=None):
        """Function value.
        """
        return self.rr.f(beta) \
             + self.l1.f(beta) \
             + self.tv.fmu(beta, mu)

    def phi(self, alpha, beta):
        """ Function value with known alpha.
        """
        return self.rr.f(beta) \
             + self.l1.f(beta) \
             + self.tv.phi(alpha, beta)

    def grad(self, beta):
        """Gradient of the differentiable part of the function.

        From the interface "Gradient".
        """
        return self.rr.grad(beta) \
             + self.tv.grad(beta)

    def L(self):
        """Lipschitz constant of the gradient.

        From the interface "LipschitzContinuousGradient".
        """
        return self.rr.L() \
             + self.tv.L()

    def prox(self, beta, factor=1.0):
        """The proximal operator of the non-differentiable part of the
        function.

        From the interface "ProximalOperator".
        """
        return self.l1.prox(beta, factor)

    def estimate_mu(self, beta):
        """Computes a "good" value of \mu with respect to the given \beta.

        From the interface "NesterovFunction".
        """
        return self.tv.estimate_mu(beta)

    def M(self):
        """The maximum value of the regularisation of the dual variable. We
        have

            M = max_{\alpha \in K} 0.5*|\alpha|²_2.

        From the interface "NesterovFunction".
        """
        return self.tv.M()

    def mu_opt(self, eps):
        """The optimal value of \mu given \epsilon.

        From the interface "Continuation".
        """
        gM = self.tv.l * self.tv.M()

        # Mu is set to 1.0, because it is in fact not here "anymore". It is
        # factored out in this solution.
        old_mu = self.tv.set_mu(1.0)
        gA2 = self.tv.L()  # Gamma is in here!
        self.tv.set_mu(old_mu)

        Lg = self.rr.L()

        return (-gM * gA2 + np.sqrt((gM * gA2) ** 2.0
             + gM * Lg * gA2 * eps)) \
             / (gM * Lg)

    def eps_opt(self, mu):
        """The optimal value of \epsilon given \mu.

        From the interface "Continuation".
        """
        gM = self.tv.l * self.tv.M()

        # Mu is set to 1.0, because it is in fact not here "anymore". It is
        # factored out in this solution.
        old_mu = self.tv.set_mu(1.0)
        gA2 = self.tv.L()  # Gamma is in here!
        self.tv.set_mu(old_mu)

        Lg = self.rr.L()

        return (2.0 * gM * gA2 * mu
             + gM * Lg * mu ** 2.0) \
             / gA2

    def eps_max(self, mu):
        """The maximum value of \epsilon.

        From the interface "Continuation".
        """
        gM = self.tv.l * self.tv.M()

        return mu * gM

    def betahat(self, alphak, betak):
        """ Returns the beta that minimises the dual function. Used when we
        compute the gap.

        From the interface "DualFunction".
        """
        if self._Xty is None:
            self._Xty = np.dot(self.X.T, self.y)

        Ata_tv = self.tv.l * self.tv.Aa(alphak)
        Ata_l1 = self.l1.l * SmoothedL1.project([betak / consts.TOLERANCE])[0]
        v = (self._Xty - Ata_tv - Ata_l1)

        shape = self.X.shape

        if shape[0] > shape[1]:  # If n > p

            # Ridge solution
            if self._invXXkI is None:
                XtXkI = np.dot(self.X.T, self.X)
                index = np.arange(min(XtXkI.shape))
                XtXkI[index, index] += self.rr.k
                self._invXXkI = np.linalg.inv(XtXkI)

            beta_hat = np.dot(self._invXXkI, v)

        else:  # If p > n
            # Ridge solution using the Woodbury matrix identity:
            if self._XtinvXXtkI is None:
                XXtkI = np.dot(self.X, self.X.T)
                index = np.arange(min(XXtkI.shape))
                XXtkI[index, index] += self.rr.k
                invXXtkI = np.linalg.inv(XXtkI)
                self._XtinvXXtkI = np.dot(self.X.T, invXXtkI)

            beta_hat = (v - np.dot(self._XtinvXXtkI, np.dot(self.X, v))) \
                       / self.rr.k

        return beta_hat

    def gap(self, beta, beta_hat=None):
        """Compute the duality gap.

        From the interface "DualFunction".
        """
        alpha = self.tv.alpha(beta)

        P = self.rr.f(beta) \
          + self.l1.f(beta) \
          + self.tv.phi(alpha, beta)

        beta_hat = self.betahat(alpha, beta)

        D = self.rr.f(beta_hat) \
          + self.l1.f(beta_hat) \
          + self.tv.phi(alpha, beta_hat)

        return P - D

    def A(self):
        """Linear operator of the Nesterov function.

        From the interface "NesterovFunction".
        """
        return self.tv.A()

    def Aa(self, alpha):
        """Computes A^\T\alpha.

        From the interface "NesterovFunction".
        """
        return self.tv.Aa(alpha)

    def project(self, a):
        """ Projection onto the compact space of the Nesterov function.

        From the interface "NesterovFunction".
        """
        return self.tv.project(a)


class RLR_L1_TV(RR_L1_TV):
    """Combination (sum) of RidgeLogisticRegression, L1 and TotalVariation

    Parameters
    ----------
    X : Ridge Regression parameter.

    y : Ridge Regression parameter.

    k : Ridge Regression parameter.

    l : L1 parameter.
            The Lagrange multiplier, or regularisation constant, of the
            function.

    g : Total Variation parameter
            The Lagrange multiplier, or regularisation constant, of the
            function.

    A : Total Variation parameter.
            The linear operator for the Nesterov formulation. May not be None!

    mu : Total Variation parameter.
            The regularisation constant for the smoothing.

    weights: array, shape = [n_samples]
        samples weights
    """

    def __init__(self, X, y, k, l, g, A=None, mu=0.0, weights=None):

        self.X = X
        self.y = y

        self.rr = RidgeLogisticRegression(X, y, k, weights=weights)
        self.l1 = L1(l)
        self.tv = TotalVariation(g, A=A, mu=0.0)

        self.reset()


class RR_L1_GL(RR_L1_TV):
    """Combination (sum) of RidgeRegression, L1 and Overlapping Group Lasso.

    Parameters
    ----------
    X : Matrix. Ridge Regression parameter.

    y : Vector. Ridge Regression parameter.

    k : Float. The Ridge Regression parameter.

    l : Float. The Lagrange multiplier, or regularisation constant, of the L1
            function.

    g : Float. The Lagrange multiplier, or regularisation constant, of the GL
            function.

    A : A (usually sparse) matrix. The linear operator for the Nesterov
            formulation for GL. May not be None!

    mu : Float. mu > 0. The regularisation constant for the smoothing of the
            GL function.
    """

    def __init__(self, X, y, k, l, g, A=None, mu=0.0):

        self.X = X
        self.y = y

        self.rr = RidgeRegression(X, y, k)
        self.l1 = L1(l)
        self.gl = GroupLassoOverlap(g, A=A, mu=mu)

        self.reset()

    def reset(self):

        self.rr.reset()
        self.l1.reset()
        self.gl.reset()

        self._Xty = None
        self._invXXkI = None
        self._XtinvXXtkI = None

    def set_params(self, **kwargs):

        mu = kwargs.pop("mu", self.get_mu())
        self.set_mu(mu)

        super(RR_L1_TV, self).set_params(**kwargs)

    def get_mu(self):
        """Returns the regularisation constant for the smoothing.

        From the interface "NesterovFunction".
        """
        return self.gl.get_mu()

    def set_mu(self, mu):
        """Sets the regularisation constant for the smoothing.

        From the interface "NesterovFunction".

        Parameters
        ----------
        mu: The regularisation constant for the smoothing to use from now on.

        Returns
        -------
        old_mu: The old regularisation constant for the smoothing that was
                overwritten and is no longer used.
        """
        return self.gl.set_mu(mu)

    def f(self, beta):
        """Function value.
        """
        return self.rr.f(beta) \
             + self.l1.f(beta) \
             + self.gl.f(beta)

    def fmu(self, beta, mu=None):
        """Function value.
        """
        return self.rr.f(beta) \
             + self.l1.f(beta) \
             + self.gl.fmu(beta, mu)

    def phi(self, alpha, beta):
        """ Function value with known alpha.
        """
        return self.rr.f(beta) \
             + self.l1.f(beta) \
             + self.gl.phi(alpha, beta)

    def grad(self, beta):
        """Gradient of the differentiable part of the function.

        From the interface "Gradient".
        """
        return self.rr.grad(beta) \
             + self.gl.grad(beta)

    def L(self):
        """Lipschitz constant of the gradient.

        From the interface "LipschitzContinuousGradient".
        """
        return self.rr.L() \
             + self.gl.L()

    def prox(self, beta, factor=1.0):
        """The proximal operator of the non-differentiable part of the
        function.

        From the interface "ProximalOperator".
        """
        return self.l1.prox(beta, factor)

    def estimate_mu(self, beta):
        """Computes a "good" value of \mu with respect to the given \beta.

        From the interface "NesterovFunction".
        """
        return self.gl.estimate_mu(beta)

    def M(self):
        """The maximum value of the regularisation of the dual variable. We
        have

            M = max_{\alpha \in K} 0.5*|\alpha|²_2.

        From the interface "NesterovFunction".
        """
        return self.gl.M()

    def mu_opt(self, eps):
        """The optimal value of \mu given \epsilon.

        From the interface "Continuation".
        """
        gM = self.gl.l * self.gl.M()

        # Mu is set to 1.0, because it is in fact not here "anymore". It is
        # factored out in this solution.
        old_mu = self.gl.set_mu(1.0)
        gA2 = self.gl.L()  # Gamma is in here!
        self.gl.set_mu(old_mu)

        Lg = self.rr.L()

        return (-gM * gA2 + np.sqrt((gM * gA2) ** 2.0
             + gM * Lg * gA2 * eps)) \
             / (gM * Lg)

    def eps_opt(self, mu):
        """The optimal value of \epsilon given \mu.

        From the interface "Continuation".
        """
        gM = self.gl.l * self.gl.M()

        # Mu is set to 1.0, because it is in fact not here "anymore". It is
        # factored out in this solution.
        old_mu = self.gl.set_mu(1.0)
        gA2 = self.gl.L()  # Gamma is in here!
        self.gl.set_mu(old_mu)

        Lg = self.rr.L()

        return (2.0 * gM * gA2 * mu
             + gM * Lg * mu ** 2.0) \
             / gA2

    def eps_max(self, mu):
        """The maximum value of \epsilon.

        From the interface "Continuation".
        """
        gM = self.gl.l * self.gl.M()

        return mu * gM

    def betahat(self, alphak, betak):
        """ Returns the beta that minimises the dual function. Used when we
        compute the gap.

        From the interface "DualFunction".
        """
        if self._Xty is None:
            self._Xty = np.dot(self.X.T, self.y)

        Ata_tv = self.gl.l * self.gl.Aa(alphak)
        Ata_l1 = self.l1.l * SmoothedL1.project([betak / consts.TOLERANCE])[0]
        v = (self._Xty - Ata_tv - Ata_l1)

        shape = self.X.shape

        if shape[0] > shape[1]:  # If n > p

            # Ridge solution
            if self._invXXkI is None:
                XtXkI = np.dot(self.X.T, self.X)
                index = np.arange(min(XtXkI.shape))
                XtXkI[index, index] += self.rr.k
                self._invXXkI = np.linalg.inv(XtXkI)

            beta_hat = np.dot(self._invXXkI, v)

        else:  # If p > n
            # Ridge solution using the Woodbury matrix identity:
            if self._XtinvXXtkI is None:
                XXtkI = np.dot(self.X, self.X.T)
                index = np.arange(min(XXtkI.shape))
                XXtkI[index, index] += self.rr.k
                invXXtkI = np.linalg.inv(XXtkI)
                self._XtinvXXtkI = np.dot(self.X.T, invXXtkI)

            beta_hat = (v - np.dot(self._XtinvXXtkI, np.dot(self.X, v))) \
                       / self.rr.k

        return beta_hat

    def gap(self, beta, beta_hat=None):
        """Compute the duality gap.

        From the interface "DualFunction".
        """
        alpha = self.gl.alpha(beta)

        P = self.rr.f(beta) \
          + self.l1.f(beta) \
          + self.gl.phi(alpha, beta)

        beta_hat = self.betahat(alpha, beta)

        D = self.rr.f(beta_hat) \
          + self.l1.f(beta_hat) \
          + self.gl.phi(alpha, beta_hat)

        return P - D

    def A(self):
        """Linear operator of the Nesterov function.

        From the interface "NesterovFunction".
        """
        return self.gl.A()

    def Aa(self, alpha):
        """Computes A^\T\alpha.

        From the interface "NesterovFunction".
        """
        return self.gl.Aa()

    def project(self, a):
        """ Projection onto the compact space of the Nesterov function.

        From the interface "NesterovFunction".
        """
        return self.gl.project(a)


class RR_SmoothedL1TV(interfaces.CompositeFunction,
                      interfaces.LipschitzContinuousGradient,
                      interfaces.GradientMap,
                      interfaces.DualFunction,
                      interfaces.NesterovFunction):
    """
    Parameters
    ----------
    X : Ridge Regression parameter.

    y : Ridge Regression parameter.

    k : Ridge Regression parameter.

    l : L1 parameter.
            The Lagrange multiplier, or regularisation constant, of the
            function.

    g : Total Variation parameter.
            The Lagrange multiplier, or regularisation constant, of the
            function.

    Atv : The linear operator for the total variation Nesterov function.

    Al1 : Matrix allocation for regression.

    mu: The regularisation constant for the smoothing.
    """
    def __init__(self, X, y, k, l, g, Atv=None, Al1=None, mu=0.0):

        self.X = X
        self.y = y

        self.g = RidgeRegression(X, y, k)
        self.h = L1TV(l, g, Atv=Atv, Al1=Al1, mu=mu)

        self.mu = mu

        self.reset()

    def reset(self):

        self.g.reset()
        self.h.reset()

        self._Xy = None
        self._XtinvXXtkI = None

    def set_params(self, **kwargs):

        mu = kwargs.pop("mu", self.get_mu())
        self.set_mu(mu)

        super(RR_SmoothedL1TV, self).set_params(**kwargs)

    def get_mu(self):
        """Returns the regularisation constant for the smoothing.

        From the interface "NesterovFunction".
        """
        return self.h.get_mu()

    def set_mu(self, mu):
        """Sets the regularisation constant for the smoothing.

        From the interface "NesterovFunction".

        Parameters
        ----------
        mu: The regularisation constant for the smoothing to use from now on.

        Returns
        -------
        old_mu: The old regularisation constant for the smoothing that was
                overwritten and is no longer used.
        """
        return self.h.set_mu(mu)

    def f(self, beta):
        """ Function value.
        """
        return self.g.f(beta) \
             + self.h.f(beta)

    def phi(self, alpha, beta):
        """ Function value.
        """
        return self.g.f(beta) \
             + self.h.phi(alpha, beta)

    def L(self):
        """Lipschitz constant of the gradient.

        From the interface "LipschitzContinuousGradient".
        """
        b = self.g.lambda_min()
        # TODO: Use max_iter here!!
        a = self.h.lambda_max()  # max_iter=max_iter)

        return a / b

    def V(self, u, beta, L):
        """The gradient map associated to the function.

        From the interface "GradientMap".
        """
        A = self.h.A()
        a = [0] * len(A)
        a[0] = (1.0 / L) * A[0].dot(beta)
        a[1] = (1.0 / L) * A[1].dot(beta)
        a[2] = (1.0 / L) * A[2].dot(beta)
        a[3] = (1.0 / L) * A[3].dot(beta)

        u_new = [0] * len(u)
        for i in xrange(len(u)):
            u_new[i] = u[i] + a[i]

        return self.h.project(u_new)

    def betahat(self, alpha, beta=None):
        """ Returns the beta that minimises the dual function.

        From the interface "DualFunction".
        """
        # TODO: Kernelise this function! See how I did in LRL2_L1_TV._beta_hat.

        A = self.h.A()
        grad = A[0].T.dot(alpha[0])
        grad += A[1].T.dot(alpha[1])
        grad += A[2].T.dot(alpha[2])
        grad += A[3].T.dot(alpha[3])

#        XXkI = np.dot(X.T, X) + self.g.k * np.eye(X.shape[1])

        if self._Xy is None:
            self._Xy = np.dot(self.X.T, self.y)

        Xty_grad = (self._Xy - grad) / self.g.k

#        t = time()
#        XXkI = np.dot(X.T, X)
#        index = np.arange(min(XXkI.shape))
#        XXkI[index, index] += self.g.k
#        invXXkI = np.linalg.inv(XXkI)
#        print "t:", time() - t
#        beta = np.dot(invXXkI, Xty_grad)

        if self._XtinvXXtkI is None:
            XXtkI = np.dot(self.X, self.X.T)
            index = np.arange(min(XXtkI.shape))
            XXtkI[index, index] += self.g.k
            invXXtkI = np.linalg.inv(XXtkI)
            self._XtinvXXtkI = np.dot(self.X.T, invXXtkI)

        beta = (Xty_grad - np.dot(self._XtinvXXtkI, np.dot(self.X, Xty_grad)))

        return beta

    def gap(self, beta, beta_hat):
        """Compute the duality gap.

        From the interface "DualFunction".
        """
        # TODO: Add this function!
        raise NotImplementedError("We cannot currently do this!")

    def estimate_mu(self, beta):
        """Computes a "good" value of \mu with respect to the given \beta.

        From the interface "NesterovFunction".
        """
        raise NotImplementedError("We do not use this here!")

    def M(self):
        """ The maximum value of the regularisation of the dual variable. We
        have

            M = max_{\alpha \in K} 0.5*|\alpha|²_2.

        From the interface "NesterovFunction".
        """
        return self.h.M()

    def project(self, a):
        """ Projection onto the compact space of the Nesterov function.

        From the interface "NesterovFunction".
        """
        return self.h.project(a)