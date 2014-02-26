# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 14:54:28 2014

@author:  Fouad Hadj Selem, Edouard Duchesnay
@email:   edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
from .. import interfaces
from ..nesterov import interfaces as nesterov_interfaces
from . import interfaces as multiblock_interfaces


__all__ = ["PCA_L1_TV"]


class PCA_L1_TV(interfaces.CompositeFunction,
               interfaces.Gradient,
               interfaces.LipschitzContinuousGradient,
               nesterov_interfaces.NesterovFunction,
               interfaces.ProximalOperator,
               interfaces.Continuation,
               interfaces.DualFunction,
               interfaces.StronglyConvex):
    """Combination (sum) of RidgeRegression, L1 and TotalVariation
    """
    def __init__(self, X, k, l, g, A=None, mu=0.0, penalty_start=0):
        """
        Parameters:
        ----------
        X : Numpy array. The X matrix for the ridge regression.

        k : Non-negative float. The Lagrange multiplier, or regularisation
                constant, for the ridge penalty.

        l : Non-negative float. The Lagrange multiplier, or regularisation
                constant, for the L1 penalty.

        g : Non-negative float. The Lagrange multiplier, or regularisation
                constant, of the smoothed TV function.

        A : Numpy array (usually sparse). The linear operator for the Nesterov
                formulation of TV. May not be None!

        mu : Non-negative float. The regularisation constant for the smoothing
                of the TV function.

        penalty_start : Non-negative integer. The number of columns, variables
                etc., to except from penalisation. Equivalently, the first
                index to be penalised. Default is 0, all columns are included.
        """
        self.X = X
        #from parsimony.algorithms.multiblock import MultiblockProjectedGradientMethod
        from  parsimony.functions.multiblock.losses import LatentVariableCovariance
        self.pca = LatentVariableCovariance(X)
        self.l1 = L1(l, penalty_start=penalty_start)
        self.tv = TotalVariation(g, A=A, mu=mu, penalty_start=penalty_start)

        self.reset()

    def reset(self):

        self.pca.reset()
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

        Parameters:
        ----------
        mu : Non-negative float. The regularisation constant for the smoothing
                to use from now on.

        Returns:
        -------
        old_mu : Non-negative float. The old regularisation constant for the
                smoothing that was overwritten and is no longer used.
        """
        return self.tv.set_mu(mu)

    def f(self, beta):
        """Function value.
        """
        return self.pca.f(beta) \
             + self.l1.f(beta) \
             + self.tv.f(beta)

    def fmu(self, beta, mu=None):
        """Function value.
        """
        return self.pca.f(beta) \
             + self.l1.f(beta) \
             + self.tv.fmu(beta, mu)

    def phi(self, alpha, beta):
        """ Function value with known alpha.
        """
        return self.pca.f(beta) \
             + self.l1.f(beta) \
             + self.tv.phi(alpha, beta)

    def grad(self, beta):
        """Gradient of the differentiable part of the function.

        From the interface "Gradient".
        """
        return self.pca.grad(beta) \
             + self.tv.grad(beta)

    def L(self):
        """Lipschitz constant of the gradient.

        From the interface "LipschitzContinuousGradient".
        """
        return self.pca.L() \
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
        raise NotImplementedError('Abstract method "betahat" must be '
                                  'specialised!')
#        if self._Xty is None:
#            self._Xty = np.dot(self.X.T, self.y)
#
#        Ata_tv = self.tv.l * self.tv.Aa(alphak)
#        Ata_l1 = self.l1.l * SmoothedL1.project([betak / consts.TOLERANCE])[0]
#        v = (self._Xty - Ata_tv - Ata_l1)
#
#        shape = self.X.shape
#
#        if shape[0] > shape[1]:  # If n > p
#
#            # Ridge solution
#            if self._invXXkI is None:
#                XtXkI = np.dot(self.X.T, self.X)
#                index = np.arange(min(XtXkI.shape))
#                XtXkI[index, index] += self.rr.k
#                self._invXXkI = np.linalg.inv(XtXkI)
#
#            beta_hat = np.dot(self._invXXkI, v)
#
#        else:  # If p > n
#            # Ridge solution using the Woodbury matrix identity:
#            if self._XtinvXXtkI is None:
#                XXtkI = np.dot(self.X, self.X.T)
#                index = np.arange(min(XXtkI.shape))
#                XXtkI[index, index] += self.rr.k
#                invXXtkI = np.linalg.inv(XXtkI)
#                self._XtinvXXtkI = np.dot(self.X.T, invXXtkI)
#
#            beta_hat = (v - np.dot(self._XtinvXXtkI, np.dot(self.X, v))) \
#                       / self.rr.k
#
#        return beta_hat

    def gap(self, beta, beta_hat=None):
        """Compute the duality gap.

        From the interface "DualFunction".
        """
        raise NotImplementedError('Abstract method "gap" must be '
                                  'specialised!')
#        alpha = self.tv.alpha(beta)
#
#        P = self.rr.f(beta) \
#          + self.l1.f(beta) \
#          + self.tv.phi(alpha, beta)
#
#        beta_hat = self.betahat(alpha, beta)
#
#        D = self.rr.f(beta_hat) \
#          + self.l1.f(beta_hat) \
#          + self.tv.phi(alpha, beta_hat)
#
#        return P - D

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

    def parameter(self):
        """Returns the strongly convex parameter for the function.

        From the interface "StronglyConvex".
        """
        return self.rr.k
