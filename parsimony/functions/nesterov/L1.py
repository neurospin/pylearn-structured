# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.functions.nesterov.L1` module contains the loss function
for the L1 penalty, smoothed using Nesterov's technique.

Created on Mon Feb  3 17:00:56 2014

@author:  Tommy Löfstedt, Vincent Guillemot, Edouard Duchesnay and
          Fouad Hadj-Selem
@email:   lofstedt.tommy@gmail.com, edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
import parsimony.functions.interfaces as interfaces
from interfaces import NesterovFunction
import parsimony.utils.consts as consts
import parsimony.utils.maths as maths

__all__ = ["L1"]


class L1(interfaces.AtomicFunction,
         interfaces.Constraint,
         NesterovFunction,
         interfaces.Gradient,
         interfaces.LipschitzContinuousGradient):
    """The proximal operator of the smoothed L1 function

        f(\beta) = l * (L1mu(\beta) - c),

    where L1mu(\beta) is the smoothed L1 function. The constrained version has
    the form

        ||\beta||_1 <= c.

    Parameters
    ----------
    l : The Lagrange multiplier, or regularisation constant, of the function.

    c : The limit of the constraint. The function is feasible if
            ||\beta||_1 <= c. The default value is c=0, i.e. the default is a
            regularisation formulation.

    A : The linear operator for the Nesterov formulation. May not be None.

    mu : The regularisation constant for the smoothing.
    """
    def __init__(self, l, c=0.0, A=None, mu=0.0):

        super(SmoothedL1, self).__init__(l, c, A, mu)

    def f(self, beta):
        """ Function value.
        """
        if self.l < consts.TOLERANCE:
            return 0.0

        return self.l * (maths.norm1(beta) - self.c)

    def feasible(self, beta):
        """Feasibility of the constraint.

        From the interface "Constraint".
        """
        return maths.norm1(beta) <= self.c

    def phi(self, alpha, beta):
        """ Function value with known alpha.

        From the interface "NesterovFunction".
        """
        if self.l < consts.TOLERANCE:
            return 0.0

        return self.l * ((np.dot(alpha[0].T, beta)[0, 0]
                         - (self.mu / 2.0) * np.sum(alpha[0] ** 2.0)) - self.c)

    def grad(self, beta):
        """ Gradient of the function at beta.

        From the interface "Gradient". Overloaded since we can be faster than
        the default.
        """
        alpha = self.alpha(beta)

        return self.l * alpha[0]

    def L(self):
        """ Lipschitz constant of the gradient.

        From the interface "LipschitzContinuousGradient".
        """
        return self.l / self.mu

    def alpha(self, beta):
        """ Dual variable of the Nesterov function.

        From the interface "NesterovFunction". Overloaded since we can be
        faster than the default.
        """
        alpha = self.project([beta / self.mu])

        return alpha

    @staticmethod
    def project(a):
        """ Projection onto the compact space of the Nesterov function.

        From the interface "NesterovFunction".
        """
        a = a[0]
        anorm = np.abs(a)
        i = anorm > 1.0
        anorm_i = anorm[i]
        a[i] = np.divide(a[i], anorm_i)

        return [a]

    def M(self):
        """ The maximum value of the regularisation of the dual variable. We
        have

            M = max_{\alpha \in K} 0.5*|\alpha|²_2.

        From the interface "NesterovFunction".
        """
        A = self.A()
        return A[0].shape[0] / 2.0

    def estimate_mu(self, beta):
        """ Computes a "good" value of \mu with respect to the given \beta.

        From the interface "NesterovFunction".
        """
        return np.max(np.absolute(beta))


class SmoothedL1TV(interfaces.AtomicFunction,
                   interfaces.Regularisation,
                   interfaces.NesterovFunction,
                   interfaces.Eigenvalues):
    """
    Parameters
    ----------
    l : L1 parameter.
            The Lagrange multiplier, or regularisation constant, of the
            function.

    g : Total Variation parameter
            The Lagrange multiplier, or regularisation constant, of the
            function.

    Atv : The linear operator for the total variation Nesterov function

    Al1 : Matrix allocation for regression

    mu: The regularisation constant for the smoothing
    """
    def __init__(self, l, g, Atv=None, Al1=None, mu=0.0):

        self.l = float(l)
        self.g = float(g)

        self._p = Atv[0].shape[1]  # WARNING: Number of rows may differ from p.
        if Al1 is None:
            Al1 = sparse.eye(self._p, self._p)
        self._A = [l * Al1,
                   g * Atv[0],
                   g * Atv[1],
                   g * Atv[2]]

        self.mu = float(mu)

        # TODO: Is reset still necessary?
        self.reset()

    def reset(self):

        self._lambda_max = None

    def f(self, beta):
        """ Function value.
        """
        if self.l < consts.TOLERANCE and self.g < consts.TOLERANCE:
            return 0.0

        A = self.A()
        return maths.norm1(A[0].dot(beta)) + \
               np.sum(np.sqrt(A[1].dot(beta) ** 2.0 +
                              A[2].dot(beta) ** 2.0 +
                              A[3].dot(beta) ** 2.0))

    def phi(self, alpha, beta):
        """ Function value with known alpha.

        From the interface "NesterovFunction".
        """
        if self.l < consts.TOLERANCE and self.g < consts.TOLERANCE:
            return 0.0

        Aa = self.Aa(alpha)

        alpha_sqsum = 0.0
        for a in alpha:
            alpha_sqsum += np.sum(a ** 2.0)

        return np.dot(beta.T, Aa)[0, 0] - (self.mu / 2.0) * alpha_sqsum

    def lambda_max(self):
        """ Largest eigenvalue of the corresponding covariance matrix.

        From the interface "Eigenvalues".
        """
        # Note that we can save the state here since lmax(A) does not change.
        if len(self._A) == 4 \
                and self._A[2].nnz == 0 and self._A[3].nnz == 0:
#        if len(self._shape) == 3 \
#            and self._shape[0] == 1 and self._shape[1] == 1:
            # TODO: Instead of p, this should really be the number of non-zero
            # rows of A.
            p = self._A[1].shape[0]
            lmaxTV = 2.0 * (1.0 - math.cos(float(p - 1) * math.pi
                                           / float(p)))
            self._lambda_max = lmaxTV * self.g ** 2.0 + self.l ** 2.0

        elif self._lambda_max is None:

            from parsimony.algorithms import FastSparseSVD

#            A = sparse.vstack(self.A())
#            v = algorithms.FastSparseSVD(A, max_iter=max_iter)
#            us = A.dot(v)
#            self._lambda_max = np.sum(us ** 2.0)

            A = sparse.vstack(self.A()[1:])
            # TODO: Add max_iter here!!
            v = FastSparseSVD()(A)  # , max_iter=max_iter)
            us = A.dot(v)
            self._lambda_max = np.sum(us ** 2.0) + self.l ** 2.0

        return self._lambda_max

#    """ Linear operator of the Nesterov function.
#
#    From the interface "NesterovFunction".
#    """
#    def A(self):
#
#        return self._A

#    """ Computes A^\T\alpha.
#
#    From the interface "NesterovFunction".
#    """
#    def Aa(self, alpha):
#
#        A = self.A()
#        Aa = A[0].T.dot(alpha[0])
#        for i in xrange(1, len(A)):
#            Aa += A[i].T.dot(alpha[i])
#
#        return Aa

    def alpha(self, beta):
        """ Dual variable of the Nesterov function.

        From the interface "NesterovFunction". Overloaded since we need to do
        more than the default.
        """
        A = self.A()

        a = [0] * len(A)
        a[0] = (1.0 / self.mu) * A[0].dot(beta)
        a[1] = (1.0 / self.mu) * A[1].dot(beta)
        a[2] = (1.0 / self.mu) * A[2].dot(beta)
        a[3] = (1.0 / self.mu) * A[3].dot(beta)
        # Remember: lambda and gamma are already in the A matrices.

        return self.project(a)

    def project(self, a):
        """ Projection onto the compact space of the Nesterov function.

        From the interface "NesterovFunction".
        """
        # L1
        al1 = a[0]
        anorm_l1 = np.abs(al1)
        i_l1 = anorm_l1 > 1.0
        anorm_l1_i = anorm_l1[i_l1]
        al1[i_l1] = np.divide(al1[i_l1], anorm_l1_i)

        # TV
        ax = a[1]
        ay = a[2]
        az = a[3]
        anorm_tv = ax ** 2.0 + ay ** 2.0 + az ** 2.0
        i_tv = anorm_tv > 1.0

        anorm_tv_i = anorm_tv[i_tv] ** 0.5  # Square root taken here. Faster.
        ax[i_tv] = np.divide(ax[i_tv], anorm_tv_i)
        ay[i_tv] = np.divide(ay[i_tv], anorm_tv_i)
        az[i_tv] = np.divide(az[i_tv], anorm_tv_i)

        return [al1, ax, ay, az]

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
        A = self.A()

        return (A[0].shape[0] / 2.0) \
             + (A[1].shape[0] / 2.0)