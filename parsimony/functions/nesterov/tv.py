# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.functions.nesterov.tv` module contains the loss function
and helper functions for Total variation, TV, smoothed using Nesterov's
technique.

Created on Mon Feb  3 10:46:47 2014

@author:  Tommy Löfstedt, Edouard Duchesnay
@email:   lofstedt.tommy@gmail.com, edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
import math

import scipy.sparse as sparse
import numpy as np

from .interfaces import NesterovFunction
from .. import interfaces
import parsimony.utils.consts as consts
import parsimony.utils.maths as maths

__all__ = ["TotalVariation", "A_from_mask", "A_from_shape"]


class TotalVariation(interfaces.AtomicFunction,
                     NesterovFunction,
                     interfaces.Penalty,
                     interfaces.Constraint,
                     interfaces.Gradient,
                     interfaces.LipschitzContinuousGradient):
    """The smoothed Total variation (TV) function

        f(beta) = l * (TV(beta) - c),

    where TV(beta) is the smoothed L1 function. The constrained version has the
    form

        TV(beta) <= c.
    """
    def __init__(self, l, c=0.0, A=None, mu=0.0, penalty_start=0):
        """
        Parameters
        ----------
        l : Non-negative float. The Lagrange multiplier, or regularisation
                constant, of the function.

        c : Float. The limit of the constraint. The function is feasible if
                TV(beta) <= c. The default value is c=0, i.e. the default is a
                regularisation formulation.

        A : Numpy array (usually sparse). The linear operator for the Nesterov
                formulation. May not be None!

        mu : Non-negative float. The regularisation constant for the smoothing.

        penalty_start : Non-negative integer. The number of columns, variables
                etc., to exempt from penalisation. Equivalently, the first
                index to be penalised. Default is 0, all columns are included.
        """
        super(TotalVariation, self).__init__(l, A=A, mu=mu,
                                             penalty_start=penalty_start)
        self._p = A[0].shape[1]

        self.c = float(c)

        self.reset()

    def reset(self):

        self._lambda_max = None

    def f(self, beta):
        """ Function value.
        """
        if self.l < consts.TOLERANCE:
            return 0.0

        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        A = self.A()
        return self.l * (np.sum(np.sqrt(A[0].dot(beta_) ** 2.0 +
                                        A[1].dot(beta_) ** 2.0 +
                                        A[2].dot(beta_) ** 2.0)) - self.c)

    def phi(self, alpha, beta):
        """Function value with known alpha.

        From the interface "NesterovFunction".
        """
        if self.l < consts.TOLERANCE:
            return 0.0

        Aa = self.Aa(alpha)

        alpha_sqsum = 0.0
        for a in alpha:
            alpha_sqsum += np.sum(a ** 2.0)

        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        return self.l * ((np.dot(beta_.T, Aa)[0, 0]
                          - (self.mu / 2.0) * alpha_sqsum) - self.c)

    def feasible(self, beta):
        """Feasibility of the constraint.

        From the interface "Constraint".
        """

        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        A = self.A()
        val = np.sum(np.sqrt(A[0].dot(beta_) ** 2.0 +
                             A[1].dot(beta_) ** 2.0 +
                             A[2].dot(beta_) ** 2.0))
        return val <= self.c

    def L(self):
        """ Lipschitz constant of the gradient.

        From the interface "LipschitzContinuousGradient".
        """
        if self.l < consts.TOLERANCE:
            return 0.0

        lmaxA = self.lambda_max()

        return self.l * lmaxA / self.mu

    def lambda_max(self):
        """ Largest eigenvalue of the corresponding covariance matrix.

        From the interface "Eigenvalues".
        """
        # Note that we can save the state here since lmax(A) does not change.
        # TODO: This only work if the elements of self._A are scipy.sparse. We
        # should allow dense matrices as well.
        if len(self._A) == 3 \
                and self._A[1].nnz == 0 and self._A[2].nnz == 0:
            # TODO: Instead of p, this should really be the number of non-zero
            # rows of A.
            self._lambda_max = 2.0 * (1.0 - math.cos(float(self._p - 1)
                                                     * math.pi
                                                     / float(self._p)))

        elif self._lambda_max is None:

            from parsimony.algorithms.implicit import FastSparseSVD

            A = sparse.vstack(self.A())
            # TODO: Add max_iter here!
            v = FastSparseSVD().run(A)  # , max_iter=max_iter)
            us = A.dot(v)
            self._lambda_max = np.sum(us ** 2.0)

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

#    """ Dual variable of the Nesterov function.
#
#    From the interface "NesterovFunction".
#    """
#    def alpha(self, beta):
#
#        # Compute a*
#        A = self.A()
#        alpha = [0] * len(A)
#        for i in xrange(len(A)):
#            alpha[i] = A[i].dot(beta) / self.mu
#
#        # Apply projection
#        alpha = self.project(alpha)
#
#        return alpha

    def project(self, a):
        """ Projection onto the compact space of the Nesterov function.

        From the interface "NesterovFunction".
        """
        ax = a[0]
        ay = a[1]
        az = a[2]
        anorm = ax ** 2.0 + ay ** 2.0 + az ** 2.0
        i = anorm > 1.0

        anorm_i = anorm[i] ** 0.5  # Square root is taken here. Faster.
        ax[i] = np.divide(ax[i], anorm_i)
        ay[i] = np.divide(ay[i], anorm_i)
        az[i] = np.divide(az[i], anorm_i)

        return [ax, ay, az]

    def M(self):
        """ The maximum value of the regularisation of the dual variable. We
        have

            M = max_{alpha in K} 0.5*|alpha|²_2.

        From the interface "NesterovFunction".
        """
        return self._A[0].shape[0] / 2.0

    def estimate_mu(self, beta):
        """ Computes a "good" value of mu with respect to the given beta.

        From the interface "NesterovFunction".
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        SS = 0.0
        A = self.A()
        for i in xrange(len(A)):
            SS = max(SS, maths.norm(A[i].dot(beta_)))

        return np.max(np.sqrt(SS))


def A_from_mask(mask):
    """Generates the linear operator for the total variation Nesterov function
    from a mask for a 3D image.

    Parameters
    ----------
    mask : Numpy array. The mask. The mask does not involve any intercept
            variables.
    """
    while len(mask.shape) < 3:
        mask = mask[:, np.newaxis]
    # TODO: This is the wrong order of the dimensions. They should be reversed.
    nx, ny, nz = mask.shape
    mask = mask.astype(bool)
    xyz_mask = np.where(mask)
    Ax_i = list()
    Ax_j = list()
    Ax_v = list()
    Ay_i = list()
    Ay_j = list()
    Ay_v = list()
    Az_i = list()
    Az_j = list()
    Az_v = list()
    n_compacts = 0
    p = np.sum(mask)
    # Mapping from image coordinate to flat masked array.
    im2flat = np.zeros(mask.shape, dtype=int)
    im2flat[:] = -1
    im2flat[mask] = np.arange(p)
    for pt in xrange(len(xyz_mask[0])):
        found = False
        x, y, z = xyz_mask[0][pt], xyz_mask[1][pt], xyz_mask[2][pt]
        i_pt = im2flat[x, y, z]
        if x + 1 < nx and mask[x + 1, y, z]:
            found = True
            Ax_i += [i_pt, i_pt]
            Ax_j += [i_pt, im2flat[x + 1, y, z]]
            Ax_v += [-1., 1.]
        if y + 1 < ny and mask[x, y + 1, z]:
            found = True
            Ay_i += [i_pt, i_pt]
            Ay_j += [i_pt, im2flat[x, y + 1, z]]
            Ay_v += [-1., 1.]
        if z + 1 < nz and mask[x, y, z + 1]:
            found = True
            Az_i += [i_pt, i_pt]
            Az_j += [i_pt, im2flat[x, y, z + 1]]
            Az_v += [-1., 1.]
        if found:
            n_compacts += 1
    Ax = sparse.csr_matrix((Ax_v, (Ax_i, Ax_j)), shape=(p, p))
    Ay = sparse.csr_matrix((Ay_v, (Ay_i, Ay_j)), shape=(p, p))
    Az = sparse.csr_matrix((Az_v, (Az_i, Az_j)), shape=(p, p))

    return [Ax, Ay, Az], n_compacts


def A_from_shape(shape):
    """Generates the linear operator for the total variation Nesterov function
    from the shape of a 3D image.

    Parameters
    ----------
    shape : List or tuple with 1, 2 or 3 elements. The shape of the 1D, 2D or
            3D image. shape has the form (X, Y, Z), where Z is the number of
            "layers", Y is the number of rows and X is the number of columns.
            The shape does not involve any intercept variables.
    """
    while len(shape) < 3:
        shape = tuple(list(shape) + [1])
    nz = shape[0]
    ny = shape[1]
    nx = shape[2]
    p = nx * ny * nz
    ind = np.arange(p).reshape((nz, ny, nx))
    if nx > 1:
        Ax = sparse.eye(p, p, 1, format='csr') - \
             sparse.eye(p, p)
        zind = ind[:, :, -1].ravel()
        for i in zind:
            Ax.data[Ax.indptr[i]: \
                    Ax.indptr[i + 1]] = 0
        Ax.eliminate_zeros()
    else:
        Ax = sparse.csc_matrix((p, p), dtype=float)
    if ny > 1:
        Ay = sparse.eye(p, p, nx, format='csr') - \
             sparse.eye(p, p)
        yind = ind[:, -1, :].ravel()
        for i in yind:
            Ay.data[Ay.indptr[i]: \
                    Ay.indptr[i + 1]] = 0
        Ay.eliminate_zeros()
    else:
        Ay = sparse.csc_matrix((p, p), dtype=float)
    if nz > 1:
        Az = (sparse.eye(p, p, ny * nx, format='csr') - \
              sparse.eye(p, p))
        xind = ind[-1, :, :].ravel()
        for i in xind:
            Az.data[Az.indptr[i]: \
                    Az.indptr[i + 1]] = 0
        Az.eliminate_zeros()
    else:
        Az = sparse.csc_matrix((p, p), dtype=float)

    return [Ax, Ay, Az], (nz * ny * nx - 1)