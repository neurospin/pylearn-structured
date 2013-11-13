# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 12:06:07 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: TBD.
"""

import numpy as np
from utils import TOLERANCE
from utils import U
from utils import norm2
import scipy.sparse as sparse

__all__ = ['grad_L1', 'grad_L1mu', 'grad_L2', 'grad_norm2',
           'grad_TV', 'grad_TVmu']


def grad_L1(beta):

    p = beta.shape[0]

    grad = np.zeros((p, 1))
    for i in range(p):
        if beta[i, 0] > TOLERANCE:
            grad[i, 0] = 1.0
        elif beta[i, 0] < -TOLERANCE:
            grad[i, 0] = -1.0
        else:
            grad[i, 0] = U(-1, 1)

    return grad


def grad_L1mu(beta, mu):

    alpha = (1.0 / mu) * beta
    asnorm = np.abs(alpha)
    i = asnorm > 1.0
    alpha[i] = np.divide(alpha[i], asnorm[i])

    return alpha


def grad_L2(beta):

    return beta


def grad_norm2(beta):

    norm_beta = norm2(beta)
    if norm_beta > TOLERANCE:
        return beta / norm_beta
    else:
        D = beta.shape[0]
        u = (np.random.rand(D, 1) * 2.0) - 1.0  # [-1, 1]^D
        norm_u = norm2(u)
        a = np.random.rand()  # [0, 1]
        return u * (a / norm_u)


def _generate_A(shape):

    D = len(shape)
    p = np.prod(shape)
    dims = shape + (1,)
    A = [0] * D
    for i in xrange(D - 1, -1, -1):
        shift = np.prod(dims[i + 1:])
        A[D - i - 1] = np.eye(p, p, shift) - np.eye(p, p)

    # TODO: Only works for up to 3 dimensions ...
    ind = np.reshape(xrange(p), shape)
    xind = ind[:, :, -1].flatten().tolist()
    yind = ind[:, -1, :].flatten().tolist()
    zind = ind[-1, :, :].flatten().tolist()

    for i in xind:
        A[0][i, :] = 0
    for i in yind:
        A[1][i, :] = 0
    for i in zind:
        A[2][i, :] = 0

    return A


def _generate_Ai(i, A, shape):

    D = len(shape)
    v = []
    for k in xrange(D):
        v.append(A[k][i, :])

    Ai = np.vstack(v)

#    Ai = np.zeros((D, p))
#    for d in xrange(D - 1, -1, -1):
#        if d == D - 1:
#            x = i % shape[d]
#            if x + 1 < shape[d]:
#                Ai[D - d - 1, i] = -1
#                Ai[D - d - 1, i + 1] = 1
#        else:
#            b = np.prod(shape[d + 1:])
#            y = int(i / b)
#            if y + 1 < shape[d]:
#                Ai[D - d - 1, i] = -1
#                Ai[D - d - 1, i + b] = 1

    return Ai


def grad_TV(beta, shape):

    p = np.prod(shape)
#    D = len(shape)

    A = _generate_A(shape)

    grad = np.zeros((p, 1))
    for i in range(p):
#        Ai = np.zeros((D, p))
#        for d in range(D):
#            if i < p:
#                b = np.prod([shape[-j] for j in range(1, d + 1)])
#                if b + i < p:
#                    Ai[d, i] = -1
#                    Ai[d, b + i] = 1

        Ai = _generate_Ai(i, A, shape)

#        print "i:", i, "Ai:\n", np.reshape(Ai[0, :] + Ai[1, :] + Ai[2, :],
#                                           shape)
#        print "i:", i, "Ai:\n", np.reshape(Ai[0, :], shape)
#        print np.reshape(Ai[1, :], shape)
#        print np.reshape(Ai[2, :], shape)

        gradnorm2 = grad_norm2(np.dot(Ai, beta))
        grad += np.dot(Ai.T, gradnorm2)

    return grad


def grad_TVmu(beta, shape, mu):

    p = np.prod(shape)
#    D = len(shape)

    A = _generate_A(shape)

    grad = np.zeros((p, 1))
    for i in range(p):

        Ai = _generate_Ai(i, A, shape)

        alphai = np.dot(Ai, beta) / mu
        anorm = np.sqrt(np.sum(alphai ** 2.0))
        if anorm > 1.0:
            alphai /= anorm

        grad += np.dot(Ai.T, alphai)

    return grad

def _find_mask_ind(mask, ind):

    xshift = np.concatenate((mask[:, :, 1:], -np.ones((mask.shape[0],
                                                      mask.shape[1],
                                                      1))),
                            axis=2)
    yshift = np.concatenate((mask[:, 1:, :], -np.ones((mask.shape[0],
                                                      1,
                                                      mask.shape[2]))),
                            axis=1)
    zshift = np.concatenate((mask[1:, :, :], -np.ones((1,
                                                      mask.shape[1],
                                                      mask.shape[2]))),
                            axis=0)

    xind = ind[(mask - xshift) > 0]
    yind = ind[(mask - yshift) > 0]
    zind = ind[(mask - zshift) > 0]

    return xind.flatten().tolist(), \
           yind.flatten().tolist(), \
           zind.flatten().tolist()


def delete_sparse_csr_row(mat, i):
    """Delete row i in-place from sparse matrix mat (CSR format).

    Implementation from:

        http://stackoverflow.com/questions/13077527/is-there-a-numpy-delete-equivalent-for-sparse-matrices
    """
    if not isinstance(mat, sparse.csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")
    n = mat.indptr[i + 1] - mat.indptr[i]
    if n > 0:
        mat.data[mat.indptr[i]:-n] = mat.data[mat.indptr[i + 1]:]
        mat.data = mat.data[:-n]
        mat.indices[mat.indptr[i]:-n] = mat.indices[mat.indptr[i + 1]:]
        mat.indices = mat.indices[:-n]
    mat.indptr[i:-1] = mat.indptr[i + 1:]
    mat.indptr[i:] -= n
    mat.indptr = mat.indptr[:-1]
    mat._shape = (mat._shape[0] - 1, mat._shape[1])



def generate_A_from_mask(mask):
    while len(mask.shape) < 3:
        mask = mask[:, np.newaxis]
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
    # mapping from image coordinate to flat masked array
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
    return Ax, Ay, Az, n_compacts

def  _generate_sparse_masked_A(shape,mask):
    Z = shape[0]
    Y = shape[1]
    X = shape[2]
    p = X * Y * Z

    smtype = 'csr'
    Ax = sparse.eye(p, p, 1, format=smtype) - sparse.eye(p, p)
    Ay = sparse.eye(p, p, X, format=smtype) - sparse.eye(p, p)
    Az = sparse.eye(p, p, X * Y, format=smtype) - sparse.eye(p, p)

    ind = np.reshape(xrange(p), (Z, Y, X))
    if mask != None:
        _mask = np.reshape(mask, (Z, Y, X))
        xind, yind, zind = _find_mask_ind(_mask, ind)
    else:
        xind = ind[:, :, -1].flatten().tolist()
        yind = ind[:, -1, :].flatten().tolist()
        zind = ind[-1, :, :].flatten().tolist()

    for i in xrange(len(xind)):
        Ax.data[Ax.indptr[xind[i]]: \
                Ax.indptr[xind[i] + 1]] = 0
    Ax.eliminate_zeros()

    for i in xrange(len(yind)):
        Ay.data[Ay.indptr[yind[i]]: \
                Ay.indptr[yind[i] + 1]] = 0
    Ay.eliminate_zeros()

    Az.data[Az.indptr[zind[0]]: \
            Az.indptr[zind[-1] + 1]] = 0
    Az.eliminate_zeros()

    # Remove rows corresponding to indices excluded in all dimensions
    toremove = list(set(xind).intersection(yind).intersection(zind))
    toremove.sort()
    # Remove from the end so that indices are not changed
    toremove.reverse()
    for i in toremove:
        delete_sparse_csr_row(Ax, i)
        delete_sparse_csr_row(Ay, i)
        delete_sparse_csr_row(Az, i)

    # Remove columns of A corresponding to masked-out variables
    if mask != None:
        Ax = Ax.T.tocsr()
        Ay = Ay.T.tocsr()
        Az = Az.T.tocsr()
        for i in reversed(xrange(p)):
            if mask[i] == 0:
                delete_sparse_csr_row(Ax, i)
                delete_sparse_csr_row(Ay, i)
                delete_sparse_csr_row(Az, i)

        Ax = Ax.T
        Ay = Ay.T
        Az = Az.T

    return [Ax, Ay, Az]

def _generate_sparse_Ai(i, A, shape):

    D = len(shape)
    v = []
    for k in xrange(D):
        v.append(A[k].getrow(i))

    Ai = sparse.vstack(v)
    return Ai

def grad_TV_sparse(beta, shape, mask):

    p = np.prod(shape)
    if mask != None:
        A = generate_A_from_mask(mask)
    else:
        A = _generate_sparse_masked_A(shape,mask)
    grad = np.zeros((p, 1))
    for i in xrange(p):
        #print i
        Ai = _generate_sparse_Ai(i, A, shape)
        gradnorm2 = grad_norm2(Ai.dot(beta))
        grad += Ai.transpose().dot(gradnorm2)

    return grad

