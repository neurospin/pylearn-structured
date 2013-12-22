# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 12:06:07 2013

@email:   tommy.loefstedt@cea.fr, edouard.duchesnay@cea.fr
@license: TBD.
"""

import numpy as np
from utils import TOLERANCE
from utils import RandomUniform, ConstantValue
from utils import norm2
import scipy.sparse as sparse

__all__ = ['grad_l1', 'grad_l1mu', 'grad_l2', 'grad_norm2',
           'grad_tv', 'grad_tvmu']


def grad_l1(beta, rnd=RandomUniform(-1, 1)):
    grad = np.zeros((beta.shape[0], 1))
    grad[beta > TOLERANCE] = 1.
    grad[beta < -TOLERANCE] = -1.
    between = (beta >= -TOLERANCE) & (beta < TOLERANCE)
    grad[between] = rnd(between.sum())
    return grad

def grad_l1mu(beta, mu):

    alpha = (1.0 / mu) * beta
    asnorm = np.abs(alpha)
    i = asnorm > 1.0
    alpha[i] = np.divide(alpha[i], asnorm[i])

    return alpha


def grad_l2(beta):

    return beta


## TODO check grad_tv with grad_tv_tommy, see code in __main__
## if ok remove grad_tv_tommy, _generate_A, _generate_Ai and modify
## modify grad_tvmu
def grad_tv(beta, A, rnd=np.random.rand):
    beta_flat = beta.ravel()
    Ab = np.vstack([Ai.dot(beta_flat) for Ai in A]).T
    Ab_norm2 = np.sqrt(np.sum(Ab ** 2.0, axis=1))    
    upper = Ab_norm2 > TOLERANCE
    grad_Ab_norm2 = Ab
    grad_Ab_norm2[upper] = (Ab[upper].T / Ab_norm2[upper]).T
    lower = Ab_norm2 <= TOLERANCE
    n_lower = lower.sum()
    if n_lower:
        D = len(A)
        vec_rnd = (rnd(n_lower, D) * 2.0) - 1.0
        norm_vec = np.sqrt(np.sum(vec_rnd ** 2.0, axis=1))
        a = rnd(n_lower)
        grad_Ab_norm2[lower] = (vec_rnd.T * (a / norm_vec)).T
    grad = np.vstack([A[i].T.dot(grad_Ab_norm2[:, i]) for i in xrange(len(A))])
    grad = grad.sum(axis=0)
    return grad.reshape(beta.shape)


def grad_norm2(beta, rnd=np.random.rand):

    norm_beta = norm2(beta)
    if norm_beta > TOLERANCE:
        return beta / norm_beta
    else:
        D = beta.shape[0]
        u = (rnd(D, 1) * 2.0) - 1.0  # [-1, 1]^D
        norm_u = norm2(u)
        a = rnd()  # [0, 1]
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
    return Ai


def grad_tv_tommy(beta, shape, rnd=np.random.rand):

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

        gradnorm2 = grad_norm2(np.dot(Ai, beta), rnd=rnd)
        grad += np.dot(Ai.T, gradnorm2)

    return grad


def grad_tvmu(beta, shape, mu):

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


def _generate_sparse_Ai(i, A, shape):

    D = len(shape)
    v = []
    for k in xrange(D):
        v.append(A[k].getrow(i))

    Ai = sparse.vstack(v)
    return Ai


if __name__ == '__main__':
    from parsimony.datasets import make_regression_struct
    import parsimony.tv
    import numpy as np
    seed = 1
    shape = (100, 100, 1)
    Xim, y, beta = make_regression_struct(n_samples=100, shape=shape, random_seed=seed)
    Ax, Ay, Az, n_compacts = parsimony.tv.tv_As_from_shape(shape)
    A = (Ax, Ay, Az)
    beta = beta.ravel()[:, np.newaxis]
    #seed = np.random.randint(10)
    #np.random.seed(seed)
    g1 = grad_tv(beta, A, rnd=ConstantValue(0))
    #g1 = grad_tv(beta, A)
    #plt.matshow(beta.reshape(shape[:2]));plt.show()
    #plt.matshow(g1.reshape(shape[:2]));plt.show()
    #np.random.seed(seed)
    g2 = grad_tv_tommy(beta, shape, rnd=ConstantValue(0))
    #g2 = grad_tv(beta, shape)
    print np.allclose(g1, g2)
