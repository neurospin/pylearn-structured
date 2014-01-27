# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 18:48:16 2013

@author:  Edouard Duchesnay
@email:   edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
import numpy as np
import scipy.sparse as sparse

__all__ = ["A_from_mask", "A_from_shape"]


def A_from_mask(mask):
    """Generates the linear operator for the total variation Nesterov function
    from a mask for a 3D image.
    """
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
    return [Ax, Ay, Az], n_compacts


def A_from_shape(shape):
    """Generates the linear operator for the total variation Nesterov function
    from the shape of a 3D image.
    """
    while len(shape) < 3:
        shape = tuple(list(shape) + [1])
    nx = shape[0]
    ny = shape[1]
    nz = shape[2]
    p = nz * ny * nx
    ind = np.arange(p).reshape((nx, ny, nz))
    if nx > 1:
        Ax = (sparse.eye(p, p, ny * nz, format='csr') -\
              sparse.eye(p, p))
        xind = ind[-1, :, :].ravel()
        for i in xind:
            Ax.data[Ax.indptr[i]: \
                    Ax.indptr[i + 1]] = 0
        Ax.eliminate_zeros()
    else:
        Ax = sparse.csc_matrix((p, p), dtype=float)
    if ny > 1:
        Ay = sparse.eye(p, p, nz, format='csr') -\
             sparse.eye(p, p)
        yind = ind[:, -1, :].ravel()
        for i in yind:
            Ay.data[Ay.indptr[i]: \
                    Ay.indptr[i + 1]] = 0
        Ay.eliminate_zeros()
    else:
        Ay = sparse.csc_matrix((p, p), dtype=float)
    if nz > 1:
        Az = sparse.eye(p, p, 1, format='csr') -\
             sparse.eye(p, p)
        zind = ind[:, :, -1].ravel()
        for i in zind:
            Az.data[Az.indptr[i]: \
                    Az.indptr[i + 1]] = 0
        Az.eliminate_zeros()
    else:
        Az = sparse.csc_matrix((p, p), dtype=float)
    return [Ax, Ay, Az], (nx * ny * nz - 1)