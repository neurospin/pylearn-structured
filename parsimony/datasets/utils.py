# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 10:35:12 2013

@author:  Edouard Duchesnay
@email:   edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""

import numpy as np
from scipy import ndimage
#import matplotlib.pyplot as plt


def corr_to_coef(v_x, v_e, cov_xe, cor):
    """In a linear model y = bx + e. Calculate b such cor(bx + e, x) = cor.

    Parameters
    ----------
    v_x: Float. The variance of x, var(x).

    v_e: Float. The variance of e, var(e).

    cov_xe: Float. The covariance between x and e, cov(x, e).

    cor: Float. The desired correlation.

    Examples
    --------
    >>> corr_to_coef(1, 1, 0, .5)
    0.57735026918962573
    """
    b2 = v_x ** 2 * (cor ** 2 - 1)
    b1 = 2 * cov_xe * v_x * (cor ** 2 - 1)
    b0 = cor ** 2 * v_x * v_e - cov_xe ** 2
    delta = b1 ** 2 - 4 * b2 * b0
    sol1 = (-b1 - np.sqrt(delta)) / (2 * b2)
    sol2 = (-b1 + np.sqrt(delta)) / (2 * b2)

    return np.max([sol1, sol2]) if cor >= 0 else np.min([sol1, sol2])


############################################################################
## utils


############################################################################
## Objects classes

class ObjImage(object):
    """
    Parameters:
    -----------
    mask: ???

    coef_info: Float. The coefficient of information.

    coef_noise: Float. Coefficient of noise.
    """
    def __init__(self, mask=None, coef_info=.5, coef_noise=.5):
        self.mask = mask
        self.coef_info = coef_info
        self.coef_noise = coef_noise

    def get_coef_info(self):
        return self.coef_info

    def get_coef_noise(self):
        return self.coef_noise

    def get_mask(self):
        return self.mask

    @staticmethod
    def object_model(objects, Xim):
        """Add object variance: x_ki =  coef^1/2 * o_k + (1 - coef)^1/2 * e_i
        """
        sigma_o = 1
        # Image of objects label
        labels_im = np.zeros(Xim.shape[1:], dtype=int)
        label = 0
        for k in xrange(len(objects)):
            o = objects[k]
            label += 1
            o.label = label
            # A) Add object latent variable
            mask_o = o.get_mask()
            labels_im[mask_o] = o.label
            obj_latent = np.random.normal(0, sigma_o, Xim.shape[0])
            obj_latent -= obj_latent.mean()  # - 0
            obj_latent /= obj_latent.std() * sigma_o
            coef_noise = o.get_coef_noise()
            Xim[:, mask_o] = (coef_noise * obj_latent + Xim[:, mask_o].T).T
        return Xim, labels_im


class Square(ObjImage):
    def __init__(self, center, size, shape, **kwargs):
        super(Square, self).__init__(**kwargs)
        self.size = size
        self.center = center
        self.x_grid, self.y_grid, self.z_grid = np.ogrid[0:shape[0],
                                                         0:shape[1],
                                                         0:shape[2]]

    def get_mask(self):
        hs = self.size / 2.
        mask = (np.abs(self.x_grid - self.center[0]) <= hs) & \
        (np.abs(self.y_grid - self.center[1]) <= hs)
        (np.abs(self.z_grid - self.center[2]) <= hs)
        return mask


class Dot(ObjImage):
    def __init__(self, center, size, shape, **kwargs):
        super(Dot, self).__init__(**kwargs)
        self.size = size
        self.center = center
        self.x_grid, self.y_grid, self.z_grid = np.ogrid[0:shape[0],
                                                         0:shape[1],
                                                         0:shape[2]]

    def get_mask(self):
        mask = np.sqrt((self.x_grid - self.center[0]) ** 2 + \
                       (self.y_grid - self.center[1]) ** 2 + \
                       (self.z_grid - self.center[2]) ** 2) <= self.size / 2
        return mask


class Dimaond(ObjImage):
    def __init__(self, center, size, shape, **kwargs):
        super(Square, self).__init__(**kwargs)
        self.size = size
        self.center = center
        self.x_grid, self.y_grid, self.z_grid = np.ogrid[0:shape[0],
                                                         0:shape[1],
                                                         0:shape[2]]

    def get_mask(self):
        mask = np.abs(self.x_grid - self.center[0]) + \
               np.abs(self.y_grid - self.center[1]) + \
               np.abs(self.z_grid - self.center[2]) <= self.size / 2
        return mask


def get_objects_edges(objects):
    m = objects[0].get_mask()
    m[::] = False
    for o in objects:
            m += o.get_mask()
    md = ndimage.binary_dilation(m)
    return md - m




############################################################################
## Spatial smoothing
def spatial_smoothing(Xim, sigma, mu_e=None, sigma_pix=None):
    for i in xrange(Xim.shape[0]):
        Xim[i, :] = ndimage.gaussian_filter(Xim[i, :],
            sigma=sigma)
    X = Xim.reshape((Xim.shape[0], np.prod(Xim.shape[1:])))
    # Spatial smoothing reduced the std-dev, reset it to 1
    if mu_e is not None:
        X -= X.mean(axis=0) + mu_e  # Also ensure null mean
    if sigma_pix is not None:
        X /= X.std(axis=0) * sigma_pix
    return Xim