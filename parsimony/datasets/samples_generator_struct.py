# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 10:35:12 2013

@author:  Edouard Duchesnay
@email:   edouard.duchesnay@cea.fr
@license: TBD.
"""
# TODO: Remove dependence on scikit learn.

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt


def corr_to_coef(v_x, v_e, cov_xe, cor):
    """In a linear model y = bx + e. Calculate b such cor(bx + e, x) = cor.
    Parameters
    ----------
    v_x: float
        var(x)

    v_e: float
        var(e)

    cov_xe: float
        cov(x, e)

    cor: float
        The desire correlation

    Example
    -------
    b = corr_to_coef(1, 1, 0, .5)
    """
    b2 = v_x ** 2 * (cor ** 2 - 1)
    b1 = 2 * cov_xe * v_x * (cor ** 2 - 1)
    b0 = cor ** 2 * v_x * v_e - cov_xe ** 2
    delta = b1 ** 2 - 4 * b2 * b0
    sol1 = (-b1 - np.sqrt(delta)) / (2 * b2)
    sol2 = (-b1 + np.sqrt(delta)) / (2 * b2)
    return np.max([sol1, sol2]) if cor >= 0 else  np.min([sol1, sol2])


############################################################################
## utils


############################################################################
## Objects classes

class ObjImage(object):
    """
    Parameters:
    -----------
    c_x, c_y: Center of object

    shape: (nx, ny, nz)
        Image x, y and z dimensions.

    coef_info: float
        coefficien of information

    coef_noise: float
        coefficien of noise

    beta_z: array of float [1 x n_regressors]

    beta_o: float
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
            coef_noize = o.get_coef_noise()
            Xim[:, mask_o] = (coef_noize * obj_latent + Xim[:, mask_o].T).T
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
## Objects builder
def dice_five_with_union_of_pairs(shape, coef_info, coef_noise):
    """Seven objects, five dot + union1 = dots 1 + 2 and union2 = dots 4 + 5
    1, 3 and 4 have coef_info
    2, 5  have coef_info = -coef_info/2
    union1 and union2 have coef_info = 0

    Example
    -------
    #shape = (5, 5, 1)
    coef_info = 1
    coef_noise = 1
    noise = np.zeros(shape)
    info = np.zeros(shape)
    for o in dice_five_with_union_of_pairs(shape, coef_info, coef_noise):
        noise[o.get_mask()] += o.coef_noise
        info[o.get_mask()] += o.coef_info
        print o.coef_noise, o.coef_info
    plot = plt.subplot(121)
    import matplotlib.pyplot as plt
    cax = plot.matshow(noise.squeeze())
    plt.colorbar(cax)
    plt.title("Noise sum coeficients")
    plot = plt.subplot(122)
    cax = plot.matshow(info.squeeze())
    plt.colorbar(cax)
    plt.title("Informative sum coeficients")
    plt.show()
    """
    nx, ny, nz = shape
    if nx < 5 or ny < 5:
        raise ValueError("Shape too small minimun is (5, 5, 0)")
    s_obj = np.max([1, np.floor(np.max(shape) / 7)])
    k = 1
    c1 = np.floor((k * nx / 4., ny / 4., nz / 2.))
    d1 = Dot(center=c1, size=s_obj, shape=shape, coef_info=coef_info,
             coef_noise=coef_noise)
    c2 = np.floor((k * nx / 4., ny - (ny / 4.), nz / 2.))
    d2 = Dot(center=c2, size=s_obj, shape=shape, coef_info=-coef_info,
             coef_noise=0)
    union1 = ObjImage(mask=d1.get_mask() + d2.get_mask(), coef_info=0,
              coef_noise=coef_noise)
    k = 3
    c4 = np.floor((k * nx / 4., ny / 4., nz / 2.))
    d4 = Dot(center=c4, size=s_obj, shape=shape, coef_info=coef_info,
             coef_noise=coef_noise)
    c5 = np.floor((k * nx / 4., ny - (ny / 4.), nz / 2.))
    d5 = Dot(center=c5, size=s_obj, shape=shape, coef_info=-coef_info,
             coef_noise=0)
    union2 = ObjImage(mask=d4.get_mask() + d5.get_mask(), coef_info=0,
             coef_noise=coef_noise)
    ## dot in the middle
    c3 = np.floor((nx / 2., ny / 2., nz / 2.))
    d3 = Dot(center=c3, size=s_obj, shape=shape, coef_info=coef_info,
             coef_noise=coef_noise * 2)
    return [d1, d2, union1, d4, d5, union2, d3]


############################################################################
## Spatial smoothing
def spatial_smoothing(Xim, sigma, mu_e=None, sigma_e=None):
    for i in xrange(Xim.shape[0]):
        Xim[i, :] = ndimage.gaussian_filter(Xim[i, :],
            sigma=sigma)
    X = Xim.reshape((Xim.shape[0], np.prod(Xim.shape[1:])))
    # Spatial smoothing reduced the std-dev, reset it to 1
    if mu_e is not None:
        X -= X.mean(axis=0) + mu_e  # Also ensure null mean
    if sigma_e is not None:
        X /= X.std(axis=0) * sigma_e
    return Xim


############################################################################
def make_regression_struct(n_samples=100, shape=(30, 30, 1),
                           r2=.75,
                           sigma_spatial_smoothing=1,
                           noize_object_pixel_ratio=.5,
                           objects=None,
                           random_seed=None):
    """Generate regression samples (images + target variable) and beta.
    Input features (X) have a covariance structure controled both at a pixel
    level (spatial smoothing) and at an object. Objects are component
    of pixels sharing a covariance that stem from a latent variable.


    Noise is sampled according to N(0, 1).
    Then y is obtained with y = X * beta + noise, where beta is scalled such
    that r_square(y, X * beta) = r2.

    The sructure of covariance of X is

    Parameters
    ----------
    n_samples: int
        nb of samples, (default 100).

    shape: (int, int, int)
        x, y, z shape each samples (default (30, 30, 1)).

    r2: float
        The desire R-squared (explained variance) ie.:

            r_square(y, X * beta) = r2

        Default is .75

    sigma_spatial_smoothing: scalar
        Standard deviation for Gaussian kernel (default 1). High value promotes
        spatial correlation pixels.

    noize_object_pixel_ratio: float
        Controls the ratio between object-level noize and pixel-level noize for
        pixels within objects. If noize_object_pixel_ratio == 1 then 100% of
        the noize of pixels within the same object is shared (ie.: no pixel
        level) noize. If noize_object_pixel_ratio == 0 then all the noize is
        pixel specific. High noize_object_pixel_ratio promotes spatial
        correlation between pixels of the same object.

    objects: list of objects
        Define objects .
        Objects carying information to be drawn in the image. If not provide
        a dice with five points (object) will be drawn. Point 1, 3, 4 are
        carying predictive information while point 2 is a suppressor of point
        1 and point 5 is a suppressor of point 3.
        Object should implement "get_mask()" method, a have "is_suppressor"
        (bool) and "r" (ref to suppressor object, possibely None)
        attributes.

    random_seed: None or int
        See numpy.random.seed(). If not None, it can be used to obtain
        reproducable samples.

    Return
    ------
    X3d: array of shape [n_sample, shape]
        The input features.

    y: array of shape [n_sample, 1]
        The target variable.

    beta3d: float array of shape [shape]
       It is the beta such that y = X * beta + noize

    Details
    -------
    The general procedure is:
        1) For each pixel i, Generate independant variables Xi ~ N(0, 1)
        2) Add object level structure. By default there are five dots
        (objects). Pixel i of dot 1, 2, 3, 4, 5 are sampled as:
        X1i = l1 + l12 + Xi
        X2i = l12 + Xi
        X3i = 2 * l3 + Ni
        X4i = l4 + l45 + Xi
        X5i = l45 + Xi
        Where l1, l12, l3, l4, l45 are latent variables ~ N(0, 1)
        So pixels of X1 share a common variance that stem from l1 + l12.
        So pixels of X2 share a common variance that stem from l12.
        So pixels of X1 and X2 share a common variance that stem from l12.
        etc.
        4) Spatial Smoothing.
        5) Model: y = X beta + e
        - Betas are null outside dots, and 1 or -1 depending on the dot:
            X1: 1, X2: -1, X3: 1, X4: 1, X5: -1.
        - Sample noise e ~ N(0, 1)
        - Compute X beta then scale beta such that: r_squared(y, X beta) = r2
        Return X, y, beta

        Note that we get:
            y = X1 - X2 + X3 + X4 -X5 + e
            y = l1 + l3 + l4 + noise
            So pixels of X2 and X5 are not correlated with the target y so they
            will not be detected by univariate analysis. However, they
            are usefull since they are suppressing unwilling variance that stem
            from latents l12 and l45.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from parsimony.datasets import make_regression_struct
    >>> n_samples = 100
    >>> shape = (11, 11, 1)
    >>> r2 = .5
    >>> X3d, y, beta3d = make_regression_struct(n_samples=n_samples, shape=shape,
    ...                            r2=r2, random_seed=1)
    >>> X = X3d.reshape(n_samples, np.prod(shape))
    >>> beta = beta3d.ravel()
    >>> from sklearn.metrics import r2_score
    >>> print np.round(r2_score(y, np.dot(X, beta)), 2)
    0.47
    >>> cax = plt.matshow(beta3d.squeeze())
    >>> plt.colorbar(cax)
    >>> plt.title("Beta")
    >>> plt.show()
    """
    sigma_e = 1  # items std-dev
    mu_e = 0
    if shape[0] < 5 or shape[1] < 5:
        raise ValueError("Shape too small. The minimun is (5, 5, 0)")

    if len(shape) == 2:
        shape = tuple(list(shape) + [1])

    n_features = np.prod(shape)
    nx, ny, nz = shape

    ##########################################################################
    ## 1. Build images with noize => e_ij
    # Sample Noize: N
    if random_seed is not None:  # If random seed, save current random state
        rnd_state = np.random.get_state()
        np.random.seed(random_seed)
    Noise = np.random.normal(mu_e, sigma_e, n_samples * n_features)
    Noise3d = Noise.reshape(n_samples, nx, ny, nz)
    #########################################################################
    ## 2. Build Objects
    if objects is None:
        objects = dice_five_with_union_of_pairs(shape, coef_info=1.,
                                                coef_noise=sigma_e)
    #########################################################################
    ## 3. Object-level structured noise N
    Noise3d, support = ObjImage.object_model(objects, Noise3d)
    #########################################################################
    ## 4. Pixel-level noize structure: spatial smoothing
    if sigma_spatial_smoothing != 0:
        Noise3d = spatial_smoothing(Noise3d, sigma_spatial_smoothing, mu_e,
                                  sigma_e)

    Noise = Noise3d.reshape((Noise3d.shape[0], np.prod(Noise3d.shape[1:])))
    Noise -= Noise.mean(axis=0)
    Noise /= Noise.std(axis=0)

    #########################################################################
    ## 5. Model: y = X beta + noise
    X3d = Noise3d
    beta3d = np.zeros(X3d.shape[1:])

    for k in xrange(len(objects)):
        o = objects[k]
        beta3d[o.get_mask()] += o.coef_info
    beta3d = ndimage.gaussian_filter(beta3d, sigma=sigma_spatial_smoothing)
    beta = beta3d.ravel()
    # Fix a scaling to get the desire r2, ie.:
    # y = coef * X * beta + noize
    # Fix coef such r2(y, coef * X * beta) = r2
    X = X3d.reshape(n_samples, np.prod(shape))
    Xbeta = np.dot(X, beta)

    if r2 < 1:
        noise = np.random.normal(0, 1, Xbeta.shape[0])
        coef = corr_to_coef(v_x=np.var(Xbeta), v_e=np.var(noise),
                     cov_xe=np.cov(Xbeta, noise)[0, 1], cor=np.sqrt(r2))
        beta *= coef
        y = np.dot(X, beta) + noise
    else:
        noise = np.zeros(Xbeta.shape[0])
        y = np.dot(X, beta)

    if False:
        X = X3d.reshape((n_samples, nx * ny))
        Xc = (X - X.mean(axis=0)) / X.std(axis=0)
        yc = (y - y.mean()) / y.std()
        cor = np.dot(Xc.T, yc).reshape(nx, ny) / y.shape[0]
        cax = plt.matshow(cor, cmap=plt.cm.coolwarm)
        plt.colorbar(cax)
        plt.show()

    if random_seed is not None:   # If random seed, restore random state
        np.random.set_state(rnd_state)

    return X3d, y.reshape((n_samples, 1)), beta3d
