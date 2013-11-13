# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 10:35:12 2013

@author: edouard.duchesnay@cea.fr
"""

import abc
import numpy as np
import scipy
import scipy.linalg
from scipy import ndimage
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg


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
#        self.brother = None  # Object can be linked with a "brother" object
#        self.is_brother = False  

#    def set_brother(self, brother):
#        self.brother = brother
#        brother.is_brother = self
#        return False if self.brother is None else True
#
#    def has_brother(self):
#        return False if self.brother is None else True
#
#    def is_brother(self):
#        return self.is_brother
#
#    def get_brother(self):
#        return self.brother
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
        labels_im = np.zeros(Xim.shape[1:], dtype=int)  # image of objects label
        label = 0
        for k in xrange(len(objects)):
            o = objects[k]
            #print o.is_suppressor, o.suppressor
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
#            if o.has_brother():
#                brother = o.get_brother()
#                brother.label = -label
#                mask_brother = brother.get_mask()
#                coef_noize = o.suppressor.get_coef_noise()
#                labels_im[mask_brother] = brother.label
#                Xim[:, mask_brother] = (np.sqrt(coef_noize) * o_ik + \
#                           np.sqrt(1 - coef_noize) * Xim[:, mask_brother].T).T
        return Xim, labels_im


class Square(ObjImage):
    def __init__(self, center, size, shape, **kwargs):
        super(Square, self).__init__(**kwargs)
        self.size = size
        self.center = center
        self.x_grid, self.y_grid , self.z_grid = np.ogrid[0:shape[0], 0:shape[1],
                                                          0:shape[2]]
    def get_mask(self):
        hs = self.size / 2
        mask = (np.abs(self.x_grid - self.center[0]) <= hs) & \
        (np.abs(self.y_grid - self.center[1]) <= hs)
        (np.abs(self.z_grid - self.center[2]) <= hs)
        return mask


class Dot(ObjImage):
    def __init__(self, center, size, shape, **kwargs):
        super(Dot, self).__init__(**kwargs)
        self.size = size
        self.center = center
        self.x_grid, self.y_grid , self.z_grid = np.ogrid[0:shape[0], 0:shape[1],
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
        self.x_grid, self.y_grid , self.z_grid = np.ogrid[0:shape[0], 0:shape[1],
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
    shape = (70, 70, 1)
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
    s_obj = np.floor(nx / 7)
    k = 1
    c1 = (k * nx / 4., ny / 4., nz / 2.)
    d1 = Dot(center=c1, size=s_obj, shape=shape, coef_info=coef_info,
             coef_noise=coef_noise / 2.)
    c2 = (k * nx / 4., ny - (ny / 4.), nz / 2.)
    d2 = Dot(center=c2, size=s_obj, shape=shape, coef_info=-coef_info / 2,
             coef_noise=coef_noise / 2.)
    union1 = ObjImage(mask=d1.get_mask() + d2.get_mask(), coef_info=0,
              coef_noise=coef_noise / 2.)
    k = 3
    c4 = (k * nx / 4., ny / 4., nz / 2.)
    d4 = Dot(center=c4, size=s_obj, shape=shape, coef_info=coef_info,
             coef_noise=coef_noise / 2.)
    c5 = (k * nx / 4., ny - (ny / 4.), nz / 2.)
    d5 = Dot(center=c5, size=s_obj, shape=shape, coef_info=-coef_info / 2,
             coef_noise=coef_noise / 2.)
    union2 = ObjImage(mask=d4.get_mask() + d5.get_mask(), coef_info=0,
             coef_noise=coef_noise / 2.)
    ## dot in the middle
    c3 = (nx / 2., ny / 2., nz / 2.)
    d3 = Dot(center=c3, size=s_obj, shape=shape, coef_info=coef_info,
             coef_noise=coef_noise)
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
    Input features (X) have a covariance structure N(0, Sigma).
    Noise is sampled according to N(0, 1).
    Then y is obtained with y = X * beta + noise, where beta is scalled such that 
    r_square(y, X * beta) = r2.

    The sructure of covariance of X is controled both at a pixel level
    (spatial smoothing) and at an object level. Objects are connected component
    of pixels sharing a covariance that stem from a latent variable.

    Parameters
    ----------
    n_samples: int
        nb of samples, (default 100).

    shape: (int, int, int)
        x, y, z shape each samples (default (30, 30, 1)).

    r2: float
        The desire R-squared (explained variance) ie.: r_square(y, X * beta) = r2
        Default is .75

    sigma_spatial_smoothing: scalar
        Standard deviation for Gaussian kernel (default 1). High value promotes
        spatial correlation pixels.

    noize_object_pixel_ratio: float
        Controls the ratio between object-level noize and pixel-level noize for
        pixels within objects. If noize_object_pixel_ratio == 1 then 100% of the 
        noize of pixels within the same object is shared (ie.: no pixel level)
        noize. If noize_object_pixel_ratio == 0 then all the noize is pixel
        specific. High noize_object_pixel_ratio promotes spatial correlation
        between pixels of the same object.

    objects: list of objects
        Define connected components of causal (or suppressor) pixels. 
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
    X: array of shape [n_sample, shape]
        the input features.

    y: array of shape [n_sample, 1]
        the target variable.

    beta: float array of shape [shape]
       It is the beta such that y = X * beta + noize

    Examples
    --------
    >>> from parsimony.datasets import make_regression_struct
    >>> n_samples = 100
    >>> shape = (10, 10, 10)
    >>> r2 = .5
    >>> X, y, beta, support = make_regression_struct(n_samples=n_samples, shape=shape,
    ...                            r2=r2, random_seed=1)
    >>> X_flat = X.reshape(n_samples, np.prod(shape))
    >>> beta_flat = beta.ravel()
    >>> from sklearn.metrics import r2_score
    >>> print np.round(r2_score(y, np.dot(X_flat, beta_flat)), 2)
    0.5
    """
    sigma_e = 1  # items std-dev
    mu_e = 0
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
    Noise = np.random.normal(mu_e, sigma_e, n_samples * n_features)#.reshape(n_samples,
                                                                   #  n_features)
    Noise = Noise.reshape(n_samples, nx, ny, nz)
    #########################################################################
    ## 2. Build Objects
    if objects is None:
        objects = dice_five_with_union_of_pairs(shape, coef_info=1., coef_noise=sigma_e)
    #########################################################################
    ## 3. Object-level structured noise N
    Noise, support = ObjImage.object_model(objects, Noise)
    #########################################################################
    ## 4. Pixel-level noize structure: spatial smoothing
    if sigma_spatial_smoothing != 0:
        Noise = spatial_smoothing(Noise, sigma_spatial_smoothing, mu_e, sigma_e)
    Noise_flat = Noise.reshape((Noise.shape[0], np.prod(Noise.shape[1:])))
    Noise_flat -= Noise_flat.mean(axis=0)
    Noise_flat /= Noise_flat.std(axis=0)
    
    #########################################################################
    ## 4. Model: y = X beta + noise
    X = Noise
    beta = np.zeros(X.shape[1:])
    for k in xrange(len(objects)):
        o = objects[k]
        beta[o.get_mask()] += o.coef_info
    beta_flat = beta.ravel()
    # Fix a scaling to get the desire r2, ie.:
    # y = coef * X * beta + noize
    # Fix coef such r2(y, coef * X * beta) = r2
    X_flat = X.reshape(n_samples, np.prod(shape))
    Xbeta = np.dot(X_flat, beta_flat)
    if r2 < 1:
        noise = np.random.normal(0, 1, Xbeta.shape[0])
        coef = corr_to_coef(v_x=np.var(Xbeta), v_e=np.var(noise),
                     cov_xe=np.cov(Xbeta, noise)[0, 1], cor=np.sqrt(r2))
        beta_flat *= coef
        y = np.dot(X_flat, beta_flat) + noise
    else:
        noise = np.zeros(Xbeta.shape[0])
        y = np.dot(X_flat, beta_flat)
    if False:
        Xflat = X.reshape((n_samples, nx * ny))
        Xc = (Xflat - Xflat.mean(axis=0)) / Xflat.std(axis=0)
        yc = (y - y.mean()) / y.std()
        cor = np.dot(Xc.T, yc).reshape(nx, ny) / y.shape[0]
        cax = plt.matshow(cor, cmap=plt.cm.coolwarm)
        plt.colorbar(cax)
        plt.show()
    if random_seed is not None:   # If random seed, restore random state
        np.random.set_state(rnd_state)
    return X, y.reshape((n_samples, 1)), beta



if __name__ == '__main__':
    # utils
    def plot_map(im, plot=None):
        if plot is None:
            plot = plt
        #fig, ax = plt.subplots()
        cax = plot.matshow(im, cmap=plt.cm.coolwarm)
        frame = plt.gca()
        frame.get_xaxis().set_visible(False)
        frame.get_yaxis().set_visible(False)
        mx = np.abs(im).max()
        k = 1
        while (10 ** k * mx) < 1 and k<10: k+=1
        ticks = np.array([-mx, -mx/4 -mx/2, 0, mx/2, mx/2, mx]).round(k+2)
        cbar = plt.colorbar(cax, ticks=ticks)
        cbar.set_clim(vmin=-mx, vmax=mx)

    n_samples = 500
    shape = (100, 100, 1)
    #shape = (30, 30, 1)
    r2 = .75
    sigma_spatial_smoothing = 1
    random_seed=None
    noize_object_pixel_ratio=.25
    objects = None
    Xim, y, beta = make_regression_struct(n_samples=n_samples,
        shape=shape, r2=r2, sigma_spatial_smoothing=sigma_spatial_smoothing,
        noize_object_pixel_ratio=noize_object_pixel_ratio,
        random_seed=random_seed)
    _, nx, ny, nz = Xim.shape


    X = Xim.reshape((n_samples, nx * ny))
    from sklearn.metrics import r2_score
    n_train = min(100, int(X.shape[1] / 10))
    Xtr = X[:n_train, :]
    ytr = y[:n_train]
    Xte = X[n_train:, :]
    yte = y[n_train:]

#    plt.figure()
#    plot = plt.subplot(331)
#    plot_map(np.sign(support.squeeze()), plot)
#    plt.title("Objects support (blue are suppressors)")

    plot = plt.subplot(332)
    if beta is not None: 
        plot_map(beta.reshape(nx, ny), plot)
    plt.title("beta")

    Xtrc = (Xtr - Xtr.mean(axis=0)) / Xtr.std(axis=0)
    ytrc = (ytr - ytr.mean()) / ytr.std()
    cor = np.dot(Xtrc.T, ytrc).reshape(nx, ny) / ytr.shape[0]
    plot = plt.subplot(333)
    plot_map(cor, plot)
    plt.title("Corr(X, y)")
    #plt.show()
    
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import ElasticNet, ElasticNetCV
    from sklearn.linear_model import Ridge

    # Global penalization paapeter: alpha according to:
    # ||y - Xw||^2_2 + alpha penalization
    alpha_g = 10.
    # Ridge ================================================================
    # Min ||y - Xw||^2_2 + alpha ||w||^2_2
    plot = plt.subplot(334)
    alpha = alpha_g
    l2 = Ridge(alpha=alpha)
    pred = l2.fit(Xtr, ytr).predict(Xte)
    plot_map(l2.coef_.reshape((nx, ny)), plot)
    plt.title("L2 (R2=%.2f)" % r2_score(yte, pred))

    # Lasso  ================================================================
    # Min (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1
    alpha = alpha_g * 1. / (2. * n_train)
    l1 = Lasso(alpha=alpha)
    pred = l1.fit(Xtr, ytr).predict(Xte)
    plot = plt.subplot(335)
    plot_map(l1.coef_.reshape((nx, ny)), plot)
    plt.title("Lasso (R2=%.2f)" % r2_score(yte, pred))

    # Enet  ================================================================
    #    Min: 1 / (2 * n_samples) * ||y - Xw||^2_2 +
    #        + alpha * l1_ratio * ||w||_1
    #        + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2
    alpha = alpha_g * 1. / (2. * n_train)
    l1_ratio = .01
    l1l2 = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    pred = l1l2.fit(Xtr, ytr).predict(Xte)
    plot = plt.subplot(336)
    plot_map(l1l2.coef_.reshape((nx, ny)), plot)
    plt.title("Elasticnet(a:%.2f, l1:%.2f) (R2=%.2f)" % (l1l2.alpha, 
              l1l2.l1_ratio, r2_score(yte, pred)))

    l1l2cv = ElasticNetCV()
    pred = l1l2cv.fit(Xtr, ytr).predict(Xte)
    plot = plt.subplot(337)
    plot_map(l1l2cv.coef_.reshape((nx, ny)), plot)
    plt.title("ElasticnetCV(a:%.2f, l1:%.2f) (R2=%.2f)" % (l1l2cv.alpha_,
              l1l2cv.l1_ratio, r2_score(yte, pred)))
    #plt.show()

    # TVL1L2 ===============================================================
    import parsimony.estimators as estimators
    import parsimony.tv
    def ratio2coef(alpha, tv_ratio, l1_ratio):
        l = alpha * (1 - tv_ratio) * l1_ratio
        k = alpha * (1 - tv_ratio) * (1 - l1_ratio) * 0.5
        gamma = alpha * tv_ratio
        return l, k, gamma

    eps = 0.01
    alpha = alpha_g #Constant that multiplies the penalty terms

    tv_ratio=.05; l1_ratio=.95
    l, k, g = ratio2coef(alpha=alpha, tv_ratio=tv_ratio, l1_ratio=l1_ratio)

    Ax, Ay, Az, n_compacts = parsimony.tv.tv_As_from_shape(shape)

    tvl1l2 = estimators.LinearRegressionL1L2TV(k, l, g, [Ax, Ay, Az],
                                       algorithm="conesta_static")
    tvl1l2.fit(Xtr, ytr)
    #f = pgm.get_algorithm().f
    plot = plt.subplot(338)
    plot_map(tvl1l2.beta.reshape(nx, ny), plot)
    r2 = r2_score(yte, np.dot(Xte, tvl1l2.beta).ravel())
    plt.title("L1L2TV(a:%.2f, tv: %.2f, l1:%.2f) (R2=%.2f)" % \
        (alpha, tv_ratio, l1_ratio, r2))

    tv_ratio=.5; l1_ratio=.95
    l, k, g = ratio2coef(alpha=alpha, tv_ratio=tv_ratio, l1_ratio=l1_ratio)
    tvl1l2 = estimators.LinearRegressionL1L2TV(k, l, g, [Ax, Ay, Az],
                                       algorithm="conesta_static")
    tvl1l2.fit(Xtr, ytr)
    plot = plt.subplot(339)
    plot_map(tvl1l2.beta.reshape(nx, ny), plot)
    r2 = r2_score(yte, np.dot(Xte, tvl1l2.beta).ravel())
    plt.title("L1L2TV(a:%.2f, tv: %.2f, l1:%.2f) (R2=%.2f)" % \
        (alpha, tv_ratio, l1_ratio, r2))

    plt.show()
    #run pylearn-parsimony/parsimony/datasets/samples_generator_struct.py
