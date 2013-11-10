# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 10:35:12 2013

@author: edouard.duchesnay@cea.fr
"""

import numpy as np
import scipy
import scipy.linalg
from scipy import ndimage
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.sparse.linalg


def corr_to_coef(v_x, v_e, cov_xe, R):
    """In a linear model y = bx + e. Calculate b such corr(bx + e, x) = R.
    Parameters
    ----------
    v_x: float
        var(x)

    v_e: float
        var(e)

    cov_xe: float
        cov(x, e)

    R: float
        The desire correlation

    Example
    -------
    b = corr_to_coef(1, 1, 0, .5)
    """
    b2 = v_x ** 2 * (R ** 2 - 1)
    b1 = 2 * cov_xe * v_x * (R ** 2 - 1)
    b0 = R ** 2 * v_x * v_e - cov_xe ** 2
    delta = b1 ** 2 - 4 * b2 * b0
    sol1 = (-b1 - np.sqrt(delta)) / (2 * b2)
    sol2 = (-b1 + np.sqrt(delta)) / (2 * b2)
    return np.max([sol1, sol2]) if R >= 0 else  np.min([sol1, sol2])


############################################################################
## utils


############################################################################
## Objects classes

class ObjImage(object):
    """
    Parameters:
    -----------
    c_x, c_y: Center of object

    im_x, im_y: int
        Image x and y dimensions.

    beta_y: float

    beta_z: array of float [1 x n_regressors]

    beta_o: float
    """
    def __init__(self, c_x, c_y, c_z, nx, ny, nz):
        self.c_x = c_x
        self.c_y = c_y
        self.c_z = c_z
        self.x_grid, self.y_grid , self.z_grid = np.ogrid[0:nx, 0:ny, 0:nz]
        self.is_suppressor = False
        self.suppressor = None

    def set_suppresor(self, suppressor):
        suppressor.is_suppressor = True
        self.suppressor = suppressor

    def get_suppresor(self):
        return self.suppressor


class Square(ObjImage):
    def __init__(self, size, coef=.5, **kwargs):
        super(Square, self).__init__(**kwargs)
        self.size = size
        self.coef = coef

    def get_mask(self):
        hs = self.size / 2
        mask = (np.abs(self.x_grid - self.c_x) <= hs) & \
        (np.abs(self.y_grid - self.c_y) <= hs)
        (np.abs(self.z_grid - self.c_z) <= hs)
        return mask


class Dot(ObjImage):
    def __init__(self, size, coef=.5, **kwargs):
        super(Dot, self).__init__(**kwargs)
        self.size = size
        self.coef = coef

    def get_mask(self):
        mask = np.sqrt((self.x_grid - self.c_x) ** 2 + \
                       (self.y_grid - self.c_y) ** 2 + \
                       (self.z_grid - self.c_z) ** 2) <= self.size / 2
        return mask


class Dimaond(ObjImage):
    def __init__(self, size, coef=.5, **kwargs):
        super(Square, self).__init__(**kwargs)
        self.size = size
        self.coef = coef

    def get_mask(self):
        mask = np.abs(self.x_grid - self.c_x) + \
               np.abs(self.y_grid - self.c_y) + \
               np.abs(self.z_grid - self.c_z) <= self.size / 2
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
def dice_five(shape, noize_object_pixel_ratio):
    nx, ny, nz = shape
    s_obj = np.floor(nx / 7)
    objects = list()
    ## corner dots
    for k in [1, 3]:
        c_x = k * nx / 4.
        c_y = ny / 4.
        c_z = nz / 2.
        o_info = Dot(size=s_obj, coef=noize_object_pixel_ratio,
                     c_x=c_x, c_y=c_y, c_z=c_z, 
                     nx=nx, ny=ny, nz=nz)
        objects.append(o_info)
        c_y = ny - (ny / 4.)
        o_supp = Dot(size=s_obj, coef=noize_object_pixel_ratio,
                     c_x=c_x, c_y=c_y, c_z=c_z, nx=nx, ny=ny, nz=nz)
        objects.append(o_supp)
        o_info.set_suppresor(o_supp)
    ## dot in the middle
    c_x = nx / 2.
    c_y = ny / 2.
    c_z = nz / 2.
    o_info = Dot(size=s_obj, coef=noize_object_pixel_ratio,
        c_x=c_x, c_y=c_y, c_z=c_z, nx=nx, ny=ny, nz=nz)
    objects.append(o_info)
    return objects


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
## Add objects-based variance
def object_model(objects, Xim):
    """Add object variance: x_ki =  coef^1/2 * o_k + (1 - coef)^1/2 * e_i
    """
    sigma_o = 1
    labels_im = np.zeros(Xim.shape[1:], dtype=int)  # image of objects label
    label = 0
    for k in xrange(len(objects)):
        o = objects[k]
        #print o.is_suppressor, o.suppressor
        if o.is_suppressor:
            continue
        label += 1
        o.label = label
        # A) Add object latent variable
        mask_o = o.get_mask()
        labels_im[mask_o] = o.label
        o_ik = np.random.normal(0, sigma_o, Xim.shape[0])
        o_ik -= o_ik.mean()  # - 0
        o_ik /= o_ik.std() * sigma_o
        Xim[:, mask_o] = (np.sqrt(o.coef) * o_ik + \
                        np.sqrt(1 - o.coef) * Xim[:, mask_o].T).T
        if o.suppressor is not None:
            o.suppressor.label = -label
            mask_o_suppr = o.suppressor.get_mask()
            labels_im[mask_o_suppr] = o.suppressor.label
            Xim[:, mask_o_suppr] = (np.sqrt(o.suppressor.coef) * o_ik + \
                       np.sqrt(1 - o.suppressor.coef) * Xim[:, mask_o_suppr].T).T
    return Xim, labels_im


############################################################################
## Apply causal model on objects
def generative_model(objects, Xim, y, R):
    """Add predictive information: x_ki += b_y * y. Returns Xim.
    """
    beta_y = corr_to_coef(v_x=1, v_e=1, cov_xe=0, R=R)
    for k in xrange(len(objects)):
        o = objects[k]
        if o.is_suppressor:
            continue
        # A) Add object latent variable
        mask_o = o.get_mask()
        # Compute the coeficient according to a desire correlation
        Xim[:, mask_o] = (Xim[:, mask_o].T + (beta_y * y)).T
    return Xim

############################################################################
## Apply causal model on objects
def predictive_model(objects, Xim, snr, coef_per_feature):
    """Add predictive information: y = X * beta_objects + noize_snr.
    Retrns y, beta_flat, noize.
    """
    beta = np.zeros(Xim.shape[1:])
    for k in xrange(len(objects)):
        o = objects[k]
        if o.is_suppressor:
            continue
        beta[o.get_mask()] = coef_per_feature
    X = Xim.reshape((Xim.shape[0], np.prod(Xim.shape[1:])))
    beta_flat = beta.ravel()
    y_true = np.dot(X, beta_flat)
    noize = np.random.normal(0, y_true.std() / snr, y_true.shape[0])
    y = y_true + noize
    return y, beta_flat, noize


############################################################################
## Parameters
def make_regression_struct(n_samples=100, shape=(30, 30, 1),
                           mode="predictive",
                           snr=.2, R=.5,
                           sigma_spatial_smoothing=1,
                           noize_object_pixel_ratio=.5,
                           objects=None,
                           random_seed=None):
    """Generate regression samples (images + target variable) with input
    features having a covariance structure for both noize and informative
    features. The sructure of covariance can be controled both at a pixel level
    (spatial smoothing) and at an object level. Objects are connected component
    of pixels sharing a covariance that stem from a latent variable.

    Parameters
    ----------
    n_samples: int
        nb of samples, (default 100).

    shape: (int, int, int)
        x, y, z shape each samples (default (30, 30, 1)).

    mode: string in "predictive" (default) or "generative"

        - In predictive mode, after having generated X, y is obtained by
        y = X * beta + noize.
        In this mode the function return the beta.

        - In generative mode, y is randomly sampled, then it is added to causal
        pixels (i) within object (k):
        x_ik = e_ik + y.
        In this setting beta is unknown.

    snr: float
        Use in mode == "predictive" is the ratio: std(Xbeta) / std(noize)

    R: float
        Use in mode == "generative" is the desire correlation between causal
        pixels and the target y.

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
       If mode == "predictive", it is the beta such that y = X * beta + noize
       If mode == "generative", beta is None.

    support: int array of shape [shape].
        Support of objects (image of labels).
        However the support of causal pixels are returned in the label image.
        Positive values of labels are informative pixels ie.: pixels with non
        zero beta in predictive mode and pixels that contain "y" in generative
        mode (where x_ik = e_ijk + y). Negative values are the support of the
        suppressor objects of the corresponding positive value. Such objects
        share some noize variance with another predictive object and thus can
        be usefull to suppress unwilling noize.

    Detail
    ------
    The general procedure is (1) Generate independant noize (e_i); (2) Add
    object level noize. (3) If mode == "generative", generate y and add it
    to causal pixels. (4) Spatial Smoothing. (5)  If mode == "predictive",
    compute y = X * beta

    The signal within each pixel i and object k is a linear combination of
    some information and object-level and pixel-level noize. Let
    b_o = noize_object_pixel_ratio, then:
    
    (1 and 2) Generate independant noize and object level noize
    x_ki =  b_o^1/2 * o_k + (1 - b_o)^1/2 * e_i
           <------------- noize ------------->
           <- object k ->   <--- pixel i ---->

    e_i ~ N(0, 1) is the pixel-level noize, for all pixels i in [1, n_features]
    o_k ~ N(0, 1) is the object-level noize, for all objects k in [1, n_objects]

    (3) If mode == "generative":
    Generate y (~ N(0, 1)) and add it to causal pixels. Causal pixels are
    pixels within the objects.
    x_ki = x_ki +  b_y * y + x_ki; for pixels within causal objects k
                   <info > + <noize>
    Here beta is not available, only the support of causal pixels is known
    ie.: the positive values in the returned beta.

    (4) Spatial Smoothing

    (5) If mode == "predictive", generate beta Causal pixels (pixels within
    objects) and compute y:
    y = X * beta + noize
    """
    sigma_y = sigma_e = 1  # items std-dev
    mu_e = mu_y = 0
    if len(shape) == 2:
        shape = tuple(list(shape) + [1])
    n_features = np.prod(shape)
    nx, ny, nz = shape

    ##########################################################################
    ## 1. Build images with noize => e_ij
    # Sample according to means and Cov
    if random_seed is not None:  # If random seed, save current random state
        rnd_state = np.random.get_state()
        np.random.seed(random_seed)

    X = np.random.normal(mu_e, sigma_e, n_samples * n_features).reshape(n_samples,
                                                                     n_features)
    Xim = X.reshape(n_samples, nx, ny, nz)

    #########################################################################
    ## 2. Build Objects
    if objects is None:
        objects = dice_five(shape, noize_object_pixel_ratio)

    #########################################################################
    ## 3. Object-level noize structure
    Xim, support = object_model(objects, Xim)
    #X = Xim.reshape((Xim.shape[0], Xim.shape[1]*Xim.shape[2]))
    #print X.mean(axis=0), X.std(axis=0)

    #########################################################################
    ## 4. Causal generative model
    if mode == "generative":
        y = np.random.normal(mu_y, sigma_y, n_samples)
        y -= y.mean()
        y /= y.std()
        Xim = generative_model(objects, Xim, y, R)
        beta = None

    #########################################################################
    ## 5. Pixel-level noize structure: spatial smoothing
    if sigma_spatial_smoothing != 0:
        Xim = spatial_smoothing(Xim, sigma_spatial_smoothing, mu_e, sigma_e)
        #X = Xim.reshape((Xim.shape[0], Xim.shape[1] * Xim.shape[2]))

    if mode == "predictive":
        y, beta_flat, noize = \
            predictive_model(objects, Xim, snr, coef_per_feature=1.)
        beta = beta_flat.reshape(nx, ny, nz)
        if False:
            X = Xim.reshape((n_samples, nx * ny))
            Xc = (X - X.mean(axis=0)) / X.std(axis=0)
            yc = (y - y.mean()) / y.std()
            cor = np.dot(Xc.T, yc).reshape(nx, ny) / y.shape[0]
            cax = plt.matshow(cor, cmap=plt.cm.coolwarm)
            plt.colorbar(cax)
            plt.show()

    if random_seed is not None:   # If random seed, restore random state
        np.random.set_state(rnd_state)

    return Xim, y.reshape((n_samples, 1)), beta, support


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
    R = .25
    snr = 2.
    sigma_spatial_smoothing = 1
    random_seed=None
    noize_object_pixel_ratio=.25
    objects = None
    mode = "predictive"
    #mode = "generative"
    Xim, y, beta, support = make_regression_struct(n_samples=n_samples,
        shape=shape, snr=1., sigma_spatial_smoothing=sigma_spatial_smoothing,
        mode=mode, noize_object_pixel_ratio=noize_object_pixel_ratio,
        random_seed=random_seed)
    _, nx, ny, nz = Xim.shape

    X = Xim.reshape((n_samples, nx * ny))
    from sklearn.metrics import r2_score
    n_train = min(100, int(X.shape[1] / 10))
    Xtr = X[:n_train, :]
    ytr = y[:n_train]
    Xte = X[n_train:, :]
    yte = y[n_train:]

    plt.figure()
    plot = plt.subplot(331)
    plot_map(np.sign(support.squeeze()), plot)
    plt.title("Objects support (blue are suppressors)")

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
    #plt.show()
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
