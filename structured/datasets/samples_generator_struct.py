# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 10:35:12 2013

@author: edouard.duchesnay@cea.fr

Generative model to sample input features (image) and the associated target
variable. Provide a model of the structure between the input features for
both noizy and informative features. The structure can be controled at pixel
level and at object (connected component of pixels) level.


Generative sampling model
-------------------------

The signal within each pixel i and object k is a linear combination of some
information and object-level and pixel-level noize.

x_ki =  b_y * y + b_o^1/2 * o_k + (1 - b_o)^1/2 * e_i
       < info >   <------------- noize ------------->
                  <- object k ->   <--- pixel i ---->

e_i ~ N(0, 1) is the pixel-level noize, for all pixels i in [1, n_features]
o_k ~ N(0, 1) is the object-level noize, for all objects k in [1, n_objects]
y   ~ N(0, 1) is the target variable

Procedure: (1) Generate e_i; (2) Spatial Smoothing; (3) Add object level noize;
(4) Add y on causal pixels.

Parameters
----------
b_o: float
    Controls the amount of shared noize within
    objects. It is null outside objects. Thus (1 - b_o) control the amount of
    pixel level noize. b_o enable the mixing between pixel and object level noize.

R: float
    Is the desire correlation between causal pixels and
    the target y. 

Details
-------

b_y is null outside objects. Within causal object, it is computed such that
corr(x_ki, y) = R

Moreovere the model assume that:
var(noize) = b_o var(o_k) + (1 - b_o) var(e_ki) = 1
cov(y, noize) = 0
 
Predictive model
----------------

Given X = [x_ki] for all pixels i in [1, n_features], the function return
beta = (X'X)^-1 X'y, for the OLS solution of the problem: min|y - X beta|_l2  

[X'X]_ij = n_sample * E(x_ki, x_lj)
E(x_ki, x_kj) = b_y^2 + b_o + (1 - b_o) * E(e_ki, e_lj)
                if i,j belong to the same causal object (k=l)
              =         b_o + (1 - b_o) * E(e_ki, e_lj)
                if i belongs to a causal object and j belongs to its suppressor
              = is not computed in all other cases.

E(e_ki, e_lj): spatial cov after spatial smoothing
              = Rij * sigma_e^2 + 2 mu_e
Rij: spatial correlation after spatial smoothing
     = Rij = exp(-dist_euc^2/(4 * sigma_spatial_smoothing^2) )


X'_i y = n_sample * E(x_ki, y)
E(x_ki, y) = b_y * E(y^2)


i: sample index
k: object index
j: pixel index

1. Build images with noize => e_ij
2. Build Objects
3. Apply causal model for each objects
   => add y_i beta_y + Zi Beta_z + o_ik beta_o
4. Smooth
"""

import numpy as np
import scipy
import scipy.linalg
from scipy import ndimage
import matplotlib.pyplot as plt

#def dist_euclidian(lx, ly):
#    """Euclidian distance
#    lx, ly: int
#        lx, ly the dimension of the 2D array
#    return
#        [[lx * ly] * [lx * ly]] matrix of euclidian diastances
#    """
#    x_grid, y_grid = np.ogrid[0:lx, 0:ly]
#    x_coord = x_grid + np.zeros(lx, dtype=int)
#    y_coord = y_grid + np.zeros(ly, dtype=int)[:, np.newaxis]
#    x_coord = x_coord.ravel()
#    y_coord = y_coord.ravel()
#    return np.sqrt((x_coord[np.newaxis].T - x_coord)**2 + \
#    (y_coord[np.newaxis].T - y_coord)**2)


def covmat_indices(cols):
    """Return indices in covariance matrix given indice in a matrix. The
    order respect the flattening of the matrix.
    Example
    -------
    >>> covmat_indices([1, 3])
    (array([1, 1, 3, 3]), array([1, 3, 1, 3]))
    >>> # order respect the flattening of the matrix
    >>> m = np.array([[1, 2], [3, 4]])
    >>> np.all(m[covmat_indices([0, 1])] == m.ravel())
    True
    """
    cols = np.asarray(cols)
    x_coord = cols[:, np.newaxis] + np.zeros(cols.shape[0], dtype=int)
    y_coord = cols + np.zeros(cols.shape[0], dtype=int)[:, np.newaxis]
    return x_coord.ravel(), y_coord.ravel()


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

def corr_to_coef_empirical(x, e, R):
    """In a linear model y = bx + e. Calculate b such corr(bx + e, x) = R.
    Parameters
    ----------
    x: array of shape [n]
    e: array of shape [n]
    R: float
        The desire correlation

    Example
    -------
    import numpy as np
    e = np.random.normal(1, 10, 100)
    x = np.random.normal(4, 3, 100)
    b = corr_to_coef_empirical(x, e, .5)
    np.corrcoef(b * x + e, x)[0, 1]
    """
    v_x = np.var(x)
    v_e = np.var(e)
    cov_xe = np.cov(x, e)[0, 1]
    return corr_to_coef(v_x, v_e, cov_xe, R)

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
    def __init__(self, c_x, c_y, im_x, im_y):
        self.c_x = c_x
        self.c_y = c_y
        self.x_grid, self.y_grid = np.ogrid[0:im_x, 0:im_y]
        self.is_suppressor = False
        self.suppressor = None
    def set_suppresor(self, suppressor):
        suppressor.is_suppressor = True
        self.suppressor = suppressor
    def get_suppresor(self):
        return self.suppressor

class Square(ObjImage):
    def __init__(self, size, **kwargs):
        super(Square, self).__init__(**kwargs)
        self.size = size
    def get_mask(self):
        hs = self.size / 2        
        mask = (np.abs(self.x_grid - self.c_x) <= hs) & \
        (np.abs(self.y_grid - self.c_y) <= hs)
        return mask

class Dot(ObjImage):
    def __init__(self, size, **kwargs):
        super(Dot, self).__init__(**kwargs)
        self.size = size
    def get_mask(self):
        mask = np.sqrt((self.x_grid - self.c_x) ** 2 + \
                       (self.y_grid - self.c_y) ** 2) <= self.size / 2
        return mask

class Dimaond(ObjImage):
    def __init__(self, size, **kwargs):
        super(Square, self).__init__(**kwargs)
        self.size = size
    def get_mask(self):
        mask = np.abs(self.x_grid - self.c_x) + \
        np.abs(self.y_grid - self.c_y) <= self.size / 2
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
def dice_five(lx, ly):
    s_obj = np.floor(lx / 7)
    objects = list()
    ## corner dots
    for k in [1, 3]:
        c_x = k * lx / 4.
        c_y = ly / 4.
        o_info = Dot(c_x=c_x, c_y=c_y, size=s_obj, im_x=lx, im_y=ly)
        objects.append(o_info)
        c_y = ly - (ly / 4.)
        o_supp = Dot(c_x=c_x, c_y=c_y, size=s_obj, im_x=lx, im_y=ly)
        objects.append(o_supp)
        o_info.set_suppresor(o_supp)
    ## dot in the middle
    c_x = lx / 2.
    c_y = ly / 2.
    o_info = Dot(c_x=c_x, c_y=c_y, size=s_obj, im_x=lx, im_y=ly)
    objects.append(o_info)
    return objects


############################################################################
## Add objects-based variance
def object_model(objects, Xim):
    """Add object variance: x_ki =  b_o^1/2 * o_k + (1 - b_o)^1/2 * e_i
    """
    for k in xrange(len(objects)):
        o = objects[k]
        #print o.is_suppressor, o.suppressor
        if o.is_suppressor:
            continue
        # A) Add object latent variable
        mask_o = o.get_mask()
        o_ik = np.random.normal(mu_o, sigma_o, y.shape[0])
        o_ik -= o_ik.mean() - mu_o
        o_ik /= o_ik.std() * sigma_o
        Xim[:, mask_o] = (np.sqrt(beta_o) * o_ik + \
                        np.sqrt(1 - beta_o) * Xim[:, mask_o].T).T
        if o.suppressor is not None:
            mask_o_suppr = o.suppressor.get_mask()
            Xim[:, mask_o_suppr] = (np.sqrt(beta_o) * o_ik + \
                       np.sqrt(1 - beta_o) * Xim[:, mask_o_suppr].T).T
    return Xim


############################################################################
## Apply causal model on objects
def causal_model(objects, Xim, y, R):
    """Add predictive information: x_ki +=  b_y * y
    """
    for k in xrange(len(objects)):
        o = objects[k]
        if o.is_suppressor:
            continue
        # A) Add object latent variable
        mask_o = o.get_mask()
        # Compute the coeficient according to a desire correlation
        beta_y = corr_to_coef(v_x=1, v_e=1, cov_xe=0, R=R)
        o.beta_y = beta_y
        Xim[:, mask_o] = (Xim[:, mask_o].T + (beta_y * y)).T
    return Xim

############################################################################
## Parameters
n_samples=1000
n_features=900
R = .5
sigma_spatial_smoothing = 1
beta_o = .5

sigma_y = sigma_e = sigma_o = 1 # items std-dev
mu_o = mu_e = mu_y = 0
lx = ly = int(np.round(np.sqrt(n_features)))

############################################################################
## 1. Build images with noize => e_ij
X = np.random.normal(mu_e, sigma_e, n_samples * lx * ly).reshape(n_samples, lx * ly)
Xim = X.reshape(n_samples, lx, ly)
y = np.random.normal(mu_y, sigma_y, n_samples)
y -= y.mean()
y /= y.std()

print X.std(axis=0).mean()
############################################################################
## 1. Pixel-level noize structure: spatial smoothing
if sigma_spatial_smoothing is not 0:
    for i in xrange(Xim.shape[0]):
        Xim[i, :, :] = ndimage.gaussian_filter(Xim[i, :, :],
            sigma=sigma_spatial_smoothing)

# Spatial smoothing reduced the std-dev, reset it to 1
X -= X.mean(axis=0) + mu_e  # Also ensure null mean
X /= X.std(axis=0) * sigma_e

############################################################################
## 2. Build Objects
objects = dice_five(lx, ly)

############################################################################
## 3. Object-level noize structure
Xim = object_model(objects, Xim)

############################################################################
## 4. Causal model
Xim = causal_model(objects, Xim, y, R)

############################################################################
## 6. Predictive model => weight vector
# Estimate (X'X)^-1 X'y for each object (consider suppressor object if present)
## 6.1 Covariance matrix: X'X
# X'Xij = n_samples * (b_y^2 + b_o + (1 - b_o) * CovNij)
# CovNij = RNij * sigma_e ** 2 + 2 * mu_e
# RNij = e(-dist ** 2 / (4 * sigma_spatial_smoothing ** 2) )

labels = np.zeros(lx * ly, dtype=int).reshape(lx, ly)
label = 0
#Cov = np.zeros(X.shape[1] ** 2).reshape((X.shape[1], X.shape[1]))
Xty = np.zeros(X.shape[1])
Cov = np.diagflat([float(sigma_e) ** 2] * X.shape[1])

for k in xrange(len(objects)):
    label += 1
    io = objects[k]
    if io.is_suppressor:
        continue
    label_io = label  # label of informative object
    mask_io_im = io.get_mask()  # mask of informative object in image
    labels[mask_io_im] = label_io
    mask_im = mask_io_im.copy()
    if io.suppressor is not None:
        label += 1
        label_so = label  # label of suppressor object
        # mask of suppressor object in image
        mask_so_im = io.suppressor.get_mask()
        labels[mask_so_im] = label_so
        mask_im += mask_so_im
    idx_im = np.where(mask_im)  # indices of objects in image
    # indices of informative object within mask
    idx_io_mask, = np.where(labels[mask_im] == label_io)
    # get euclidian distances between pixel in the mask
    dist_mat = np.sqrt((idx_im[0][np.newaxis].T - idx_im[0]) ** 2 + \
        (idx_im[1][np.newaxis].T - idx_im[1]) ** 2)
    # True correlation between noize pixels: RN_k within current object k
    RN_k = np.exp(-dist_mat ** 2 / (4 * sigma_spatial_smoothing ** 2))
    # True cov between noize pixels: CovN_k
    CovN_k = RN_k * sigma_e ** 2 + 2 * mu_e
    # True cov between pixels: Cov_k
    # Cov_k = b_y^2 + b_o + (1 - b_o) * CovN_k
    Cov_k = beta_o + (1 - beta_o) * CovN_k
    # Add cov caused by y in informative object
    xy_o_cov_io = covmat_indices(idx_io_mask)
    Cov_k[xy_o_cov_io] += io.beta_y ** 2
    ## Compare with empirical cov
    Cov_k_hat = np.cov(Xim[:, mask_im].T)
    if False:
        p1 = plt.plot(dist_mat, Cov_k_hat, "ob")
        p2 = plt.plot(dist_mat, Cov_k, "or")
        plt.ylabel('Pixels covariance (blue empirical/ red true)')
        plt.xlabel('Pixels distance')
        plt.show()
    ## 6.2 X'y
    #X'y = n_samples * E(X'y)
    # X'y    = n_samples * b_y * E(y^2) if causal else 0
    Xty_k = np.zeros(Cov_k.shape[0])
    Xty_k[idx_io_mask] = n_samples * io.beta_y * (sigma_y + mu_y ** 2)
    Xty_k_hat = np.dot(Xim[:, mask_im].T, y)
    print "Xty corr(theoretical and empirical)", np.corrcoef(Xty_k, Xty_k_hat)[0, 1]
    idx_flt, = np.where(mask_im.ravel()) # indices of pixels in flatten image
    idx_flt_cov = covmat_indices(idx_flt)# indices of pixels in cov of flatten image
    Xty[idx_flt] = Xty_k
    Cov[idx_flt_cov] = Cov_k.ravel()

def plot_map(im):
    fig, ax = plt.subplots()
    cax = ax.matshow(im, cmap=plt.cm.coolwarm)
    mx = np.abs(im).max()
    cbar = fig.colorbar(cax, ticks=[-mx, -mx/4 -mx/2, 0, mx/2, mx/2, mx])
    cbar.set_clim(vmin=-mx, vmax=mx)
    return fig, ax




"""
from sklearn.metrics import r2_score
n_train = int(X.shape[1] / 10)

weights = np.dot(scipy.linalg.inv(Cov * (n_samples - 1)), Xty)
pred = np.dot(X[n_train:, :], weights)
plot_map(im=weights.reshape((lx, ly)))
plt.show()
print r2_score(y, pred), np.corrcoef(y, pred)[0, 1]


weights_icov = Xty / n_train
pred = np.dot(X[n_train:, :], weights_icov)
plot_map(im=weights_icov.reshape((lx, ly)))
plt.show()
print r2_score(y, pred), np.corrcoef(y, pred)[0, 1]


weights_hat = np.dot(scipy.linalg.pinv(X[:n_train, :]), y[:n_train])
pred = np.dot(X[n_train:, :], weights_hat)
plot_map(im=weights_hat.reshape((lx, ly)))
plt.show()
print r2_score(y, pred), np.corrcoef(y, pred)[0, 1]


weights_hat_idcov = np.dot(X[:n_train, :].T, y[:n_train]) / n_train
pred = np.dot(X[n_train:, :], weights_hat_idcov)
plot_map(im=weights_hat_idcov.reshape((lx, ly)))
plt.show()
print r2_score(y, pred), np.corrcoef(y, pred)[0, 1]


"""

"""
p = 4
x = np.zeros(p ** 2).reshape((p, p))
x[np.diag_indices(x.shape[0])] = np.random.rand(x.shape[0])

t = 2
s = 0
x[s:(s+t), s:(s+t)] = np.random.rand(t**2).reshape((t, t))
s = 2
x[s:(s+t), s:(s+t)] = np.random.rand(t**2).reshape((t, t))


np.diag(x)

from scipy.sparse import lil_matrix
A = lil_matrix((1000, 1000))
A[0, :100] = np.random.rand(100)
A[1, 100:200] = A[0, :100]
A.setdiag(np.random.rand(1000))
A = A.tocsr()

import scipy.sparse.linalg 
"""

############################################################################
## 5. Vizu

plt.matshow(get_objects_edges(objects), cmap=plt.cm.gray)
plt.show()

"""
plt.matshow(Xim[0,:,:], cmap=plt.cm.gray)
plt.show()

from sklearn.metrics import r2_score

o1 = objects[0]
o2 = objects[1]
o3 = objects[2]
o4 = objects[3]
o5 = objects[4]

o1s = o1.suppressor ## == o2
o3s = o3.suppressor ## == o4

x1 = Xim[:, o1.c_x, o1.c_y][:, np.newaxis]
x1s = Xim[:, o1s.c_x, o1s.c_y][:, np.newaxis]
x3 = Xim[:, o3.c_x, o3.c_y][:, np.newaxis]
x3s = Xim[:, o3s.c_x, o3s.c_y][:, np.newaxis]
x5 = Xim[:, o5.c_x, o5.c_y][:, np.newaxis]

# Correlation between y a objects
np.corrcoef(x1.ravel(), y)[0, 1]
np.corrcoef(x1s.ravel(), y)[0, 1]
np.corrcoef(x3.ravel(), y)[0, 1]
np.corrcoef(x3s.ravel(), y)[0, 1]
np.corrcoef(x5.ravel(), y)[0, 1]

inter = np.ones(n_samples)[:, np.newaxis]
# With supressor
x = np.hstack([x1, x1s, inter])
betas = np.dot(scipy.linalg.pinv(x), y)
#xtx = np.dot(x.T, x)
#xtx_inv  = scipy.linalg.inv(xtx)
#xty = np.dot(x.T, y)
#np.dot(xtx_inv, xty)
#betas = np.dot(scipy.linalg.pinv(x), y)
y_pred = np.dot(x, betas)
r2_score(y, y_pred)

# Without suppressor
x = np.hstack([x1, inter])
betas = np.dot(scipy.linalg.pinv(x), y)
y_pred = np.dot(x, betas)
r2_score(y, y_pred)
"""
# run pylearn-structured/structured/datasets/samples_generator_struct.py

