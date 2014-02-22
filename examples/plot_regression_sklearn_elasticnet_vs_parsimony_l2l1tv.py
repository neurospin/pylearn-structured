# -*- coding: utf-8 -*-
"""
Created on Wed Jan  1 18:54:46 2014

@author: edouard.duchesnay@cea.fr
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from parsimony.datasets import make_regression_struct

if __name__ == '__main__':
    # utils
    def plot_map(im, plot=None):
        if plot is None:
            plot = plt
        cax = plot.matshow(im, cmap=plt.cm.coolwarm)
        frame = plt.gca()
        frame.get_xaxis().set_visible(False)
        frame.get_yaxis().set_visible(False)
        mx = np.abs(im).max()
        k = 1
        while (10 ** k * mx) < 1 and k < 10:
            k += 1
        ticks = np.array([-mx, -mx / 4 - mx / 2, 0, mx / 2, mx / 2,
                          mx]).round(k + 2)
        cbar = plt.colorbar(cax, ticks=ticks)
        cbar.set_clim(vmin=-mx, vmax=mx)

    n_samples = 500
    shape = (100, 100, 1)
    r2 = .75
    sigma_spatial_smoothing = 1
    random_seed = None
    object_pixel_ratio = .25
    objects = None
    Xim, y, beta = make_regression_struct(n_samples=n_samples,
        shape=shape, r2=r2, sigma_spatial_smoothing=sigma_spatial_smoothing,
        object_pixel_ratio=object_pixel_ratio,
        random_seed=random_seed)
    _, nx, ny, nz = Xim.shape

    X = Xim.reshape((n_samples, nx * ny))
    n_train = min(100, int(X.shape[1] / 10))
    Xtr = X[:n_train, :]
    ytr = y[:n_train]
    Xte = X[n_train:, :]
    yte = y[n_train:]

    plot = plt.subplot(331)
    if beta is not None:
        plot_map(beta.reshape(nx, ny), plot)
    plt.title("beta")

    Xtrc = (Xtr - Xtr.mean(axis=0)) / Xtr.std(axis=0)
    ytrc = (ytr - ytr.mean()) / ytr.std()
    cor = np.dot(Xtrc.T, ytrc).reshape(nx, ny) / ytr.shape[0]
    plot = plt.subplot(332)
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
    plot = plt.subplot(333)
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
    plot = plt.subplot(334)
    plot_map(l1.coef_.reshape((nx, ny)), plot)
    plt.title("Lasso (R2=%.2f)" % r2_score(yte, pred))

    # Enet  ================================================================
    #    Min: 1 / (2 * n_samples) * ||y - Xw||^2_2 +
    #        + alpha * l1_ratio * ||w||_1
    #        + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2
    alpha = alpha_g * 1. / (2. * n_train)
    l1_ratio = .5
    l1l2 = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
    pred = l1l2.fit(Xtr, ytr).predict(Xte)
    plot = plt.subplot(335)
    plot_map(l1l2.coef_.reshape((nx, ny)), plot)
    plt.title("Elasticnet(a:%.2f, l1:%.2f) (R2=%.2f)" % (l1l2.alpha,
              l1l2.l1_ratio, r2_score(yte, pred)))

#    l1l2cv = ElasticNetCV()
#    pred = l1l2cv.fit(Xtr, ytr).predict(Xte)
#    plot = plt.subplot(336)
#    plot_map(l1l2cv.coef_.reshape((nx, ny)), plot)
#    plt.title("ElasticnetCV(a:%.2f, l1:%.2f) (R2=%.2f)" % (l1l2cv.alpha_,
#              l1l2cv.l1_ratio, r2_score(yte, pred)))
#    #plt.show()

    # TVL1L2 ===============================================================
    import parsimony.estimators as estimators
    import parsimony.algorithms.explicit as algorithms
    import parsimony.functions.nesterov.tv as tv


    def ratio2coef(alpha, tv_ratio, l1_ratio):
        l2_ratio = 1 - tv_ratio - l1_ratio
        l, k, g = alpha * l1_ratio,  alpha * l2_ratio, alpha * tv_ratio
        return l, k, g

    eps = 0.01
    alpha = alpha_g

    tv_ratio = .05
    l1_ratio = .9
    l, k, g = ratio2coef(alpha=alpha, tv_ratio=tv_ratio, l1_ratio=l1_ratio)

    A, n_compacts = tv.A_from_shape(shape)
    tvl1l2 = estimators.RidgeRegression_L1_TV(k, l, g, A,
                        algorithm=algorithms.StaticCONESTA(max_iter=100))
    tvl1l2.fit(Xtr, ytr)
    plot = plt.subplot(337)
    plot_map(tvl1l2.beta.reshape(nx, ny), plot)
    r2 = r2_score(yte, np.dot(Xte, tvl1l2.beta).ravel())
    plt.title("L1L2TV(a:%.2f, tv: %.2f, l1:%.2f) (R2=%.2f)" % \
        (alpha, tv_ratio, l1_ratio, r2))

    tv_ratio = .5
    l1_ratio = .45
    l, k, g = ratio2coef(alpha=alpha, tv_ratio=tv_ratio, l1_ratio=l1_ratio)
    tvl1l2 = estimators.RidgeRegression_L1_TV(k, l, g, A,
                        algorithm=algorithms.StaticCONESTA(max_iter=100))
    tvl1l2.fit(Xtr, ytr)
    plot = plt.subplot(338)
    plot_map(tvl1l2.beta.reshape(nx, ny), plot)
    r2 = r2_score(yte, np.dot(Xte, tvl1l2.beta).ravel())
    plt.title("L1L2TV(a:%.2f, tv: %.2f, l1:%.2f) (R2=%.2f)" % \
        (alpha, tv_ratio, l1_ratio, r2))

    tv_ratio = .9
    l1_ratio = .05
    l, k, g = ratio2coef(alpha=alpha, tv_ratio=tv_ratio, l1_ratio=l1_ratio)
    tvl1l2 = estimators.RidgeRegression_L1_TV(k, l, g, A,
                        algorithm=algorithms.StaticCONESTA(max_iter=100))
    tvl1l2.fit(Xtr, ytr)
    plot = plt.subplot(339)
    plot_map(tvl1l2.beta.reshape(nx, ny), plot)
    r2 = r2_score(yte, np.dot(Xte, tvl1l2.beta).ravel())
    plt.title("L1L2TV(a:%.2f, tv: %.2f, l1:%.2f) (R2=%.2f)" % \
        (alpha, tv_ratio, l1_ratio, r2))

    plt.show()
    #run pylearn-parsimony/parsimony/datasets/samples_generator_struct.py
