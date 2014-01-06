# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:00:10 2013

@author: edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import parsimony.datasets
import parsimony.tv as tv
from parsimony.estimators import RidgeRegression_L1_TV
from sklearn.linear_model import ElasticNet
from parsimony.utils import plot_map2d

###########################################################################
## Dataset
n_samples = 500
shape = (30, 30, 1)
X3d, y, beta3d = parsimony.datasets.make_regression_struct(n_samples=n_samples,
    shape=shape, r2=.75)
X = X3d.reshape((n_samples, np.prod(shape)))
n_train = 100
Xtr = X[:n_train, :]
ytr = y[:n_train]
Xte = X[n_train:, :]
yte = y[n_train:]

alpha_g = 10.  # global penalty

###########################################################################
## Sklearn Elasticnet
#    Min: 1 / (2 * n_samples) * ||y - Xw||^2_2 +
#        + alpha * l1_ratio * ||w||_1
#        + 0.5 * alpha * (1 - l1_ratio) * ||w||^2_2
alpha = alpha_g * 1. / (2. * n_train)
l1_ratio = .5
enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
yte_pred_enet = enet.fit(Xtr, ytr).predict(Xte)

###########################################################################
## Fit RidgeRegression_L1_TV
k, l, g = alpha_g * np.array((.1, .4, .5))  # l2, l1, tv penalties
A, n_compacts = tv.A_from_shape(shape)
ridgel1tv = RidgeRegression_L1_TV(k, l, g, A)
yte_pred_ridgel1tv = ridgel1tv.fit(X, y).predict(Xte)

###########################################################################
## Plot
plot = plt.subplot(131)
plot_map2d(beta3d.reshape(shape), plot, title="beta star")
plot = plt.subplot(132)
plot_map2d(enet.coef_.reshape(shape), plot, title="beta enet (R2=%.2f)" %
    r2_score(yte, yte_pred_enet))
plot = plt.subplot(133)
plot_map2d(ridgel1tv.beta.reshape(shape), plot, title="beta ridgel1tv (R2=%.2f)" %
    r2_score(yte, yte_pred_ridgel1tv))
plt.show()
