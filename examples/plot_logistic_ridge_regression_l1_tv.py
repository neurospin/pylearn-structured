# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:00:10 2013

@author: edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from parsimony.datasets import make_classification_struct
import parsimony.tv as tv
from parsimony.estimators import RidgeLogisticRegression_L1_TV
from sklearn.linear_model import LogisticRegression
from parsimony.utils import plot_map2d

###########################################################################
## Dataset
n_samples = 500
shape = (100, 100, 1)
X3d, y, beta3d, proba = make_classification_struct(n_samples=n_samples,
                                                    shape=shape, snr=5)
X = X3d.reshape((n_samples, np.prod(beta3d.shape)))
plt.plot(proba[y.ravel() == 1], "ro", proba[y.ravel() == 0], "bo")
plt.show()

n_train = 100
Xtr = X[:n_train, :]
ytr = y[:n_train]
Xte = X[n_train:, :]
yte = y[n_train:]

alpha_g = 1.  # global penalty

###########################################################################
## Use sklearn LogisticRegression
# Minimize:
# f(beta) = - C Sum wi[yi log(pi) + (1 − yi) log(1 − pi)] + 1/2 * ||beta||^2_2
ridgelr = LogisticRegression(C=1.0 / alpha_g, fit_intercept=False)
%time yte_pred_ridgelr = ridgelr.fit(Xtr, ytr).predict(Xte)
_, recall_ridgelr, _, _ = precision_recall_fscore_support(yte, yte_pred_ridgelr, average=None)

###########################################################################
## RidgeLogisticRegression_L1_TV
# Minimize:
#    f(beta, X, y) = - Sum wi * (yi * log(pi) + (1 − yi) * log(1 − pi))
#                    + k/2 * ||beta||^2_2 
#                    + l * ||beta||_1
#                    + g * TV(beta)
k, l, g = alpha_g * np.array((.1, .4, .5))  # l2, l1, tv penalties

A, n_compacts = tv.A_from_shape(beta3d.shape)
ridgel1tv = RidgeLogisticRegression_L1_TV(k, l, g, A)
%time yte_pred_ridgel1tv = ridgel1tv.fit(Xtr, ytr).predict(Xte)
_, recall_ridgel1tv, _, _ = precision_recall_fscore_support(yte, yte_pred_ridgel1tv, average=None)
# 100 x 100 Wall time: 479.72 s
# 500 x 500 Wall time: 10116.70 s

###########################################################################
## Plot
plot = plt.subplot(131)
plot_map2d(beta3d.reshape(shape), plot, title="beta star")
plot = plt.subplot(132)
plot_map2d(ridgelr.coef_.reshape(shape), plot, title="beta LR L2 (%.2f, %.2f)" % tuple(recall_ridgelr))
plot = plt.subplot(133)
plot_map2d(ridgel1tv.beta.reshape(shape), plot, title="beta LR L1 L2 TV (%.2f, %.2f)" % tuple(recall_ridgel1tv))
plt.show()