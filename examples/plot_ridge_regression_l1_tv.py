# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 15:00:10 2013

@author: ed203246
"""
import numpy as np
import matplotlib.pyplot as plt
import parsimony.datasets
import parsimony.tv
import parsimony.estimators as estimators

n_samples = 100
shape = (30, 30, 1)


X3d, y, betastar = parsimony.datasets.make_regression_struct(n_samples=n_samples,
    shape=shape, r2=.75)
X = X3d.reshape((n_samples, np.prod(shape)))

alpha = 10.
k, l, g = alpha * .1,  alpha * .4, alpha * .5


A, n_compacts = parsimony.tv.A_from_shape(shape)


tvl1l2 = estimators.RidgeRegression_L1_TV(k, l, g, A)
tvl1l2.fit(X, y)

plot = plt.subplot(121)
cax = plot.matshow(betastar.squeeze(), cmap=plt.cm.coolwarm)
plt.title("Beta star")
plot = plt.subplot(122)
cax = plot.matshow(tvl1l2.beta.reshape(shape).squeeze(), cmap=plt.cm.coolwarm)
mx = np.abs(tvl1l2.beta).max()
ticks = np.array([-mx, -mx / 4 - mx / 2, 0, mx / 2, mx / 2, mx])
cbar = plt.colorbar(cax, ticks=ticks)
cbar.set_clim(vmin=-mx, vmax=mx)

plt.title("Beta hat")
plt.show()
