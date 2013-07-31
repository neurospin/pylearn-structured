# -*- coding: utf-8 -*-
"""
Created on Thu May 16 09:41:13 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: TBD
"""

import numpy as np
import strukturerad.utils as utils
import strukturerad.models as models
#import strukturerad.preprocess as preprocess
import strukturerad.start_vectors as start_vectors
import strukturerad.loss_functions as loss_functions
import strukturerad.algorithms as algorithms
#import strukturerad.data.simulated.lasso as lasso
#import strukturerad.data.simulated.ridge as ridge
#import strukturerad.data.simulated.l1_l2 as l1_l2
#import strukturerad.data.simulated.l1_l2_2D as l1_l2_2D
#import strukturerad.data.simulated.ridge_2D as ridge_2D
#import strukturerad.data.simulated.lasso_2D as lasso_2D
#import strukturerad.data.simulated.l2_2D as l2_2D
#import strukturerad.data.simulated.l1_tv as l1_tv
#import strukturerad.data.simulated.l1_l2_tv as l1_l2_tv
#import strukturerad.data.simulated.l1_l2_tv_2D as l1_l2_tv_2D
#from sklearn.datasets import load_linnerud
from time import time

import matplotlib.pyplot as plot
import matplotlib.cm as cm
#import pylab
import copy


np.random.seed(42)

num_plots = 8

eps = 0.01
maxit = 10000

px = 100
py = 1
pz = 1
p = px * py * pz  # Must be even!
n = 60
X = np.random.randn(n, p)
betastar = np.concatenate((np.zeros((p / 2, 1)),
                           np.random.randn(p / 2, 1)))
betastar = np.sort(np.abs(betastar), axis=0)
y = np.dot(X, betastar)

print "LinearRegression"
model = models.LinearRegression()
model.set_max_iter(maxit)
model.set_tolerance(eps)
model.fit(X, y)
computed_beta = model.beta()

plot.subplot(num_plots, 1, 1)
plot.plot(betastar[:, 0], '-', computed_beta[:, 0], '*')
plot.title("Linear regression")

print "LinearRegressionL1"
l = 1.0
model = models.LinearRegressionL1(l)
model.set_max_iter(maxit)
model.set_tolerance(eps)
model.fit(X, y)
computed_beta = model.beta()

plot.subplot(num_plots, 1, 2)
plot.plot(betastar[:, 0], '-', computed_beta[:, 0], '*')
plot.title("LinearRegressionL1")

print "LinearRegressionTV"
gamma = 0.1
mu = 0.01
model = models.LinearRegressionTV(gamma, mu=mu, shape=(pz, py, px), A=None, mask=None)
model.set_max_iter(maxit)
model.set_tolerance(eps)
model.fit(X, y)
#cr = models.ContinuationRun(lrtv, [1.0, 0.1, 0.01, 0.001, 0.0001])
#cr.fit(X, y)
computed_beta = model.beta()

plot.subplot(num_plots, 1, 3)
plot.plot(betastar[:, 0], '-', computed_beta[:, 0], '*')
plot.title("LinearRegressionTV")

print "LinearRegressionL1TV"
l = 1.0
gamma = 0.1
mu = 0.01
model = models.LinearRegressionL1TV(l, gamma, mu=mu, shape=(pz, py, px), A=None, mask=None)
model.set_max_iter(maxit)
model.set_tolerance(eps)
model.fit(X, y)
#cr = models.ContinuationRun(lrtv, [1.0, 0.1, 0.01, 0.001, 0.0001])
#cr.fit(X, y)
computed_beta = model.beta()

plot.subplot(num_plots, 1, 4)
plot.plot(betastar[:, 0], '-', computed_beta[:, 0], '*')
plot.title("LinearRegressionL1TV")

print "RidgeRegression"
l = 10.0
model = models.RidgeRegression(l)
model.set_max_iter(maxit)
model.set_tolerance(eps)
model.fit(X, y)
computed_beta = model.beta()

plot.subplot(num_plots, 1, 5)
plot.plot(betastar[:, 0], '-', computed_beta[:, 0], '*')
plot.title("RidgeRegression")

print "RidgeRegressionL1"
l = 0.8
k = 0.2
model = models.RidgeRegressionL1(l, k)
model.set_max_iter(maxit)
model.set_tolerance(eps)
model.fit(X, y)
computed_beta = model.beta()

plot.subplot(num_plots, 1, 6)
plot.plot(betastar[:, 0], '-', computed_beta[:, 0], '*')
plot.title("RidgeRegressionL1")

print "RidgeRegressionTV"
l = 1.0
gamma = 1.0
mu = 0.01
model = models.RidgeRegressionTV(l, gamma, mu=mu, shape=(pz, py, px), A=None, mask=None)
model.set_max_iter(maxit)
model.set_tolerance(eps)
model.fit(X, y)
computed_beta = model.beta()

plot.subplot(num_plots, 1, 7)
plot.plot(betastar[:, 0], '-', computed_beta[:, 0], '*')
plot.title("RidgeRegressionTV")

print "RidgeRegressionL1TV"
l = 1.0
k = 1.0
gamma = 1.0
mu = 0.01
model = models.RidgeRegressionL1TV(l, k, gamma, mu=mu, shape=(pz, py, px), A=None, mask=None)
model.set_max_iter(maxit)
model.set_tolerance(eps)
model.fit(X, y)
computed_beta = model.beta()

plot.subplot(num_plots, 1, 8)
plot.plot(betastar[:, 0], '-', computed_beta[:, 0], '*')
plot.title("RidgeRegressionL1TV")

plot.show()