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

import strukturerad.datasets.simulated.l1_l2 as l1_l2

from time import time

import matplotlib.pyplot as plot
import matplotlib.cm as cm
#import pylab
import copy


np.random.seed(42)

num_plots = 8

eps = 0.0001
maxit = 50000

px = 100
py = 1
pz = 1
p = px * py * pz  # Must be even!
n = 60
#X = np.random.randn(n, p)
#betastar = np.concatenate((np.zeros((p / 2, 1)),
#                           np.random.randn(p / 2, 1)))
#betastar = np.sort(np.abs(betastar), axis=0)
#y = np.dot(X, betastar)

e = np.random.randn(n, 1) * 100.0
M = np.random.multivariate_normal(np.zeros(p), np.eye(p,p), n)
print "RidgeRegressionTV"
density = 0.5
l = 1.0
k = 1.0
X, y, betastar = l1_l2.load(l, k, density, 10.0, M, e)

gamma = 0.1
mu = 0.01
model = models.RidgeRegressionL1(l, k, X.shape[1])
model.set_max_iter(maxit)
model.set_tolerance(eps)
model = models.ContinuationGap(model, iterations=250, continuations=100)
model.fit(X, y)
computed_beta = model.beta()

#A = loss_functions.TotalVariation.precompute(shape=(pz, py, px), mask=None, compress=True)
#model2 = models.LinearRegressionTV(gamma, mu=mu, A=A, mask=None, compress=True)
#model2.set_max_iter(maxit)
#model2.set_tolerance(eps)
#model2.fit(X, y)
#computed_beta_2 = model2.beta()

plot.subplot(2, 1, 1)
#plot.plot(betastar[:, 0], '-', computed_beta[:, 0], '*', computed_beta_2[:, 0], 'r*')
plot.plot(betastar[:, 0], '-', computed_beta[:, 0], '*')
plot.title("LinearRegressionTV")

plot.subplot(2, 1, 2)
plot.plot(model.output['f'], '-b')
plot.title("Function value")

plot.show()

## Test the functions compute_mu and compute_gap
#utils.debug("Testing compute_mu and compute_gap:")
#model.set_data(X, y)
#g = model.get_g()
#lr = g.a
#tv = g.b
#
#D = tv.num_compacts() / 2.0
## A innehåller gamma ** 2.0
#A = tv.Lipschitz(1.0)
#l = lr.Lipschitz()
#
##    print "D:", D
##    print "A:", A
##    print "l:", l
#
#def mu_plus(eps):
#    return (-2.0 * D * A + np.sqrt((2.0 * D * A) ** 2.0 \
#            + 4.0 * D * l * eps * A)) / (2.0 * D * l)
#
#def eps_plus(mu):
#    return ((2.0 * mu * D * l + 2.0 * D * A) ** 2.0 \
#            - (2.0 * D * A) ** 2.0) / (4.0 * D * l * A)
#
#utils.debug("Testing eps:")
#for eps in [1000000.0, 100000.0, 10000.0, 1000.0, 100.0, 10.0, \
#            1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]:
#    mu1 = mu_plus(eps)
#    eps1 = eps_plus(mu1)
#    err1 = abs(eps - eps1) / eps
#    mu2 = model.compute_mu(eps)
#    eps2 = model.compute_gap(mu2)
#    err2 = abs(eps - eps2) / eps
#
#    utils.debug("eps: %.8f -> mu: %.8f -> eps: %.8f (err: %.8f)" \
#            % (eps, mu1, eps1, err1))
#    utils.debug("eps: %.8f -> mu: %.8f -> eps: %.8f (err: %.8f)" \
#            % (eps, mu2, eps2, err2))
#
##    if eps < 0.0001:
##        assert err1 < 1.0
##        assert err2 < 1.0
##    elif eps < 0.00001:
##        assert err1 < 0.005
##        assert err2 < 0.005
##    else:
##        assert err1 < 0.0005
##        assert err2 < 0.0005
#
#utils.debug("")
#utils.debug("Testing mu:")
#for mu in [1000000.0, 100000.0, 10000.0, 1000.0, 100.0, 10.0,
#           1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]:
#    eps1 = eps_plus(mu)
#    mu1 = mu_plus(eps_plus(mu))
#    err1 = abs(mu - mu1) / mu
#    eps2 = model.compute_gap(mu)
#    mu2 = model.compute_mu(model.compute_gap(mu))
#    err2 = abs(mu - mu2) / mu
#
#    utils.debug("mu: %.8f -> eps: %.8f -> mu: %.8f (err: %.8f)" \
#            % (mu, eps1, mu1, err1))
#    utils.debug("mu: %.8f -> eps: %.8f -> mu: %.8f (err: %.8f)" \
#            % (mu, eps2, mu2, err2))
#
##    if mu > 100000.0:
##        assert err1 < 0.01
##        assert err2 < 0.01
##    else:
##        assert err1 < 0.0005
##        assert err2 < 0.0005



#print "LinearRegression"
#model = models.LinearRegression()
#model.set_max_iter(maxit)
#model.set_tolerance(eps)
#model.fit(X, y)
#computed_beta = model.beta()
#
#plot.subplot(num_plots, 1, 1)
#plot.plot(betastar[:, 0], '-', computed_beta[:, 0], '*')
#plot.title("Linear regression")
#
#print "LinearRegressionL1"
#l = 1.0
#model = models.LinearRegressionL1(l)
#model.set_max_iter(maxit)
#model.set_tolerance(eps)
#model.fit(X, y)
#computed_beta = model.beta()
#
#plot.subplot(num_plots, 1, 2)
#plot.plot(betastar[:, 0], '-', computed_beta[:, 0], '*')
#plot.title("LinearRegressionL1")
#
#print "LinearRegressionTV"
#gamma = 0.1
#mu = 0.01
#model = models.LinearRegressionTV(gamma, mu=mu, shape=(pz, py, px), A=None, mask=None, compress=True)
#model.set_max_iter(maxit)
#model.set_tolerance(eps)
#model.fit(X, y)
##cr = models.ContinuationRun(lrtv, [1.0, 0.1, 0.01, 0.001, 0.0001])
##cr.fit(X, y)
#computed_beta = model.beta()
#
#A = loss_functions.TotalVariation.precompute(shape=(pz, py, px), mask=None, compress=True)
#model2 = models.LinearRegressionTV(gamma, mu=mu, A=A, mask=None, compress=True)
#model2.set_max_iter(maxit)
#model2.set_tolerance(eps)
#model2.fit(X, y)
#computed_beta_2 = model2.beta()
#
#plot.subplot(num_plots, 1, 3)
#plot.plot(betastar[:, 0], '-', computed_beta[:, 0], '*', computed_beta_2[:, 0], 'r*')
#plot.title("LinearRegressionTV")
#
#print "LinearRegressionL1TV"
#l = 1.0
#gamma = 0.1
#mu = 0.01
#model = models.LinearRegressionL1TV(l, gamma, mu=mu, shape=(pz, py, px), A=None, mask=None)
#model.set_max_iter(maxit)
#model.set_tolerance(eps)
#model.fit(X, y)
##cr = models.ContinuationRun(lrtv, [1.0, 0.1, 0.01, 0.001, 0.0001])
##cr.fit(X, y)
#computed_beta = model.beta()
#
#A = loss_functions.TotalVariation.precompute(shape=(pz, py, px), mask=None, compress=True)
#model2 = models.LinearRegressionL1TV(l, gamma, mu=mu, A=A, mask=None, compress=True)
#model2.set_max_iter(maxit)
#model2.set_tolerance(eps)
#model2.fit(X, y)
#computed_beta_2 = model2.beta()
#
#plot.subplot(num_plots, 1, 4)
#plot.plot(betastar[:, 0], '-', computed_beta[:, 0], '*', computed_beta_2[:, 0], 'r*')
#plot.title("LinearRegressionL1TV")
#
#print "RidgeRegression"
#l = 10.0
#model = models.RidgeRegression(l)
#model.set_max_iter(maxit)
#model.set_tolerance(eps)
#model.fit(X, y)
#computed_beta = model.beta()
#
#plot.subplot(num_plots, 1, 5)
#plot.plot(betastar[:, 0], '-', computed_beta[:, 0], '*')
#plot.title("RidgeRegression")
#
#print "RidgeRegressionL1"
#l = 0.8
#k = 0.2
#model = models.RidgeRegressionL1(l, k)
#model.set_max_iter(maxit)
#model.set_tolerance(eps)
#model.fit(X, y)
#computed_beta = model.beta()
#
#plot.subplot(num_plots, 1, 6)
#plot.plot(betastar[:, 0], '-', computed_beta[:, 0], '*')
#plot.title("RidgeRegressionL1")
#
#print "RidgeRegressionTV"
#l = 1.0
#gamma = 1.0
#mu = 0.01
#model = models.RidgeRegressionTV(l, gamma, mu=mu, shape=(pz, py, px), A=None, mask=None)
#model.set_max_iter(maxit)
#model.set_tolerance(eps)
#model.fit(X, y)
#computed_beta = model.beta()
#
#A = loss_functions.TotalVariation.precompute(shape=(pz, py, px), mask=None, compress=True)
#model2 = models.RidgeRegressionTV(l, gamma, mu=mu, A=A, mask=None, compress=True)
#model2.set_max_iter(maxit)
#model2.set_tolerance(eps)
#model2.fit(X, y)
#computed_beta_2 = model2.beta()
#
#plot.subplot(num_plots, 1, 7)
#plot.plot(betastar[:, 0], '-', computed_beta[:, 0], '*', computed_beta_2[:, 0], 'r*')
#plot.title("RidgeRegressionTV")
#
#print "RidgeRegressionL1TV"
#l = 1.0
#k = 1.0
#gamma = 1.0
#mu = 0.01
#model = models.RidgeRegressionL1TV(l, k, gamma, mu=mu, shape=(pz, py, px), A=None, mask=None)
#model.set_max_iter(maxit)
#model.set_tolerance(eps)
#model.fit(X, y)
#computed_beta = model.beta()
#
#A = loss_functions.TotalVariation.precompute(shape=(pz, py, px), mask=None, compress=True)
#model2 = models.RidgeRegressionL1TV(l, k, gamma, mu=mu, A=A, mask=None, compress=True)
#model2.set_max_iter(maxit)
#model2.set_tolerance(eps)
#model2.fit(X, y)
#computed_beta_2 = model2.beta()
#
#plot.subplot(num_plots, 1, 8)
#plot.plot(betastar[:, 0], '-', computed_beta[:, 0], '*', computed_beta_2[:, 0], 'r*')
#plot.title("RidgeRegressionL1TV")
#
#plot.show()