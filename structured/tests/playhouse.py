# -*- coding: utf-8 -*-
"""
Created on Thu May 16 09:41:13 2013

@author: Tommy Löfstedt
'"""

import numpy as np
import structured.utils as utils
#from structured import *
import structured.models as models
import structured.preprocess as preprocess
import structured.start_vectors as start_vectors
import structured.loss_functions as loss_functions
import structured.algorithms as algorithms
#import multiblock.start_vectors as start_vectors
#import multiblock.prox_ops as prox_ops
#import multiblock.schemes as schemes
#from sklearn.datasets import load_linnerud
from time import time

import matplotlib.pyplot as plot
import matplotlib.cm as cm
#import pylab


def test_lasso():

    np.random.seed(42)

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
    lr = models.LinearRegression()
    lr.set_max_iter(maxit)
    lr.set_tolerance(eps)
    lr.fit(X, y)
    computed_beta = lr.beta

    plot.subplot(3, 1, 1)
    plot.plot(betastar[:, 0], '-', computed_beta[:, 0], '*')
    plot.title("Linear regression")



    print "LASSO"
    l = 20.0
    lasso = models.LASSO(l)
    lasso.set_max_iter(maxit)
    lasso.set_tolerance(eps)

    lasso.fit(X, y)
    computed_beta = lasso.beta

    plot.subplot(3, 1, 2)
    plot.plot(betastar[:, 0], '-', computed_beta[:, 0], '*')
    plot.title("LASSO")



    print "LinearRegressionTV"
    gamma = 0.01
    lrtv = models.LinearRegressionTV(gamma, (pz, py, px), mu=0.01)
    lrtv.set_max_iter(maxit)
    lrtv.set_tolerance(eps)
    cr = models.ContinuationRun(lrtv, [1.0, 0.1, 0.01, 0.001, 0.0001])

    cr.fit(X, y)
    computed_beta = cr.beta

    plot.subplot(3, 1, 3)
    plot.plot(betastar[:, 0], '-', computed_beta[:, 0], '*')
    plot.title("Linear regression + TV")



#    print "LinearRegressionL1TV"
#    gamma = 0.01
#    l = 1.0
#    lrl1tv = models.LinearRegressionL1TV(l, gamma, (pz, py, px), mu=0.01)
#    lrl1tv.set_max_iter(maxit)
#    lrl1tv.set_tolerance(eps)
#    cr = models.ContinuationRun(lrtv, [1.0, 0.1, 0.01, 0.001, 0.0001])
#
#    cr.fit(X, y)
#    computed_beta = cr.beta
#
#    plot.subplot(4, 1, 4)
#    plot.plot(betastar[:, 0], '-', computed_beta[:, 0], '*')
#    plot.title("Linear regression + TV + L1")

    plot.show()


def test_lasso_tv():
    np.random.seed(42)

    x = np.arange(-10, 10, 1)
    y = np.arange(-10, 10, 1)
    nrows, ncols = len(x), len(y)
    px = ncols
    py = nrows
    pz = 1
    p = nrows * ncols
    n = 70
    mask = np.zeros((nrows, ncols))
    beta = np.zeros((nrows, ncols))
    for i in xrange(nrows):
        for j in xrange(ncols):
#            if (((x[i] - 3) ** 2 + (y[j] - 3) ** 2 > 8) &
#                ((x[i] - 3) ** 2 + (y[j] - 3) ** 2 < 25)):
#                mask[i, j] = 1

            if ((x[i] - 3) ** 2 + (y[j] - 3) ** 2 < 25):
                mask[i, j] = 1

            if (((x[i] + 1) ** 2 + (y[j] - 5) ** 2 > 5) &
                ((x[i] + 1) ** 2 + (y[j] - 5) ** 2 < 16)):
                mask[i, j] = 1

            if (y[j] > 1) & (x[i] > 3) & (y[j] + x[i] < 10):
                beta[i, j] = (x[i] - 3) ** 2 + (y[j] - 3) ** 2 + 25

#    beta = np.random.rand(nrows, ncols)
#    beta = np.sort(np.abs(beta), axis=0)
#    beta = np.sort(np.abs(beta), axis=1)

    beta1D = beta.reshape((p, 1))
    mask1D = mask.reshape((p, 1))

#    u = np.random.randn(p, p)
    u = np.eye(p, p)
    sigma = np.dot(u.T, u)
    mean = np.zeros(p)

    X = np.random.multivariate_normal(mean, sigma, n)
    y = np.dot(X, beta1D)

    eps = 0.01
    maxit = 50000

    num_mus = 1
    mus = [0] * num_mus
    mus[0] = 10.0
#    mus[1] = 0.01
#    mus[2] = 0.0001
#    mus[3] = 0.000001
#    mus[4] = 0.00000001

    lr = models.LinearRegression()
    lr.set_max_iter(maxit)
    lr.set_tolerance(eps)
    lr.fit(X, y)
    computed_beta = lr.beta

    plot.subplot(3, 3, 1)
    plot.plot(beta1D[:, 0], '-', computed_beta[:, 0], '*')
    plot.title("Linear regression")

    plot.subplot(3, 3, 2)
    plot.imshow(beta, interpolation='nearest', cmap=cm.gist_rainbow)

    plot.subplot(3, 3, 3)
    plot.imshow(np.reshape(computed_beta, (pz, py, px))[0, :, :],
                interpolation='nearest', cmap=cm.gist_rainbow)



    l = 1.0
    l1 = models.LASSO(l)
    l1.set_max_iter(maxit)
    l1.set_tolerance(eps)
    l1.fit(X, y)
    computed_beta = l1.beta

    plot.subplot(3, 3, 4)
    plot.plot(beta1D[:, 0], '-', computed_beta[:, 0], '*')
    plot.title("LASSO")

    plot.subplot(3, 3, 5)
    plot.imshow(beta, interpolation='nearest', cmap=cm.gist_rainbow)

    plot.subplot(3, 3, 6)
    plot.imshow(np.reshape(computed_beta, (pz, py, px))[0, :, :],
                interpolation='nearest', cmap=cm.gist_rainbow)


    gamma = 10.0
    lrtv = models.LinearRegressionTV(gamma, (pz, py, px), mu=mus[0])
    lrtv.set_max_iter(maxit)
    lrtv.set_tolerance(eps)
    lrtv.fit(X, y)
    computed_beta = lrtv.beta

    plot.subplot(3, 3, 7)
    plot.plot(beta1D[:, 0], '-', computed_beta[:, 0], '*')
    plot.title("Linear regression + TV")

    plot.subplot(3, 3, 8)
    plot.imshow(beta, interpolation='nearest', cmap=cm.gist_rainbow)

    plot.subplot(3, 3, 9)
    plot.imshow(np.reshape(computed_beta, (pz, py, px))[0, :, :],
                interpolation='nearest', cmap=cm.gist_rainbow)

    plot.show()


def test_tv():

    np.random.seed(42)

    x = np.arange(-10, 10, 1)
    y = np.arange(-10, 10, 1)
    nrows, ncols = len(x), len(y)
    px = ncols
    py = nrows
    pz = 1
    p = nrows * ncols
    n = 70
    mask = np.zeros((nrows, ncols))
    beta = np.zeros((nrows, ncols))
    for i in xrange(nrows):
        for j in xrange(ncols):
#            if (((x[i] - 3) ** 2 + (y[j] - 3) ** 2 > 8) &
#                ((x[i] - 3) ** 2 + (y[j] - 3) ** 2 < 25)):
#                mask[i, j] = 1

            if ((x[i] - 3) ** 2 + (y[j] - 3) ** 2 < 25):
                mask[i, j] = 1

            if (((x[i] + 1) ** 2 + (y[j] - 5) ** 2 > 5) &
                ((x[i] + 1) ** 2 + (y[j] - 5) ** 2 < 16)):
                mask[i, j] = 1

            if (y[j] > 1) & (x[i] > 3) & (y[j] + x[i] < 10):
                beta[i, j] = (x[i] - 3) ** 2 + (y[j] - 3) ** 2 + 25

#    beta = np.random.rand(nrows, ncols)
#    beta = np.sort(np.abs(beta), axis=0)
#    beta = np.sort(np.abs(beta), axis=1)

    beta1D = beta.reshape((p, 1))
    mask1D = mask.reshape((p, 1))

    r = 0.0
    u = r * np.random.randn(p, p)
    u += (1.0 - r) * np.eye(p, p)
    sigma = np.dot(u.T, u)
    mean = np.zeros(p)

#    pylab.imshow(beta, extent=(x.min(), x.max(), y.max(), y.min()),
#               interpolation='nearest', cmap=cm.gist_rainbow)
#    pylab.show()
#
#    pylab.imshow(mask, extent=(x.min(), x.max(), y.max(), y.min()),
#               interpolation='nearest', cmap=cm.gist_rainbow)
#    pylab.show()

    X = np.random.multivariate_normal(mean, sigma, n)
    y = np.dot(X, beta1D)

#    px = 1
#    py = 300
#    pz = 1
#    p = px * py * pz  # Must be even!
#    n = 50
#    X = np.random.randn(n, p)
#    betastar = np.concatenate((np.zeros((p / 2, 1)),
#                               np.random.randn(p / 2, 1)))
#    beta1D = np.sort(np.abs(betastar), axis=0)
#    y = np.dot(X, beta1D)

    eps = 0.001
    maxit = 100000

    gamma = 5.0
    l = 0.1
    en_lambda = 0.95

    num_mus = 1
    mus = [0] * num_mus
    mus[0] = 0.1  # 2.0 * eps / (p - 1.0)
#    mus[1] = 0.01
#    mus[2] = 0.0001
#    mus[3] = 0.000001
#    mus[4] = 0.00000001
#    for k in xrange(0, num_mus - 1):
#        tau = 2.0 / (float(k) + 3.0)
#        mus[k + 1] = (1.0 - tau) * mus[k]

#    r = 0
#    for i in xrange(X.shape[1]):
#        r = max(r, abs(utils.cov(X[:, [i]], y)))
#    mus = [r * 0.5 ** i for i in xrange(num_mus)]

    total_start = time()
    init_start = time()

    mask1D = mask1D.flatten().astype(int).tolist()
    preprocess_mask = preprocess.Mask(mask1D)
#    X = preprocess_mask.process(X)

    lrtv = models.LinearRegressionTV(gamma, (pz, py, px), mu=mus[0])#,
                                     #mask=mask1D)
    lrtv.set_max_iter(maxit)
    lrtv.set_tolerance(eps)
    method = lrtv
#    cr = models.ContinuationRun(lrtv, mus)
#    method = cr

#    rrtv = RidgeRegressionTV(1.0 - en_lambda, gamma, (pz, py, px), mask=mask1D)
#    rrtv.set_max_iter(maxit)
#    rrtv.set_tolerance(eps)
#    method = rrtv

    print "Init time:", (time() - init_start)

    method.fit(X, y)
    computed_beta = method.beta
#    computed_beta = preprocess_mask.revert(method.beta.T).T

    alg = method.get_algorithm()
    print "Algorithm:", alg
    print "Total time:", (time() - total_start)
    print "Total iterations:", alg.iterations, "(%d)" % len(alg.f)
    print "Error:", alg.f[-1]

#    from scipy import fftpack
#    ft = fftpack.fft(alg.f)
#    n = ft.shape[0]
#    k = 20000
#    if n % 2 != 0:
#        n += 1  # Must be even
#    test = np.array(ft)
#    test[k:n / 2] = 0
#    test[n / 2:-k] = 0
#    ift = fftpack.ifft(test).real
#    from scipy.interpolate import UnivariateSpline
#    x = range(len(alg.f))
#    y = np.log10(alg.f)
#    spline = UnivariateSpline(x, y, s=1)
#    smooth = spline(x)

#    y = (alg.f[30000:])
#    n = len(y)

#    k = 10
#    if n % k != 0:
#        n -= n % k
#    y = y[:n]
#    x = range(n)
#    print "len: ", n
#
#    from scipy.interpolate import UnivariateSpline
#    spline = UnivariateSpline(x, y, s=5e-16, k=1)
#    smooth = spline(x)

#    smooth = [0] * n
#    for i in xrange(0, n, k):
#        X = np.reshape(np.array(x[i:i + k]), (k, 1))
#        X = np.hstack((np.ones((k, 1)), X))
#        b = np.dot(np.linalg.pinv(X), y[i:i + k])
#        smooth[i:i + k] = np.dot(X, b)

#    for i in xrange(n):
#        left = max(0, i - k)
#        right = min(i + k, n - 1) + 1
#        smooth[i] = sum(y[left:right]) / (right - left)
#        print "i:", i
#        print "left:", left
#        print "right:", right
#        print "values:", y[left:right]
#        print "sum:", sum(y[left:right])
#        print "denom:", (right - left)
#        print
#    F = np.array(y)
#    S = np.array(smooth)
#    d = np.min(F)
#    F /= d
#    S /= d
#    y = F.tolist()
#    smooth = S.tolist()
#
#    print "alg.f[i]:", y[-1]
#    i = len(y) - 1
#    while y[i] == y[-1]:
#        i -= 1
#    print "alg.f[i]:", y[i]
#    print "diff:", (y[i] - y[-1])
#
#    fig = plot.figure()
#    ax = fig.add_subplot(1, 1, 1)
#    ax.plot(x, y, '-b', x, smooth, '-r')
##    ax.plot(x, y, '-b')
#    plot.title("f: " + str(alg.f[-1]))
#    plot.show()

#    return

    plot.subplot(2, 2, 1)
    plot.plot(beta1D[:, 0], '-', computed_beta[:, 0], '*')
    plot.title("Iterations: " + str(alg.iterations))

    plot.subplot(2, 2, 3)
    plot.plot(alg.f, '-b')
    plot.title("f: " + str(alg.f[-1]))

    plot.subplot(2, 2, 2)
    plot.imshow(beta,  # , extent=(x.min(), x.max(), y.max(), y.min()),
                interpolation='nearest', cmap=cm.gist_rainbow)

    plot.subplot(2, 2, 4)
    plot.imshow(np.reshape(computed_beta, (pz, py, px))[0, :, :],# + mask,
                # extent=(x.min(), x.max(), y.max(), y.min()),
                interpolation='nearest', cmap=cm.gist_rainbow)
    plot.show()


def test_logistic_regression():

    import numpy as np

    n = 200
    p = 50
    # generate a Gaussian dataset
    x = np.random.randn(n, p)
    # generate a beta with "overlapping groups" of coefficients
    beta1 = beta2 = beta3 = np.zeros((p, 1))
    beta1[0:20] = np.random.randn(20, 1)
    beta2[15:35] = np.random.randn(20, 1)
    beta3[27:50] = np.random.randn(23, 1)
    beta = beta1 + beta2 + beta3

    # compute X beta
    combi = np.dot(x, beta)

    # compute the class of each individual
    proba = 1 / (1 + np.exp(-combi))
    y = np.zeros((n, 1))
    for i in xrange(n):
        y[i] = np.random.binomial(1, proba[i], 1)


def test_data():

#    vector = start_vectors.GaussianCurveVector(normalise=False)
#
#    M = 100
#    N = 100
#    n_points = 5
#
#    means = np.random.rand(1, 2)
#    for i in xrange(1, n_points):
##        p = np.random.rand(1, 2)
##        while np.any(np.sqrt(np.sum((means - p) ** 2.0, axis=1)) < max(0.2, (1.0 / n_points))):
##            p = np.random.rand(1, 2)
##        print np.sqrt(np.sum((means - p) ** 2.0, axis=1))
##        means = np.vstack((means, p))
#
##        det = 0.0
##        p_best = 0
##        for j in xrange(100):
##            p = np.random.rand(1, 2)
##            Ap = np.vstack((means, p))
##            det_curr = abs(np.linalg.det(np.dot(Ap.T, Ap)))
##            if det_curr > det:
##                p_best = p
##                det = det_curr
##        print det
##        means = np.vstack((means, p_best))
#
##    while abs(np.linalg.det(np.dot(means.T, means))) < 0.15:
##        means = np.random.rand(n_points, 2)
#
#        dist = 0.0
#        p_best = 0
#        for j in xrange(20):
#            p = np.random.rand(1, 2)
#            dist_curr = np.min(np.sqrt(np.sum((means - p) ** 2.0, axis=1)))
#            if dist_curr > dist:
#                p_best = p
#                dist = dist_curr
#            if dist_curr > 0.3:
#                break
#        means = np.vstack((means, p_best))
#
#    means[means < 0.05] = 0.05
#    means[means > 0.95] = 0.95
#    means[:, 0] *= M
#    means[:, 1] *= N
#    means = means.tolist()
##    means = [[0.3 * M, 0.3 * N], [0.7 * M, 0.7 * N]]
#
#    covs = [0] * n_points
#    for i in xrange(n_points):
#        S1 = np.diag((np.abs(np.diag(np.random.rand(2, 2))) * 0.5) + 0.5)
#
#        S2 = np.random.rand(2, 2)
#        S2 = (((S2 + S2.T) / 2.0) - 0.5) * 0.9  # [0, 0.45]
#        S2 = S2 - np.diag(np.diag(S2))
#
#        S = S1 + S2
#
#        S /= np.max(S)
#
#        S *= float(min(M, N))
#
#        covs[i] = S.tolist()
#
##    d = min(M, N)
##    covs = [[[1.0*M, 0.2*d], [0.2*d, 1.0*N]],
##            [[1.0*M, -0.2*d], [-0.2*d, 1.0*N]]]
#
#    size=[M, N]
#    dims = 2
#    p = size[0] * size[1]
##    S = 2.0 * (np.random.rand(dims, dims) - 0.5)
##    S = np.dot(S.T, S) / 2.0
##    for i in xrange(dims):
##        if abs(S[i, i]) < 0.5:
##            if S[i, i] > 0:
##                S[i, i] = 0.5
##            else:
##                S[i, i] = -0.5
##    S = (p ** (1.0 / dims)) * S / np.max(S)
##    X = vector.get_vector(shape=(p, 1), dims=dims)
#    X = np.zeros((p, 1))
#    for i in xrange(n_points):
#        X = X + vector.get_vector(size=size, dims=dims,
#                                  mean=means[i], cov=covs[i])
#
#    X = np.reshape(X, size)

    dims = 2
    size = [100, 100]
    vector = start_vectors.GaussianCurveVectors(num_points=3, normalise=False)

    w = vector.get_vector(size=size, dims=dims)
    X = np.reshape(w, size)

    cmap = cm.hot  # cm.RdBu
    if dims == 1:
        plot.plot(X, '-')
    elif dims == 2:
        plot.imshow(X, interpolation='nearest', cmap=cmap)
    elif dims == 3:
        m = np.max(X)
        for i in xrange(X.shape[0]):
            plot.subplot(X.shape[0], 1, i)
            plot.imshow(X[i, :, :], interpolation='nearest', vmin=0.0, vmax=m,
                        cmap=cmap)
#            plot.set_cmap('hot')
    plot.show()


def create_data(
        grp_desc =[range(100), range(100,200), range(200,300), range(300,400),range(400,1000)],
        grp_cors = [0.8, 0, 0.8, 0, 0],
        grp_assoc = [0.5, 0.5, 0.3, 0.3, 0],
        grp_firstonly = [True, False, True, False, False],
        n = 100,
        labelswapprob = 0,
        basehaz = 0.2,
        intercept = 0):
    """
    create data with X : n x p observations with groups of variables 
    with intra-group correlation (grp_cors) and effect on the outcome(betas).
     the outcome is logistic and potentially noisy (intercept, labelswapprob)
    """
    p = sum([len(i) for i in grp_desc])
    X = np.zeros( (n, p) ) 
    y = np.zeros( n )
    sigma = np.zeros( (p,p) )
    betas = np.zeros( p )

    for b,c,a,o in zip(grp_desc, grp_cors, grp_assoc, grp_firstonly):
       #print b[0]
       sigma[b[0]:(b[-1]+1), b[0]:(b[-1]+1)] = c
       sigma[b[0]:(b[-1]+1), b[0]:(b[-1]+1)] += (1.0 - c) * np.eye(len(b),len(b))
       #print sigma
       center = np.zeros(len(b))
       X[:,b] = np.random.multivariate_normal(center, sigma[b[0]:(b[-1]+1), b[0]:(b[-1]+1)], n)
       if o:
           betas[b[0]] = a
       else:
           betas[b[0]:(b[-1]+1)] = a

    predlin = np.dot(X, betas)

    p = 1./(1. + np.exp(-(predlin + intercept)))
    for i in xrange(n):
        y[i] = np.random.binomial(1, p[i], 1)

    return X, y, betas, grp_desc, sigma


if __name__ == "__main__":
#    test_tv()
#    test_lasso()
#    test_lasso_tv()
#    test_data()

#    from pylab import *
#    from numpy import outer
#    rc('text', usetex=False)
#    a=outer(arange(0,1,0.01),ones(10))
#    figure(figsize=(10,5))
#    subplots_adjust(top=0.8,bottom=0.05,left=0.01,right=0.99)
#    maps=[m for m in cm.datad if not m.endswith("_r")]
#    maps.sort()
#    l=len(maps)+1
#    for i, m in enumerate(maps):
#        subplot(1,l,i+1)
#        axis("off")
#        imshow(a,aspect='auto',cmap=get_cmap(m),origin="lower")
#        title(m,rotation=90,fontsize=10)
#    show()

#    pz = 2
#    py = 2
#    px = 3
#    p = px * py * pz
##    beta = np.ones((pz * py * px, 1))
##    X = np.reshape(xrange(p), (pz, py, px))
#    X = np.ones((5, pz * py * px))
#    print X
#    lrtv = models.LinearRegressionTV(10.0, (pz, py, px), mu=10.0)
#    beta = lrtv.get_start_vector().get_vector(X)
##    print lrtv._tv.grad(beta).T
#    Ax, Ay, Az = lrtv.get_g().A()
#    print Ax.todense()
#    print Ay.todense()
#    print Az.todense()
#
#    mu = 0.1
#    asx = Ax.dot(beta) / mu
#    asy = Ay.dot(beta) / mu
#    asz = Az.dot(beta) / mu
#
#    print asx
#    print asy
#    print asz
#
#    print "norm: ", np.sqrt(asx ** 2.0 + asy ** 2.0 + asz ** 2.0)
#
#    # Apply projection
#    asx, asy, asz = lrtv.get_g().projection((asx, asy, asz))
#
#    print asx
#    print asy
#    print asz


#    import pickle
#    O = pickle.load(open("/home/tl236864/objs.pickle"))
#    y = O[0]
#    X = O[1]
#    groups = O[2]
#
#    for i in xrange(len(groups) - 1, -1, -1):
#        if len(groups[i]) == 0:
#            del groups[i]
#            print "group %d deleted!" % (i,)
#
#    gamma = 1.0
#    mu = 0.01
#    weights = [1.0] * len(groups)
#
#    lr = loss_functions.LogisticRegressionError()
#    gl = loss_functions.GroupLassoOverlap(gamma, X.shape[1], groups, mu,
#                                          weights)
#    combo = loss_functions.CombinedNesterovLossFunction(lr, gl)
#
#    algorithm = algorithms.ISTARegression(combo)
#    algorithm._set_tolerance(0.01)
#    algorithm._set_max_iter(1000)
#    lr.set_data(X, y)
#    beta = algorithm.run(X, y)

#    # Test group lasso!!
#    import scipy
#
#    np.random.seed(42)
#
#    p = 40
#    betastar = np.zeros((p, 1)).ravel()
##    betastar = [0., 0., 0., 0., 0., .5, .7, 1., .6, .7, 0., 0., 0., 0., 0.]
##    groups = [[5, 6, 7, 8, 9]]
##    groups = [range(p / 3), range(p / 3, 2 * p / 3), range(2 * p / 3, p)]
#    groups = [range(p / 3, 2 * p / 3), range(p)]
#    betastar[groups[0]] = 1
#
#    p = len(betastar)
#    n = 20
#
#    r = 0.0
#    u = r * np.random.randn(p, p)
#    u += (1.0 - r) * np.eye(p, p)
#    sigma = np.dot(u.T, u)
#    mean = np.zeros(p)
#
#    X = np.random.multivariate_normal(mean, sigma, n)
#    y = np.reshape(np.dot(X, betastar), (n, 1))
#
##    eps = 0.0001
#    gamma = 1.0
##    weights = [1.0] * len(groups)
#
##    lrgl = models.LinearRegressionGL(gamma, p, groups)
#    lrgl = models.LinearRegressionTV(gamma, (1, 1, p))
#    cont = models.ContinuationRun(lrgl,
#                                  mus=[10, 1.0, 0.1, 0.01, 0.001, 0.0001])
#    cont.set_tolerance(0.00001)
#    cont.set_max_iter(10000)
##    lrgl.set_data(X, y)
##    lrgl.set_mu(lrgl.compute_mu(eps))
#    cont.fit(X, y)
#
#    alg = cont.get_algorithm()
#    print cont.get_transform()
#    print alg.iterations
#
#    plot.subplot(2, 1, 1)
#    plot.plot(betastar, '-g', cont.beta, '*r')
#
#    plot.subplot(2, 1, 2)
#    plot.plot(alg.f)
#    plot.title("Iterations: " + str(alg.iterations))
#
#    plot.show()


#    #  Test Logistic Group Lasso ==========
#    np.random.seed(42)
#    X, y, betas, groups, sigma = create_data()
#    #    eps = 0.0001
#    gamma = 10.
#    #    weights = [1.0] * len(groups)
#    p = len(betas)
#    lrgl = models.LogisticRegressionGL(gamma,p, groups, mu=None, weights=None)
#    cont = models.ContinuationRun(lrgl,
#                                 tolerances=[ 0.1, 0.01, 0.001, 0.0001])
#    #    lrgl.set_tolerance(eps)        
#    cont.set_max_iter(1000)
#    #    lrgl.set_data(X, y)
#    #    lrgl.set_mu(lrgl.compute_mu(eps))
#    cont.fit(X, y)
#   
#    alg = cont.get_algorithm()
#    print cont.get_transform()
#    print alg.iterations
#
#    plot.subplot(2, 1, 1)
#    plot.plot(betas, '-g', cont.beta, '*r')
#
#    plot.subplot(2, 1, 2)
#    plot.plot(alg.f)
#    plot.title("Iterations: " + str(alg.iterations))
#
#    plot.show()
#
#    # Test group lasso!!
#    import scipy
#
#    return


#    np.random.seed(42)
#
#    maxit = 10000
#
#    px = 100
#    py = 1
#    pz = 1
#    p = px * py * pz  # Must be even!
#    n = 60
#    X = np.random.randn(n, p)
#    betastar = np.concatenate((np.zeros((p / 2, 1)),
#                               np.random.randn(p / 2, 1)))
#    betastar = np.sort(np.abs(betastar), axis=0)
#    y = np.dot(X, betastar)
#
#    m = models.NesterovProximalGradientMethod()
#
#    gamma = 1.0
#    shape = [pz, py, px]
#    mu = 0.01
#    tv = loss_functions.TotalVariation(gamma, shape, mu)
#    lr = loss_functions.LinearRegressionError()
#    combo = loss_functions.CombinedNesterovLossFunction(lr, tv)
#
#    m.set_g(combo)
#    combo.set_data(X, y)
#
#    D = tv.num_compacts() / 2.0
#    print "D:", D
##    _A = tv.Lipschitz(mu) * mu
#    A = tv.Lipschitz(1.0)
##    assert abs(_A - A) < 0.0000001
#    l = lr.Lipschitz()
#
##    import scipy.sparse as sparse
##    A_ = sparse.vstack(tv.A()).todense()
##    L, V = np.linalg.eig(np.dot(A_.T, A_))
##    print max(L)
#
#    print "A:", A
#    print "l:", l
#
#    def mu_plus(eps):
#        return (-2.0 * D * A + np.sqrt((2.0 * D * A) ** 2.0 + 4.0 * D * l * eps * A)) / (2.0 * D * l)
#
#    def eps_plus(mu):
#        return ((2.0 * mu * D * l + 2.0 * D * A) ** 2.0 - (2.0 * D * A) ** 2.0) / (4.0 * D * l * A)
#
#    m.algorithm._set_tolerance(m.compute_tolerance(mu))
#    beta = m.algorithm.run(X, y)
#
#    for eps in [1000000.0, 100000.0, 10000.0, 1000.0, 100.0, 10.0, 1.0, 0.1, 0.01, 0.001]:
#        print "eps: %.7f -> mu: %.7f -> eps: %.7f" % (eps, mu_plus(eps), eps_plus(mu_plus(eps)))
#        print "eps: %.7f -> mu: %.7f -> eps: %.7f" % (eps, m.compute_mu(eps), m.compute_tolerance(m.compute_mu(eps)))
#        print "D * mu = %.7f" % (D * mu)

#    mu1 = []
#    mu2 = []
#    eps = []
#    for _eps in xrange(1, 1000):
#        eps.append(_eps / 1000.0)
#
##        mu1.append(mu_plus(eps[-1]))
##        mu2.append(m.compute_mu(eps[-1]))
#        mu1.append(eps_plus(eps[-1]))
#        mu2.append(m.compute_tolerance(eps[-1]))
#
#    plot.plot(eps, mu1, '-r')
#    plot.plot(eps, mu2, '-g')
#    plot.show()
#    print mu1[-10:]
#    print mu2[-10:]

#    eps = 0.01
#    mu = mu_plus(eps)
#
#    s = time()
#    print "%.5f = %.5f? (%f s)" % (m.compute_mu(eps), mu, time() - s)
#    s = time()
#    print "%.5f = %.5f? (%f s)" % (m.compute_tolerance(mu), eps, time() - s)
#    print
#
#    eps = 0.3
#    mu = mu_plus(eps)
#    s = time()
#    print "%.5f = %.5f? (%f s)" % (m.compute_mu(eps), mu, time() - s)
#    s = time()
#    print "%.5f = %.5f? (%f s)" % (m.compute_tolerance(mu), eps, time() - s)
#    print
#
#    eps = 1.0
#    mu = mu_plus(eps)
#    s = time()
#    print "%.5f = %.5f? (%f s)" % (m.compute_mu(eps), mu, time() - s)
#    s = time()
#    print "%.5f = %.5f? (%f s)" % (m.compute_tolerance(mu), eps, time() - s)
#    print
#
#    mu = 0.00149
#    eps = eps_plus(mu)
#    s = time()
#    print "%.5f = %.5f? (%f s)" % (m.compute_tolerance(mu), eps, time() - s)
#    s = time()
#    print "%.5f = %.5f? (%f s)" % (m.compute_mu(eps), mu, time() - s)
#    print
#
#    mu = 0.01949
#    eps = eps_plus(mu)
#    s = time()
#    print "%.5f = %.5f? (%f s)" % (m.compute_tolerance(mu), eps, time() - s)
#    s = time()
#    print "%.5f = %.5f? (%f s)" % (m.compute_mu(eps), mu, time() - s)
#    print
#
#    mu = 0.03975
#    eps = eps_plus(mu)
#    s = time()
#    print "%.5f = %.5f? (%f s)" % (m.compute_tolerance(mu), eps, time() - s)
#    s = time()
#    print "%.5f = %.5f? (%f s)" % (m.compute_mu(eps), mu, time() - s)
#    print

    np.random.seed(42)

    eps = 0.001
    maxit = 10
    cont_maxit = 100
    gamma = 15.0

    px = 1000
    py = 1
    pz = 1
    p = px * py * pz  # Must be even!
    n = 100
    X = np.random.randn(n, p)
    betastar = np.concatenate((np.zeros((p / 2, 1)),
                               np.random.randn(p / 2, 1)))
    betastar = np.sort(np.abs(betastar), axis=0)
    y = np.dot(X, betastar)

    print "LinearRegressionTV"
    start = time()
    lrtv = models.LinearRegressionTV(gamma, shape=(pz, py, px))
    lrtv.set_tolerance(eps)
    lrtv.set_max_iter(maxit)
    c = models.Continuation(lrtv, cont_maxit)
    c.fit(X, y)
    computed_beta = c.beta
    print "time: ", (time() - start)

    print "f: ", c.get_algorithm().f[-1]
    print "its: ", c.get_algorithm().iterations

    plot.subplot(2, 2, 1)
    plot.plot(betastar[:, 0], '-', computed_beta[:, 0], '*')
    plot.subplot(2, 2, 2)
    plot.plot(c.get_algorithm().f)
    plot.title("Continuation")

    start = time()
    lrtv = models.LinearRegressionTV(gamma, shape=(pz, py, px))
    lrtv.set_tolerance(eps)
    lrtv.set_max_iter(cont_maxit)
    cr = models.ContinuationRun(lrtv, tolerances=[1.0, 0.1, 0.01, 0.001, 0.0001])
    cr.fit(X, y)
    computed_beta = cr.beta
    print "time: ", (time() - start)

    print "f: ", cr.get_algorithm().f[-1]
    print "its: ", cr.get_algorithm().iterations

    plot.subplot(2, 2, 3)
    plot.plot(betastar[:, 0], '-', computed_beta[:, 0], '*')
    plot.subplot(2, 2, 4)
    plot.plot(cr.get_algorithm().f)
    plot.title("Continuation Run")

    plot.show()

#    l = float(1.0)
#    density = float(0.3)  # \in [0, 1]
#    rho = float(1.0)  # ~SNR
#    n = 500
#    p = 1000
#    ps = round(p * density)  # <= p
#    P = (np.random.rand(n, p) - 0.5) * 2.0  # Should be normally distributed
#    v = np.random.rand(n, 1)  # Should be normally distributed
#
#    e = v / utils.norm(v)
#
#    b = np.dot(P.T, e)
#    ind = np.flipud(np.argsort(np.abs(b), axis=0))
#    b = b[ind[:, 0]]
#    sign_b = np.sign(b)
#    abs_b = np.abs(b)
#
#    a_plus = l / abs_b[:ps, [0]]
#
#    xi = np.random.rand(p - ps, 1)
#    a_zero = np.divide(l * xi, abs_b[ps:, [0]])
#    ind = abs_b[ps:, [0]] < l * 1.0  # !!!
#    a_zero[ind] = 1.0
#
#    a = np.vstack((a_plus, a_zero))
#
#    X = P / a.T
#
#    beta = np.zeros((p, 1))
#    xi = np.random.rand(ps, 1) * (rho / np.sqrt(ps))
#    beta[:ps, [0]] = -np.multiply(xi, sign_b[:ps, [0]])
#
#    y = np.dot(X, beta) + e
#
#    tolerance = 0.01
#    maxit = 10000
#
#    for l in xrange(50, 150):
#        l = l / float(100.0)
#        lr = models.Lasso(l)
#        lr.set_tolerance(tolerance)
#        lr.set_max_iter(maxit)
#        lr.fit(X, y)
#
#        print "l = %.2f => %f" % (l, np.sum((beta - lr.beta) ** 2.0))
#
##    plot.plot(beta, '-g', lr.beta, ':*r')
##    plot.show()