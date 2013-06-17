# -*- coding: utf-8 -*-
"""
Created on Thu May 23 09:37:48 2013

TODO: Fix the proximal operators. Normalisation?

@author: Tommy Löfstedt
@email: tommy.loefstedt@cea.fr
"""

import numpy as np
import structured.preprocess as preprocess
import structured.models as models

import structured.utils as utils
from structured.utils.testing import assert_array_almost_equal
from structured.utils.testing import orth_matrix, fleiss_kappa

import matplotlib.pyplot as plot
import matplotlib.cm as cm


def test():

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
    sigma = np.dot(u.T, u)
    mean = np.zeros(p)

    X = np.random.multivariate_normal(mean, sigma, n)
    y = np.dot(X, beta1D)

    eps = 0.01
    maxit = 50000

    num_mus = 1
    mus = [0] * num_mus
    mus[0] = 1.0
#    mus[1] = 0.01
#    mus[2] = 0.0001
#    mus[3] = 0.000001
#    mus[4] = 0.00000001

    # Linear regression
#    lr = models.LinearRegression()
#    lr.set_max_iter(maxit)
#    lr.set_tolerance(eps)
#    lr.fit(X, y)
#    computed_beta = lr.beta
#
#    plot.subplot(4, 4, 1)
#    plot.plot(beta1D[:, 0], '-', computed_beta[:, 0], '*')
#    plot.title("Linear regression")
#
#    plot.subplot(4, 4, 2)
#    plot.imshow(np.reshape(computed_beta, (pz, py, px))[0, :, :],
#                interpolation='nearest', cmap=cm.gist_rainbow)
#
#    lr = models.RidgeRegressionTV(utils.TOLERANCE, 0.0, (pz, py, px),
#                                  mu=mus[0])
#    lr.set_max_iter(maxit)
#    lr.set_tolerance(eps)
#    lr.fit(X, y)
#    computed_beta = lr.beta
#
#    plot.subplot(4, 4, 3)
#    plot.plot(beta1D[:, 0], '-', computed_beta[:, 0], '*')
#    plot.title("Linear regression")
#
#    plot.subplot(4, 4, 4)
#    plot.imshow(np.reshape(computed_beta, (pz, py, px))[0, :, :],
#                interpolation='nearest', cmap=cm.gist_rainbow)

    # LASSO (Linear regression + L1 penalty)
    l = 1.0
    l1 = models.Lasso(l)
    l1.set_max_iter(maxit)
    l1.set_tolerance(eps)
    l1.fit(X, y)
    computed_beta = l1.beta
    print "ss: ", np.sum(l1.beta[:300] ** 2.0)

    plot.subplot(4, 4, 5)
    plot.plot(beta1D[:, 0], '-', computed_beta[:, 0], '*')
    plot.title("LASSO")

    plot.subplot(4, 4, 6)
    plot.imshow(np.reshape(computed_beta, (pz, py, px))[0, :, :],
                interpolation='nearest', cmap=cm.gist_rainbow)

    l1 = models.EGMLinearRegressionL1L2(l, 0.00002, p, mu=mus[0])
    l1.set_max_iter(maxit)
    l1.set_tolerance(eps)
    l1.fit(X, y)
    computed_beta = l1.beta
    print "ss: ", np.sum(l1.beta[:300] ** 2.0)

    plot.subplot(4, 4, 7)
    plot.plot(beta1D[:, 0], '-', computed_beta[:, 0], '*')
    plot.title("LASSO")

    plot.subplot(4, 4, 8)
    plot.imshow(np.reshape(computed_beta, (pz, py, px))[0, :, :],
                interpolation='nearest', cmap=cm.gist_rainbow)

    plot.show()
    return

    # Linear regression + Total variation penalty
    gamma = 1.0
    lrtv = models.LinearRegressionTV(gamma, (pz, py, px), mu=mus[0])
    lrtv.set_max_iter(maxit)
    lrtv.set_tolerance(eps)
    lrtv.fit(X, y)
    computed_beta = lrtv.beta

    plot.subplot(4, 4, 9)
    plot.plot(beta1D[:, 0], '-', computed_beta[:, 0], '*')
    plot.title("Linear regression + TV")

    plot.subplot(4, 4, 10)
    plot.imshow(np.reshape(computed_beta, (pz, py, px))[0, :, :],
                interpolation='nearest', cmap=cm.gist_rainbow)

    # Lasso + Total variation penalty (Linear regression + L1 + TV)
    l = 1.0
    gamma = 1.0
    lrtv = models.LinearRegressionL1TV(l, gamma, (pz, py, px), mu=mus[0])
    lrtv.set_max_iter(maxit)
    lrtv.set_tolerance(eps)
    lrtv.fit(X, y)
    computed_beta = lrtv.beta

    plot.subplot(4, 2, 13)
    plot.plot(beta1D[:, 0], '-', computed_beta[:, 0], '*')
    plot.title("Linear regression + L1 + TV")

    plot.subplot(4, 2, 14)
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


if __name__ == "__main__":

    test()