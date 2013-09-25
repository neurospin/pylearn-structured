# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 12:32:00 2013

@author: Tommy Löfstedt
@email: tommy.loefstedt@cea.fr
"""

__all__ = ['load']

import numpy as np
import structured.utils as utils
import structured.algorithms as algorithms


def load(l, k, g, beta, M, e, snr, shape=None):
    """Returns data generated such that we know the exact solution.

    The data generated by this function is fit to the Linear regression + L1 +
    L2  + Total variation function, i.e.:

        f(b) = (1 / 2).|Xb - y|² + l.|b|_1 + (k / 2).|b|² + g.TV(b),

    where |.|_1 is the L1 norm, |.|² is the squared L2 norm and TV is the
    total variation penalty.

    In order to reduce the numbers of TV cases to handle we enforce a
    constraint on the regression vector, beta, such that all elements are
    non-zero and in decreasing order.

    Parameters
    ----------
    l : The L1 regularisation parameter.

    k : The L2 regularisation parameter.

    g : The total variation regularisation parameter.

    beta : The regression vector to generate data from.

    M : The matrix to use when building data. This matrix carries the desired
            distribution of the generated data. The generated data will be a
            column-scaled version of this matrix.

    e : The error vector e = Xb - y. This vector carries the desired
            distribution of the residual.

    Returns
    -------
    X : The generated X matrix.

    y : The generated y vector.

    beta : The generated regression vector.
    """

    if shape == None:
        shape = (beta.shape[0],)

#    seed = np.random.randint(2147483648)
#
#    low = 0.0
#    high = 1.0
#    for i in xrange(30):
#        np.random.seed(seed)
#        X, y, beta = _generate(l, k, gamma, density, high, M, e)
#        val = np.sqrt(np.sum(np.dot(X, beta) ** 2.0) / np.sum(e ** 2.0))
#        if val > snr:
#            break
#        else:
#            low = high
#            high = high * 2.0
#
#    def f(x):
#        np.random.seed(seed)
#        X, y, beta = _generate(l, k, gamma, density, x, M, e)
#        return np.sqrt(np.sum(np.dot(X, beta) ** 2.0) / np.sum(e ** 2.0)) - snr
#
#    bm = algorithms.BisectionMethod(max_iter=20)
#    bm.run(utils.AnonymousClass(f=f), low, high)
#
#    np.random.seed(seed)
#    X, y, beta = _generate(l, k, gamma, density, bm.x, M, e)
#    print "snr = %.5f = %.5f = |X.b| / |e| = %.5f / %.5f" \
#            % (snr, np.linalg.norm(np.dot(X, beta) / np.linalg.norm(e)),
#               np.linalg.norm(np.dot(X, beta)), np.linalg.norm(e))
#
#    return X, y, beta
    return _generate(l, k, g, beta, M, e, shape)


def _generate(l, k, g, beta, M, e, shape):

    l = float(l)
    k = float(k)
    g = float(g)
    print "gamma:", g
    p = np.prod(shape)

    A = np.zeros((3 * p, p))

    Ax = np.eye(p, p, 1) - np.eye(p, p)
    Ay = np.eye(p, p, shape[-1]) - np.eye(p, p)
    Az = np.eye(p, p, shape[-2] * shape[-1]) - np.eye(p, p)

    X = np.zeros(M.shape)
    for i in xrange(p):
        Mte = np.dot(M[:, i].T, e)

        alpha = 0.0

        # L1
#        if i < ps:
        if abs(beta[i, 0]) > utils.TOLERANCE:
            alpha += -l1 * sign(beta[i, 0])
        else:
            alpha += -l1 * U(-1, 1)

        # L2
        alpha += -l2 * beta[i, 0]

        # TV
        if i == 0:  # Case 1: Positive edge (left-most edge) [?][x>0][+]
            alpha += -gamma * 1.0
#        elif i < ps:  # Case 2 and 3: All neighbours positive [+][x>0][+]
        elif i < p - 1 and abs(beta[i-1, 0]) > utils.TOLERANCE \
                       and abs(beta[i, 0]) > utils.TOLERANCE \
                       and abs(beta[i+1, 0]) > utils.TOLERANCE:
            alpha += -gamma * 0.0
#        elif i == ps:  # Case 4: Positive left, zero right [+][x=0][0]
        elif i < p - 1 and abs(beta[i-1, 0]) > utils.TOLERANCE \
                       and abs(beta[i, 0]) <= utils.TOLERANCE \
                       and abs(beta[i+1, 0]) <= utils.TOLERANCE:
            alpha += -gamma * (-1.0 - u[i+1])
#        elif i < p - 1:  # Case 5: Zero neighbours left and right [0][x=0][0]
        elif i < p - 1 and abs(beta[i-1, 0]) <= utils.TOLERANCE \
                       and abs(beta[i, 0]) <= utils.TOLERANCE \
                       and abs(beta[i+1, 0]) <= utils.TOLERANCE:
            alpha += -gamma * (u[i] - u[i+1])
        elif i == p - 1:  # Case 6: Zero edge (right-most edge) [0][x=0][?]
            alpha += -gamma * u[i]

        alpha /= Mte

        X[:, i] = alpha * M[:, i]

#    b = np.dot(M.T, e)
#    a = np.zeros((p, 1))
#
#    # Case 1: Positive edge
#    a[0, 0] = (-l2 * beta[0, 0] - l1 - gamma) / b[0, 0]
#    X[:, 0] = M[:, 0] * a[0, 0]
#
#    # Case 2 and 3: Positive neighbours left and right
#    for i in xrange(1, ps):
#        a[i, 0] = (-l2 * beta[i, 0] - l1) / b[i, 0]
#        X[:, i] = M[:, i] * a[i, 0]
#
#    # Case 4: Positive neighbour left, zero neighbour right
#    a[ps, 0] = (-l2 * beta[ps, 0] - l1 * U(-1, 1) - gamma * (U(-1, 1) - 1)) \
#                    / b[ps, 0]
#    X[:, ps] = M[:, ps] * a[ps, 0]
#
#    # Case 5: Zero neighbours left and right
#    for i in xrange(ps + 1, p - 1):
#        a[i, 0] = (-l2 * beta[i, 0] - l1 * U(-1, 1) - gamma * (U(-1, 1) + U(-1, 1))) \
#                    / b[i, 0]
#        X[:, i] = M[:, i] * a[i, 0]
#
#    # Case 6: Zero edge
#    a[p - 1, 0] = (-l2 * beta[p - 1, 0] - l1 * U(-1, 1) - gamma * U(-1, 1)) \
#                    / b[p - 1, 0]
#    X[:, p - 1] = M[:, p - 1] * a[p - 1, 0]

    y = np.dot(X, beta) - e

    return X, y


def U(a, b):
#    t = max(a, b)
#    a = float(min(a, b))
#    b = float(t)
#    return (np.random.rand() * (b - a)) + a
    return 0.0


def sign(x):
    if x > 0:
        return 1.0
    elif x < 0:
        return -1.0
    else:
        return 0.0


#def load(l, k, g, beta, M, e):
#    """Returns data generated such that we know the exact solution.
#
#    The data generated by this function is fit to the Linear regression + L1 +
#    L2  + Total variation function, i.e.:
#
#        f(b) = (1 / 2).|Xb - y|² + l.|b|_1 + (k / 2).|b|² + g.TV(b),
#
#    where |.|_1 is the L1 norm, |.|² is the squared L2 norm and TV is the
#    total variation penalty.
#
#    In order to reduce the numbers of TV cases to handle we enforce a
#    constraint on the regression vector, beta, such that all elements are
#    non-zero and in decreasing order.
#
#    Parameters
#    ----------
#    l : The L1 regularisation parameter.
#
#    k : The L2 regularisation parameter.
#
#    g : The total variation regularisation parameter.
#
#    beta : The regression vector to generate data from.
#
#    M : The matrix to use when building data. This matrix carries the desired
#            distribution of the generated data. The generated data will be a
#            column-scaled version of this matrix.
#
#    e : The error vector e = Xb - y. This vector carries the desired
#            distribution of the residual.
#
#    Returns
#    -------
#    X : The generated X matrix.
#
#    y : The generated y vector.
#
#    beta : The generated regression vector.
#    """
##    seed = np.random.randint(2147483648)
##
##    low = 0.0
##    high = 1.0
##    for i in xrange(30):
##        np.random.seed(seed)
##        X, y, beta = _generate(l, k, gamma, density, high, M, e)
##        val = np.sqrt(np.sum(np.dot(X, beta) ** 2.0) / np.sum(e ** 2.0))
##        if val > snr:
##            break
##        else:
##            low = high
##            high = high * 2.0
##
##    def f(x):
##        np.random.seed(seed)
##        X, y, beta = _generate(l, k, gamma, density, x, M, e)
##        return np.sqrt(np.sum(np.dot(X, beta) ** 2.0) / np.sum(e ** 2.0)) - snr
##
##    bm = algorithms.BisectionMethod(max_iter=20)
##    bm.run(utils.AnonymousClass(f=f), low, high)
##
##    np.random.seed(seed)
##    X, y, beta = _generate(l, k, gamma, density, bm.x, M, e)
##    print "snr = %.5f = %.5f = |X.b| / |e| = %.5f / %.5f" \
##            % (snr, np.linalg.norm(np.dot(X, beta) / np.linalg.norm(e)),
##               np.linalg.norm(np.dot(X, beta)), np.linalg.norm(e))
##
##    return X, y, beta
#    return _generate(l, k, g, beta, M, e)
#
#
#def _generate(l1, l2, gamma, beta, M, e):
#
#    l1 = float(l1)
#    l2 = float(l2)
#    gamma = float(gamma)
#    print "gamma:", gamma
##    density = float(density)
##    snr = float(snr)
#    p = M.shape[1]
##    ps = int(round(p * density))
#
##    beta = np.zeros((p, 1))
##    for i in xrange(p):
##        if i < ps:
##            beta[i, 0] = U(0, 1) * snr / np.sqrt(ps)
##        else:
##            beta[i, 0] = 0.0
##    beta = np.flipud(np.sort(beta, axis=0))
#
#    u = [0] * p
#    for i in xrange(p):
#        u[i] = U(-1, 1)
#
#    X = np.zeros(M.shape)
#    for i in xrange(p):
#        Mte = np.dot(M[:, i].T, e)
#
#        alpha = 0.0
#
#        # L1
##        if i < ps:
#        if abs(beta[i, 0]) > utils.TOLERANCE:
#            alpha += -l1 * sign(beta[i, 0])
#        else:
#            alpha += -l1 * U(-1, 1)
#
#        # L2
#        alpha += -l2 * beta[i, 0]
#
#        # TV
#        if i == 0:  # Case 1: Positive edge (left-most edge) [?][x>0][+]
#            alpha += -gamma * 1.0
##        elif i < ps:  # Case 2 and 3: All neighbours positive [+][x>0][+]
#        elif i < p - 1 and abs(beta[i-1, 0]) > utils.TOLERANCE \
#                       and abs(beta[i, 0]) > utils.TOLERANCE \
#                       and abs(beta[i+1, 0]) > utils.TOLERANCE:
#            alpha += -gamma * 0.0
##        elif i == ps:  # Case 4: Positive left, zero right [+][x=0][0]
#        elif i < p - 1 and abs(beta[i-1, 0]) > utils.TOLERANCE \
#                       and abs(beta[i, 0]) <= utils.TOLERANCE \
#                       and abs(beta[i+1, 0]) <= utils.TOLERANCE:
#            alpha += -gamma * (-1.0 - u[i+1])
##        elif i < p - 1:  # Case 5: Zero neighbours left and right [0][x=0][0]
#        elif i < p - 1 and abs(beta[i-1, 0]) <= utils.TOLERANCE \
#                       and abs(beta[i, 0]) <= utils.TOLERANCE \
#                       and abs(beta[i+1, 0]) <= utils.TOLERANCE:
#            alpha += -gamma * (u[i] - u[i+1])
#        elif i == p - 1:  # Case 6: Zero edge (right-most edge) [0][x=0][?]
#            alpha += -gamma * u[i]
#
#        alpha /= Mte
#
#        X[:, i] = alpha * M[:, i]
#
##    b = np.dot(M.T, e)
##    a = np.zeros((p, 1))
##
##    # Case 1: Positive edge
##    a[0, 0] = (-l2 * beta[0, 0] - l1 - gamma) / b[0, 0]
##    X[:, 0] = M[:, 0] * a[0, 0]
##
##    # Case 2 and 3: Positive neighbours left and right
##    for i in xrange(1, ps):
##        a[i, 0] = (-l2 * beta[i, 0] - l1) / b[i, 0]
##        X[:, i] = M[:, i] * a[i, 0]
##
##    # Case 4: Positive neighbour left, zero neighbour right
##    a[ps, 0] = (-l2 * beta[ps, 0] - l1 * U(-1, 1) - gamma * (U(-1, 1) - 1)) \
##                    / b[ps, 0]
##    X[:, ps] = M[:, ps] * a[ps, 0]
##
##    # Case 5: Zero neighbours left and right
##    for i in xrange(ps + 1, p - 1):
##        a[i, 0] = (-l2 * beta[i, 0] - l1 * U(-1, 1) - gamma * (U(-1, 1) + U(-1, 1))) \
##                    / b[i, 0]
##        X[:, i] = M[:, i] * a[i, 0]
##
##    # Case 6: Zero edge
##    a[p - 1, 0] = (-l2 * beta[p - 1, 0] - l1 * U(-1, 1) - gamma * U(-1, 1)) \
##                    / b[p - 1, 0]
##    X[:, p - 1] = M[:, p - 1] * a[p - 1, 0]
#
#    y = np.dot(X, beta) - e
#
#    return X, y
#
#
#def U(a, b):
##    t = max(a, b)
##    a = float(min(a, b))
##    b = float(t)
##    return (np.random.rand() * (b - a)) + a
#    return 0.0
#
#
#def sign(x):
#    if x > 0:
#        return 1.0
#    elif x < 0:
#        return -1.0
#    else:
#        return 0.0