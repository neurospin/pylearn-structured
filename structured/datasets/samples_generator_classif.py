# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 12:29:03 2013

@author: edouard.duchesnay@cea.fr
"""

import numpy as np
import scipy.linalg


def make_classification(n_samples=100, n_complementary_patterns=1,
                        size_complementary_patterns=2, n_redundant_patterns=1,
                        size_redundant_patterns=2, n_independant_features=2,
                        snr=3., grp_proportion=.5, n_noize=None):
    """
    Parameters
    ----------

    n_samples: int
        nb of samples, defualt 100.

    n_complementary_patterns: int
        Number of patterns made of complementary features. In those patterns
        of features, groups means is orthogonal to shared variance, ie.: the
        predictive information is in un-shared variance. Features provide
        complementary information. All (at least two of them) features should
        be included in a predictive model to exploit the pattern. The orientation
        of the group means is colinear to the smallest eigen vector. Default 1.

    size_complementary_patterns: int
        The number of features in each pattern. Default 2.

    n_redundant_patterns: int
        Number of patterns made of redundant features. In those patterns
        of features, groups means is colinear to shared variance, ie.: the
        predictive information is in shared variance. Features provide
        redundant predictive information. Only one feature should
        be included in a predictive model to exploit the pattern. The orientation
        of the group means is colinear to the largest eigen vector. Default 1.

    size_redundant_patterns: int
        The number of features in each pattern. Default 2.

    n_independant_features: int
        Number of "independant" informative features.

    n_noize:
        Number of non informative features. Default 10 * number of informative
        features.

    snr: float
        Global Signal to noize ratio

    grp_proportion: float
        The proportion of samples of the first group. Default 0.5

    Returns
    -------
    X : array of shape [n_samples, n_features]
        The generated samples.

    y : array of shape [n_samples]
        The integer labels for class membership of each sample.

    weigths: The weigths of a Fisher's linear discriminant function, knowing
        the true covariance and mean os the data: Sigma^-1 (m1 - m0)

    m0, m1, cov the parameters used to generate the multivariate normal samples.
    """
    # distribute snr over informative patterns such that feature contribute equally
    # to the snr
    n_informative_features = \
    n_complementary_patterns * size_complementary_patterns + \
    n_redundant_patterns * size_redundant_patterns + \
    n_independant_features

    snr_per_feature = snr / n_informative_features
    snr_per_complementary_pattern = size_complementary_patterns * snr_per_feature
    snr_per_redundant_pattern = size_redundant_patterns * snr_per_feature

    var = 1.
    cov_informative = .5

    # build covariance matrix and means vectors
    cov = np.zeros((n_informative_features, n_informative_features))
    m0 = np.zeros(n_informative_features)
    m1 = np.zeros(n_informative_features)

    # n_complementary_patterns
    idx = 0
    for i in xrange(n_complementary_patterns):
        bloc = cov[idx:(idx + size_complementary_patterns),
                   idx:(idx + size_complementary_patterns)]
        bloc[::] = cov_informative
        bloc[np.diag_indices(bloc.shape[0])] = var
        la, v = scipy.linalg.eig(bloc)
        la = np.real(la)
        min_var = np.min(np.abs(la))
        dmeans = v[:, np.argmin(np.abs(la))].ravel() * np.sqrt(min_var) * \
            snr_per_complementary_pattern
        m1[idx:(idx + size_complementary_patterns)] += dmeans
        idx += size_complementary_patterns

    # n_redundant_patterns
    for i in xrange(n_redundant_patterns):
        bloc = cov[idx:(idx + size_redundant_patterns),
                   idx:(idx + size_redundant_patterns)]
        bloc[::] = cov_informative
        bloc[np.diag_indices(bloc.shape[0])] = var
        la, v = scipy.linalg.eig(bloc)
        la = np.real(la)
        max_var = np.max(np.abs(la))
        dmeans = v[:, np.argmax(np.abs(la))].ravel() * np.sqrt(max_var) * \
            snr_per_redundant_pattern
        #dmeans = snr_per_feature * 3 * np.sqrt(min_var) * dmeans
        m1[idx:(idx + size_redundant_patterns)] += dmeans
        idx += size_redundant_patterns

    # independant features
    for i in xrange(n_independant_features):
        cov[idx, idx] = var
        m1[idx] = np.sqrt(var) * snr_per_feature
        idx += 1

    # Nomalize m1 to scale to the desire global snr
    snr_actual = np.sqrt(np.dot(np.dot(scipy.linalg.inv(cov), m1),  m1))
    m1 = m1 * snr / snr_actual

    n_g0 = int(np.round(n_samples * grp_proportion))

    X0 = np.random.multivariate_normal(m0, cov, n_g0)
    X1 = np.random.multivariate_normal(m1, cov, n_samples - n_g0)
    X = np.vstack([X0, X1])
    y = np.array([0] * n_g0 + [1] * (n_samples - n_g0))

    weigths = np.dot((m1 - m0), scipy.linalg.inv(cov))

    # Add noize
    if n_noize is None:
        n_noize = n_informative_features * 10

    X = np.hstack([X, np.random.rand(n_samples, n_noize)])
    weigths = np.concatenate((weigths, [0] * n_noize))

    return X, y, weigths, m0, m1, cov

if __name__ == '__main__':
    X, y, weigths, m0, m1, cov = make_classification(n_samples=500, n_complementary_patterns=1,
                            size_complementary_patterns=2, n_redundant_patterns=1,
                            size_redundant_patterns=2, n_independant_features=2,
                            snr=3., grp_proportion=.5, n_noize=None)

    import matplotlib.pyplot as plt

    proj = np.dot(X, weigths)

    plt.subplot(221)
    plt.plot(X[y == 0, 0], X[y == 0, 1], 'or', X[y == 1, 0], X[y == 1, 1], 'ob')
    plt.xlabel("X0")
    plt.ylabel("X1")
    plt.title('Complementary pattern')

    plt.subplot(222)
    plt.plot(X[y == 0, 2], X[y == 0, 3], 'or', X[y == 1, 2], X[y == 1, 3], 'ob')
    plt.xlabel("X2")
    plt.ylabel("X3")
    plt.title('Redundant pattern')

    plt.subplot(223)
    plt.plot(X[y == 0, 4], X[y == 0, 5], 'or', X[y == 1, 4], X[y == 1, 5], 'ob')
    plt.xlabel("X4")
    plt.ylabel("X5")
    plt.title('Independant features')

    plt.subplot(224)
    plt.plot(proj[y==0], 'or', proj[y==1], 'ob')
    plt.xlabel("X'w")
    plt.title('Projection on discriminative axis')

    plt.show()
