# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 12:29:03 2013

@author: edouard.duchesnay@cea.fr
"""

import numpy as np
import scipy.linalg


def make_classification(n_samples=100,
                n_complementary_patterns=1, size_complementary_patterns=2,
                n_suppressor_patterns=1,
                n_redundant_patterns=1, size_redundant_patterns=2,
                n_independant_features=2,
                n_noize=None,
                snr=3., grp_proportion=.5,
                random_seed=None,
                full_info=False):
    """ Generate classification samples with features having a covariance
    structure.

    The function creates patterns of informative features with a structure
    of covariance. This covariance structure may be of four types:
    (1) Independant features:
    The is no covariance betwenne informative features. It is the
    "n_independant_features" parameter ("n_redundant_patterns parameter").

    (2) Pattern of redundant features:
    The covariance between features is mixed up with the predictive
    information. Thus each feature of the pattern carries the same predictive
    information, ie.: the features of the pattern are redundant. A predictive
    algorithm need only to select one of them to exploit the predictive
    information contained in the pattern

    (3) Pattern of complementary features:
    The covariance between features is orthogonal with the predictive
    information. Thus the difference of two features within the pattern will
    cancel out non predictive information resulting to a highly predictive
    pattern. This is controled by the "n_complementary_patterns".

    (4) Pattern of one informative feature an one suppressor feature.
    The covariance between features is independant to the predictive
    information. And one feature of the pattern carries predictive information.
    So any other feature of the pattern will cancel out non predictive
    information resulting to a highly predictive pattern.
    This is controled by the "n_suppressor_patterns".

    The SNR (Mahalanobis distance) is distributed accross informative features
    such that the mean of SNR individual feature is constant for all patterns.
    Thus we can observ additionnal SNR emerging from features covariation.

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

    n_suppressor_patterns: int
        Number of patterns made of 2 features: one is linked with the target
        an the other acts as a suppressor variable.

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
        Global Signal to noize ratio ie.: the Mahalanobis distance between 
        the two groups.

    grp_proportion: float
        The proportion of samples of the first group. Default 0.5

    full_info: bool
        If False (default) return only X and y. Otherwise return 

    random_seed: None or int
        See numpy.random.seed(). If not None, it can be used to obtain
        reproducable samples.

    Returns
    -------
    X : array of shape [n_samples, n_features]
        The generated samples.

    y : array of shape [n_samples]
        The integer labels for class membership of each sample.

    If full_info is True, also return:

    weigths: arrays of shape [n_informative_features]
        The weigths of a Fisher's linear discriminant function, knowing
        the true covariance and mean os the data: Sigma^-1 (m1 - m0)

    m0, m1: arrays of shape [n_informative_features]
        The means of the two classes, for the informative_features only.
        Those means were used to generate multivariate normal samples.

    Cov: array of shape [n_informative_features, n_informative_features]
        The within class covariance for the informative_features only.
        This covariance was used to generate multivariate normal samples.    
    """
    n_informative_features = \
    n_complementary_patterns * size_complementary_patterns + \
    n_suppressor_patterns * 2 + \
    n_redundant_patterns * size_redundant_patterns + \
    n_independant_features

    snr_per_feature = snr / n_informative_features

    var = 1.
    cov = .5

    # build covariance matrix and means vectors
    Cov = np.zeros((n_informative_features, n_informative_features))
    m0 = np.zeros(n_informative_features)
    m1 = np.zeros(n_informative_features)

    def calc_pattern_snr(m0, m1, cov):
        return np.sqrt(np.dot(np.dot(scipy.linalg.inv(cov), m1 - m0),  
                              m1 - m0))

    def calc_univ_snrs(m0, m1, var):
        """Indivudual snrs"""
        return np.sqrt((m1 - m0) ** 2 / var)

    # Complementary patterns
    idx = 0
    for i in xrange(n_complementary_patterns):
        bloc = Cov[idx:(idx + size_complementary_patterns),
                   idx:(idx + size_complementary_patterns)]
        bloc[::] = cov
        bloc[np.diag_indices(bloc.shape[0])] = var
        la, v = scipy.linalg.eig(bloc)
        #la = np.real(la)
        eig_vec_min = v[:, np.argmin(np.abs(la))].ravel()
        m1[idx:(idx + size_complementary_patterns)] += eig_vec_min
        # Scale the mean of ALL features such that the sum of univariate snrs
        # is size_complementary_patterns * snr_per_feature
        snrs = calc_univ_snrs(
            m0[idx:(idx + size_complementary_patterns)],
            m1[idx:(idx + size_complementary_patterns)], var)
        m1[idx:(idx + size_complementary_patterns)] *= \
            size_complementary_patterns * snr_per_feature * np.sqrt(var) \
            / snrs.sum()
        snrs = calc_univ_snrs(
            m0[idx:(idx + size_complementary_patterns)],
            m1[idx:(idx + size_complementary_patterns)], var)
        print "Complementary patterns snrs:", snrs
        idx += size_complementary_patterns

    # Suppressors patterns
    for i in xrange(n_suppressor_patterns):
        bloc = Cov[idx:(idx + 2),
                   idx:(idx + 2)]
        bloc[::] = cov
        bloc[np.diag_indices(bloc.shape[0])] = var
        m1[idx] += 1
        # Scale the mean of ONE feature such that the sum of univariate snrs
        # is 2 * snr_per_feature
        snrs = calc_univ_snrs(m0[idx:(idx + 2)], m1[idx:(idx + 2)], var)
        m1[idx] *= 2 * snr_per_feature * np.sqrt(var) / snrs.sum()
        snrs = calc_univ_snrs(m0[idx:(idx + 2)], m1[idx:(idx + 2)], var)
        print "Suppressors patterns snrs:", snrs
        idx += 2

    # Redundant patterns
    for i in xrange(n_redundant_patterns):
        bloc = Cov[idx:(idx + size_redundant_patterns),
                   idx:(idx + size_redundant_patterns)]
        bloc[::] = cov
        bloc[np.diag_indices(bloc.shape[0])] = var
        la, v = scipy.linalg.eig(bloc)
        #la = np.real(la)
        eig_vec_max = v[:, np.argmax(np.abs(la))].ravel()
        m1[idx:(idx + size_redundant_patterns)] += eig_vec_max
        # Scale the mean of ALL features such that the sum of univariate snrs
        # is size_complementary_patterns * snr_per_feature
        snrs = calc_univ_snrs(
            m0[idx:(idx + size_redundant_patterns)],
            m1[idx:(idx + size_redundant_patterns)], var)
        m1[idx:(idx + size_redundant_patterns)] *= \
            size_redundant_patterns * snr_per_feature * np.sqrt(var) \
            / snrs.sum()
        snrs = calc_univ_snrs(
            m0[idx:(idx + size_redundant_patterns)],
            m1[idx:(idx + size_redundant_patterns)], var)
        print "Redundant patterns snrs:", snrs
        #dmeans = v[:, np.argmax(np.abs(la))].ravel() * np.sqrt(max_var) * \
        #    snr_per_feature
        #m1[idx:(idx + size_redundant_patterns)] += dmeans
        idx += size_redundant_patterns

    # Independant features
    for i in xrange(n_independant_features):
        Cov[idx, idx] = var
        m1[idx] = snr_per_feature * np.sqrt(var)
        idx += 1
    # Nomalize m1 to scale to the desire global snr
    snr_actual = np.sqrt(np.dot(np.dot(scipy.linalg.inv(Cov), m1 - m0),
                                m1 - m0))
    print "Global snr before scaling:", snr_actual
    m1 *= snr / snr_actual

    n_g0 = int(np.round(n_samples * grp_proportion))

    # Sample according to means and Cov
    np.random.seed(random_seed)
    X0 = np.random.multivariate_normal(m0, Cov, n_g0)
    X1 = np.random.multivariate_normal(m1, Cov, n_samples - n_g0)
    X = np.vstack([X0, X1])
    y = np.array([0] * n_g0 + [1] * (n_samples - n_g0))

    weigths = np.dot((m1 - m0), scipy.linalg.inv(Cov))

    # Add noize
    if n_noize is None:
        n_noize = n_informative_features * 10

    X = np.hstack([X, 
        np.random.normal(0, 1, n_samples*n_noize).reshape(n_samples, n_noize)])
    weigths = np.concatenate((weigths, [0] * n_noize))
    
    if full_info:
        return X, y, weigths, m0, m1, Cov
    else:
        return X, y
    
if __name__ == '__main__':
    X, y, weigths, m0, m1, Cov = make_classification(n_samples=500,
                        n_complementary_patterns=1,
                        size_complementary_patterns=2,
                        n_suppressor_patterns=1,
                        n_redundant_patterns=1,
                        size_redundant_patterns=2, n_independant_features=2,
                        snr=3., grp_proportion=.5,
                        n_noize=None, full_info=True)

    def plot_2d(plot, X, y, xlab, ylab, title):
        m0_hat = X[y == 0, :].mean(axis=0)
        m1_hat = X[y == 1, :].mean(axis=0)
        Xc = X.copy()
        Xc[y == 0, :] -= m0_hat
        Xc[y == 1, :] -= m1_hat
        cov_hat = np.cov(Xc.T)
        weigths_hat = np.dot((m1_hat - m0_hat), scipy.linalg.inv(cov_hat))
        snr_glob = np.sqrt(np.dot(np.dot(scipy.linalg.inv(cov_hat),
                                         m1_hat - m0_hat), m1_hat - m0_hat))
        snr_glob = round(snr_glob, 2)
        snrs = np.round(np.sqrt((m1_hat - m0_hat) ** 2 / cov_hat.diagonal()), 2)
        # plot x, y
        plt.plot(X[y == 0, 0], X[y == 0, 1], 'or', X[y == 1, 0], X[y == 1, 1],
                 'ob')
        # plot means
        plt.plot(m0_hat[0], m0_hat[1], 'ok', m1_hat[0], m1_hat[1], 'ok',
                 markersize=10)
        # plot Cov
        la, v = scipy.linalg.eig(cov_hat)
        std_dev = np.sqrt(np.real(la))
        e0 = patches.Ellipse(m0_hat, 2 * std_dev[0], 2 * std_dev[1],
                 angle=np.arctan(v[1, 0] / v[0, 0]) * 180. / np.pi,
                 linewidth=3, fill=False, color='k', linestyle='dashed',
                 zorder=10)
        plot.add_patch(e0)
        e1 = patches.Ellipse(m1_hat, 2 * std_dev[0], 2 * std_dev[1],
                 angle=np.arctan(v[1, 0] / v[0, 0]) * 180. / np.pi,
                 linewidth=3, fill=False, color='k', linestyle='dashed',
                 zorder=10)
        plot.add_patch(e1)
        # plot weights
        plot.arrow(m0_hat[0], m0_hat[1], weigths_hat[0], weigths_hat[1],
                   head_width=.3, fc="y", ec="y", head_length=.3, linewidth=3,
                   zorder=10)
        plt.xlabel(xlab + ': snr=%.2f' % snrs[0])
        plt.ylabel(ylab + ': snr=%.2f' % snrs[1])
        plt.title(title + ': snr=%.2f' % snr_glob)

    #plt.show()

    import matplotlib.pyplot as plt
    from matplotlib import patches

    proj = np.dot(X, weigths)

    plt.figure(figsize=(10, 10))
    idx = 0
    plot = plt.subplot(221)
    plot_2d(plot=plot, X=X[:, idx:(idx + 2)], y=y,
        xlab="X" + str(idx), ylab="X" + str(idx + 1),
        title='Complementary pattern')
    idx += 2
    plot = plt.subplot(222)
    plot_2d(plot=plot, X=X[:, idx:(idx + 2)], y=y,
        xlab="X" + str(idx), ylab="X" + str(idx + 1),
        title='Suppressor variables')
    idx += 2
    plot = plt.subplot(223)
    plot_2d(plot=plot, X=X[:, idx:(idx + 2)], y=y,
        xlab="X" + str(idx), ylab="X" + str(idx + 1),
        title='Redundant pattern')
    idx += 2
    plot = plt.subplot(224)
    plot_2d(plot=plot, X=X[:, idx:(idx + 2)], y=y,
        xlab="X" + str(idx), ylab="X" + str(idx + 1),
        title='Independant features')
    plt.show()