# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:34:45 2013

@author: edouard.duchesnay@cea.fr
"""
#import unittest

import numpy as np
from structured import datasets
from numpy.testing import assert_array_almost_equal
from time import time
import structured.utils as utils


def test():
    ## Test Logistic regression
    X, y, weigths, m0, m1, Cov = datasets.make_classification(n_samples=500,
                            n_complementary_patterns=1,
                            size_complementary_patterns=2,
                            n_suppressor_patterns=1,
                            n_redundant_patterns=1,
                            size_redundant_patterns=2, n_independant_features=2,
                            snr=3., grp_proportion=.5,
                            n_noize=10, full_info=True, random_seed=0)
    # center X
    Xc = X - X.mean(axis=0)
    from structured import models
    lr = models.LogisticRegression()
    lr.fit(Xc, y)
    # beta_r as been obtain wit R logistic regression on centered data
    # without intercept
    # modc = glm(y~.-1, data=data.centered, family=binomial("logit"))
    beta_r = np.asarray([
    -1.830376046 ,2.136440846 ,2.310228451, -1.418557866 ,0.589448660, 0.397224097, 
     0.548192201 ,0.475168811 ,0.401900579, -0.217723267 ,0.006007388, -0.113258946, 
    -0.002004698 ,0.172740504 ,-0.122215721, -0.215561789 ,0.105610917, -0.117197163]) 
    assert_array_almost_equal(lr.beta.ravel(), beta_r)

if __name__ == "__main__":
    t = time()
    test()
    utils.debug("test_structure took %.2f seconds" % (time() - t))