# -*- coding: utf-8 -*-
"""
Created on Sun Jun 16 09:13:14 2013

@author: edouard.duchesnay@cea.fr
"""

import numpy as np
import mulm
n_samples = 10
X = np.random.randn(n_samples, 5)
X[:, -1] = 1  # Add intercept
Y = np.random.randn(n_samples, 4)
betas = np.array([1, 2, 2, 0, 3])
Y[:, 0] += np.dot(X, betas)
Y[:, 1] += np.dot(X, betas)

betas, ss_errors = mulm.ols(X, Y)
p, t = mulm.ols_stats_tcon(X, betas, ss_errors, contrast=[1, 0, 0, 0, 0], pval=True)
p, f = mulm.ols_stats_fcon(X, betas, ss_errors, contrast=[1, 0, 0, 0, 0], pval=True)

