.. _tutorials:


Building Dataset
================

.. code-block:: python

    >>> import numpy as np
    >>> import parsimony.estimators as estimators
    >>> import parsimony.algorithms as algorithms
    >>> import parsimony.tv
    >>> shape = (4, 4, 1)
    >>> num_samples = 10
    >>> num_ft = shape[0] * shape[1] * shape[2]
    >>> np.random.seed(seed=1)
    >>> X = np.random.random((num_samples, num_ft))
    >>> y = np.random.randint(0, 2, (num_samples, 1))
    >>> k = 0.9  # ridge regression coefficient
    >>> l = 0.1  # l1 coefficient
    >>> g = 1.0  # tv coefficient
    >>> A, n_compacts = parsimony.tv.A_from_shape(shape)
    >>> ridge_l1_tv = estimators.RidgeRegression_L1_TV(k, l, g, A,
    ...                     algorithm=algorithms.StaticCONESTA(max_iter=1000))
    >>> res = ridge_l1_tv.fit(X, y)
    >>> error = np.sum(np.abs(np.dot(X, ridge_l1_tv.beta) - y))
    >>> print "error = ", error
    error =  4.70079220678
    >>> ridge_l1_tv = estimators.RidgeRegression_L1_TV(k, l, g, A,
    ...                     algorithm=algorithms.DynamicCONESTA(max_iter=1000))
    >>> res = ridge_l1_tv.fit(X, y)
    >>> error = np.sum(np.abs(np.dot(X, ridge_l1_tv.beta) - y))
    >>> print "error = ", error
    error =  4.70096488794
    >>> ridge_l1_tv = estimators.RidgeRegression_L1_TV(k, l, g, A,
    ...                     algorithm=algorithms.FISTA(max_iter=1000))
    >>> res = ridge_l1_tv.fit(X, y)
    >>> error = np.sum(np.abs(np.dot(X, ridge_l1_tv.beta) - y))
    >>> print "error = ", error