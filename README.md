ParsimonY: structured and sparse machine learning in Python
===========================================================

ParsimonY contains the following features:
* `parsimony` provides structured and sparse penalties in machine learning. It currently contains:
    * Loss functions:
        * OLS
    * Penalties:
        * L1 (Lasso)
        * L2 (Ridge)
        * L1+L2 (Elastic net)
        * Total variation (TV)
        * Any combination of the above
    * Algorithms:
        * _F_ast _I_terative _S_hrinkage-_T_hresholding _A_lgorithm (fista)
        * _CO_ntinuation of _NEST_sterov’s smoothing _A_lgorithm (conesta)
        * Excessive gap method
    * Estimators
        * RidgeRegression_L1_TV
        * RidgeRegression_SmoothedL1TV

Quick start
-----------

To quick start an algorithm, we first build a simulated dataset `X` and `y`.

\( \underline{x}_{1} x_{2} )\

```
    import numpy as np
    np.random.seed(seed=1)
    # Three-dimension matrix is defined as:
    shape = (4, 4, 1)
    # The number of samples is defined as:
    num_samples = 10
    # The number of features per sample is defined as:
    num_ft = shape[0] * shape[1] * shape[2]
    # Define X randomly as simulated data
    X_raw = np.random.random((num_samples, shape[0], shape[1], shape[2]))
    X = np.reshape(X_raw, (num_samples, num_ft))
    # Define beta randomly
    beta = np.random.random((num_ft, 1))
    # Define y by adding noise
    y = np.dot(X, beta) + 0.001 * np.random.random((num_samples, 1))
```


```
    import parsimony.estimators as estimators
    import parsimony.algorithms as algorithms
    import parsimony.tv
    k = 0.0  # l2 ridge regression coefficient
    l = 0.0  # l1 lasso coefficient
    g = 0.0  # tv coefficient
    A, n_compacts = parsimony.tv.A_from_shape(shape)  # Memory allocation for TV
    ols_estimator = estimators.RidgeRegression_L1_TV(
                        k, l, g, A,
                        algorithm=algorithms.FISTA(max_iter=1000))
    res = ols_estimator.fit(X, y)
    print "Estimated beta error =", np.linalg.norm(ols_estimator.beta - beta)
```

Dependencies
------------

* Python 2.7.x
* NumPy >= 1.6.1
* Scipy >= 0.9.0


Important links
----------------

* [Tutorials](http://neurospin.github.io/pylearn-parsimony/tutorials.html)

* [Documentation](http://neurospin.github.io/pylearn-parsimony/)
