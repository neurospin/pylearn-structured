.. _tutorials:


General framework
=================

Three principle modules have been defined: **algorithms**, **estimators**, and **functions**.

* **functions** define the loss function OLS (ordinary least squares) with different penalties 
  (for instance, L1 Lasso, L2 Ridge Regression, etc.). We need to minimize this functions.
* **algorithms** define algorithms (Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) [FISTA2009]_,
  COntinuation of NESTerov's smoothing Algorithm (conesta), Excessive gap method)
  to minimize the above function.
* **estimators** define the combination of function and algorithm.

A general framework for our package:

* Functions

  * Loss function

    * OLS

  * Penalties

    * L1 (Lasso)
    * L2 (Ridge)
    * L1 + L2 (Elastic net)
    * Total variation (TV)

  * Any combination of the above

* Algorithms

  * Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)
  * COntinuation of NESTerov's smoothing Algorithm (conesta)
  * Excessive gap method

* Estimators

  * RidgeRegression_L1_TV
  * RidgeRegression_SmoothedL1TV

Build Simulated Dataset  
=======================

We build a simple simulated dataset as :math:`y = X \beta + noise`:

.. code-block:: python

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
    
In later sessions, we want to discover :math:`\beta` using different loss functions.

OLS + L1 + L2 + TV Functions
============================

Knowing :math:`X` and :math:`y`, we want to discover :math:`\beta` by minimizing only OLS loss function by FISTA algorithm:

.. math::

   min ||y - X\beta||^2_2


.. code-block:: python

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


By adding l2 ridge regression coefficient :math:`k=0.1, l=0.0, g=0.0`

.. math::

   min (1/2 ||y - X\beta||2 + k/2 * ||\beta||_2 + l * ||\beta||_1 + g * TV)

where p-norm is defined as

.. math::

   ||V||_p = (\sum\limits_{i=1, v_i \in V}^n |v_i|^p)^{1/p}

.. code-block:: python

    import parsimony.estimators as estimators
    import parsimony.algorithms as algorithms
    import parsimony.tv
    k = 0.1  # l2 ridge regression coefficient
    l = 0.0  # l1 lasso coefficient
    g = 0.0  # tv coefficient
    A, n_compacts = parsimony.tv.A_from_shape(shape)
    ridge_estimator = estimators.RidgeRegression_L1_TV(
                        k, l, g, A,
                        algorithm=algorithms.FISTA(max_iter=1000))
    res = ridge_estimator.fit(X, y)
    print "Estimated beta error =", np.linalg.norm(ridge_estimator.beta - beta)

Similarly, you can add l1 coefficient and TV coefficient :math:`k=0.0, l=0.1, g=0.1`

.. code-block:: python

    import parsimony.estimators as estimators
    import parsimony.algorithms as algorithms
    import parsimony.tv
    k = 0.0  # l2 ridge regression coefficient
    l = 0.1  # l1 lasso coefficient
    g = 0.1  # tv coefficient
    A, n_compacts = parsimony.tv.A_from_shape(shape)
    estimator = estimators.RidgeRegression_L1_TV(
                        k, l, g, A,
                        algorithm=algorithms.FISTA(max_iter=1000))
    res = estimator.fit(X, y)
    print "Estimated beta error =", np.linalg.norm(estimator.beta - beta)


Algorithms
==========

In the previous sections, only the FISTA algorithm (c.f. [FISTA2009]_) has been applied.
In this section, we switch to the other algorithms (CONESTA dynamic and CONESTA static) to minimize the function. 

.. code-block:: python

    Atv, n_compacts = parsimony.tv.A_from_shape(shape)
    tvl1l2_conesta_static = estimators.RidgeRegression_L1_TV(
			    k, l, g, Atv,
			    algorithm=algorithms.CONESTA(dynamic=False))
    res = tvl1l2_conesta_static.fit(X, y)
    print "Estimated beta error =", np.linalg.norm(tvl1l2_conesta_static.beta - beta)
    tvl1l2_conesta_dynamic = estimators.RidgeRegression_L1_TV(
			    k, l, g, Atv,
			    algorithm=algorithms.CONESTA(dynamic=True))
    res = tvl1l2_conesta_dynamic.fit(X, y)
    print "Estimated beta error =", np.linalg.norm(tvl1l2_conesta_dynamic.beta - beta)

Excessive gap method
--------------------

Excessive gap method algorithm (ExcessiveGapMethod) only works with the function smoothed l1 "RidgeRegression_SmoothedL1TV".
:math:`k`, :math:`l`, and :math:`g` should be large than zero otherwise you will get Nan :math:`\beta`.


.. code-block:: python

    import scipy.sparse as sparse
    Atv, n_compacts = parsimony.tv.A_from_shape(shape)
    Al1 = sparse.eye(num_ft, num_ft)
    k = 0.05  # ridge regression coefficient
    l = 0.05  # l1 coefficient
    g = 0.05  # tv coefficient
    rr_smoothed_l1_tv = estimators.RidgeRegression_SmoothedL1TV(
		k, l, g,
		Atv=Atv, Al1=Al1,
		algorithm=algorithms.ExcessiveGapMethod(max_iter=1000))
    res = rr_smoothed_l1_tv.fit(X, y)
    print "Estimated beta error =", np.linalg.norm(tvl1l2_conesta_dynamic.beta - beta)


References
==========
.. [FISTA2009] Amir Beck and Marc Teboulle, A Fast Iterative Shrinkage-Thresholding Algorithm for Linear Inverse Problems, SIAM Journal on Imaging Sciences, 2009
.. [NESTA2011] Stephen Becker, Jerome Bobin, and Emmanuel J. Candes, NESTA: A Fast and Accurate First-Order Method for Sparse Recovery, SIAM Journal on Imaging Sciences, 2011