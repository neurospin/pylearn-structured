Structured and sparse machine learning in python
================================================

Structured sparse learning for Python 2.7.
* `structured` provides structured and sparse machine learning. It currently contains:
    * Loss functions:
        * OLS
    * Penalties:
        * L1 (Lasso)
        * L2 (Ridge)
        * L1+L2 (Elastic net)
        * Total variation (TV)
        * Any combination of the above
    * Algorithms:
        * Fast iterative soft-thresholding algorithm (FISTA)
        * __CO__ntinuation of __NEST__sterov’s smoothing __A__lgorithm (CONESTA)
        * Excessive gap method


Dependencies
------------
* Python 2.7.x
* NumPy >= 1.6.1
* Scipy >= 0.9.0
