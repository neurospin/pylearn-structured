ParsimonY: structured and sparse machine learning in Python
===========================================================

ParsimonY contains the following features:
* `parsimony` provides parsimony and sparse machine learning. It currently contains:
    * Loss functions:
        * OLS
    * Penalties:
        * L1 (Lasso)
        * L2 (Ridge)
        * L1+L2 (Elastic net)
        * Total variation (TV)
        * Any combination of the above
    * Algorithms:
        * _F_ast _I_terative _S_oft-_T_hresholding _A_lgorithm (FISTA)
        * _CO_ntinuation of _NEST_sterov’s smoothing _A_lgorithm (CONESTA)
        * Excessive gap method


Dependencies
------------
* Python 2.7.x
* NumPy >= 1.6.1
* Scipy >= 0.9.0