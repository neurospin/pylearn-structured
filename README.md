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
        * _F_ast _I_terative _S_oft-_T_hresholding _A_lgorithm (fista)
        * _CO_ntinuation of _NEST_sterov’s smoothing _A_lgorithm (conesta)
        * Excessive gap method


Dependencies
------------
* Python 2.7.x
* NumPy >= 1.6.1
* Scipy >= 0.9.0


[Documentation](http://neurospin.github.io/pylearn-parsimony/)
