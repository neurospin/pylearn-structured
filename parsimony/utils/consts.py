# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 11:26:45 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: TBD.
"""
import numpy as np

__all__ = ["TOLERANCE", "MAX_ITER", "FLOAT_EPSILON"]

# Settings
TOLERANCE = 5e-8
# TODO: MAX_ITER is heavily algorithm-dependent, so we have to think about if
# we should include a package-wide maximum at all.
MAX_ITER = 10000
#mu_zero = 5e-8

FLOAT_EPSILON = np.finfo(float).eps