# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 17:37:17 2013

@author:  Edouard Duchesnay
@email:   duchesnay@gmail.com
@license: TBD
"""
import numpy as np


def check_arrays(*arrays):
    """Check that:
        - List are converted to array.
        - All arrays are re-casted into float.
        - All arrays have consistent first dimensions.
        - Array are at least 2-d array, if not reshape them.

    Parameters
    ----------
    *arrays : sequence of arrays or scipy.sparse matrices with same shape[0]
        Python lists or tuples occurring in arrays are converted to 2D numpy
        arrays.

    Example
    -------
    >>> import numpy as np
    >>> check_arrays([1, 2], np.array([3, 4]), np.array([[1., 2.], [3., 4.]]))
    [array([[ 1.],
       [ 2.]]), array([[ 3.],
       [ 4.]]), array([[ 1.,  2.],
       [ 3.,  4.]])]
       """
    if len(arrays) == 0:
        return None
    n_samples = None
    checked_arrays = []
    for array in arrays:
        # Recast a input as float array
        array = np.asarray(array, dtype=np.float)
        if n_samples is None:
            n_samples = array.shape[0]
        if array.shape[0] != n_samples:
            raise ValueError("Found array with dim %d. Expected %d"
                             % (array.shape[0], n_samples))
        if len(array.shape) == 1:
            array = array[:, np.newaxis]
        checked_arrays.append(array)
    return checked_arrays