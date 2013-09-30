# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 10:12:56 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: TBD.
"""

import numpy as np
import utils

__all__ = ['rand']


def rand(shape, density=1.0, rng=utils.RandomUniform(0, 1).rand,
         sort=False):
    """ Generates a random p-by-1 vector.

    shape: A tuple. The shape of the underlying data. E.g., beta may represent
            an underlying 2-by-3-by-4 image, and will in that case be 24-by-1.

    density : A scalar in (0, 1]. The density of the returned regression vector
            (fraction of non-zero elements). Zero-elements will be randomly
            distributed in the vector.

    rng: The random number generator. Must be a function that takes *shape as
            input. Default is utils.RandomUniform in the interval [0, 1).

    sort: A boolean. Whether or not to sort the data. The data sorted along the
            dimensions in order from the first.
    """
    if not isinstance(shape, (list, tuple)):
        shape = (shape,)

    density = max(0.0, min(density, 1.0))

    p = np.prod(shape)
    ps = int(density * p + 0.5)

    beta = rng(p)
    beta[ps:] = 0.0
    beta = np.random.permutation(beta)

    if sort:
        beta = np.reshape(beta, shape)
        for i in xrange(len(shape)):
            beta = np.sort(beta, axis=i)

    return np.reshape(beta, (p, 1))