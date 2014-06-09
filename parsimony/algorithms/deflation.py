# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.algorithms.deflation` module contains deflation procedures.
functions.

Created on Fri Mar 21 15:18:56 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import abc

import numpy as np

__all__ = ["ProjectionDeflation"]


class Deflation(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def deflate(self, X, w):
        raise NotImplementedError('Abstract method "deflate" must be ' \
                                  'specialised!')


class ProjectionDeflation(Deflation):

    def deflate(self, X, w):
        return X - np.dot(np.dot(X, w), w.T) / np.dot(w.T, w)