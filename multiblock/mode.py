# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 10:33:37 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD Style
"""

__all__ = ['Mode', 'A', 'NewA', 'B']

import abc
import normalisation
from multiblock.utils import dot
from numpy import pinv


class Mode(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def estimation(self, X, u):
        raise NotImplementedError('Abstract method "compute" must be '\
                                  'specialised!')

    def normalise(self, w, X=None):
        return self.normalise.apply(w, X)


class A(Mode):

    def __init__(self):
        Mode.__init__(self)
        self.normalise = normalisation.UnitVarianceScores()

    def estimation(self, X, u):
        # TODO: Ok with division here?
        return dot(X.T, u) / dot(u.T, u)


class NewA(A):

    def __init__(self):
        Mode.__init__(self)
        self.normalise = normalisation.UnitNormWeights()


class B(Mode):

    def __init__(self):
        Mode.__init__(self)
        self.normalise = normalisation.UnitVarianceScores()

    def estimation(self, X, u):
        return dot(pinv(X), u)