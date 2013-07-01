# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 10:33:37 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: TBD
"""

__all__ = ['Mode', 'A', 'NewA', 'B']

import abc
import numpy as np
from numpy.linalg import pinv
import normalisation
from utils import TOLERANCE


class Mode(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(Mode, self).__init__()

    @abc.abstractmethod
    def estimation(self, X, u):
        raise NotImplementedError('Abstract method "compute" must be '\
                                  'specialised!')

    def normalise(self, w, X=None):
        return self.norm.apply(w, X)


class A(Mode):

    def __init__(self):
#        Mode.__init__(self)
        super(A, self).__init__()

        self.norm = normalisation.UnitVarianceScores()

    def estimation(self, X, u):
        ssu = np.sum(u ** 2.0)  # np.dot(u.T, u)
        if ssu < TOLERANCE:
            return np.dot(X.T, u)
        else:
            # TODO: Ok with division here?
            return np.dot(X.T, u) / np.dot(u.T, u)


class NewA(A):

    def __init__(self):
#        Mode.__init__(self)
        super(NewA, self).__init__()

        self.norm = normalisation.UnitNormWeights()


class B(Mode):

    def __init__(self):
#        Mode.__init__(self)
        super(B, self).__init__()

        self.norm = normalisation.UnitVarianceScores()

    def estimation(self, X, u):
        return np.dot(pinv(X), u)
