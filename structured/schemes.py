# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 08:58:10 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: TBD
"""

__all__ = ['WeightingScheme', 'Horst', 'Centroid', 'Factorial']

import abc
from utils import sign, cov


class WeightingScheme(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        super(WeightingScheme, self).__init__()

    @abc.abstractmethod
    def compute(self, ti=None, tj=None):
        raise NotImplementedError('Abstract method "compute" must be '\
                                  'specialised!')


class Horst(WeightingScheme):
    def __init__(self):
        super(Horst, self).__init__()

    def compute(self, *args):
        return 1


class Centroid(WeightingScheme):
    def __init__(self):
        super(Centroid, self).__init__()

    def compute(self, ti, tj):
        return sign(cov(ti, tj))


class Factorial(WeightingScheme):
    def __init__(self):
        super(Factorial, self).__init__()

    def compute(self, ti, tj):
        return cov(ti, tj)
