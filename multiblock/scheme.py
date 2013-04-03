# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 08:58:10 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD Style
"""

__all__ = ['WeightingScheme', 'Horst', 'Centroid', 'Factorial']

import abc
from multiblock.utils import sign, corr


class WeightingScheme(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def compute(self, ti=None, tj=None):
        raise NotImplementedError('Abstract method "compute" must be '\
                                  'specialised!')


class Horst(WeightingScheme):
    def compute(self, *args):
        return 1


class Centroid(WeightingScheme):
    def compute(self, ti, tj):
        return sign(corr(ti, tj))


class Factorial(WeightingScheme):
    def compute(self, ti, tj):
        return corr(ti, tj)