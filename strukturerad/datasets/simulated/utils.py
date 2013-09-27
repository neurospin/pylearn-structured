# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 10:50:17 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: TBD.
"""

import numpy as np

__all__ = ['TOLERANCE', 'RandomUniform', 'U', 'norm2']

TOLERANCE = 5e-8


class RandomUniform(object):

    def __init__(self, a=0, b=1):

        self.a = float(a)
        self.b = float(b)

    def rand(self, *d):

        R = np.random.rand(*d)
        R = R * (self.b - self.a) + self.a

        return R


def U(a, b):

    t = max(a, b)
    a = float(min(a, b))
    b = float(t)
    return (np.random.rand() * (b - a)) + a


def norm2(x):

    return np.sqrt(np.sum(x ** 2.0))