# -*- coding: utf-8 -*-
"""
Created on Thu Feb 8 09:22:00 2013

@author:  Tommy Löfstedt and Edouard Duchesnay
@email:   lofstedt.tommy@gmail.com, edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
import maths
import consts

from .utils import time_cpu, time_wall, deprecated, approx_grad
from .utils import AnonymousClass, optimal_shrinkage
from .plot import plot_map2d

__all__ = ["maths", "consts",
           "time_cpu", "time_wall", "deprecated", "approx_grad",
           "AnonymousClass", "optimal_shrinkage", "plot_map2d"]