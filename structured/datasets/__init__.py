# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 17:16:07 2013

@author: edouard.duchesnay@cea.fr
"""

from .samples_generator_nostruct import make_classification
from .samples_generator_struct import make_regression_struct


__all__ = ['make_classification' 'make_regression_struct']