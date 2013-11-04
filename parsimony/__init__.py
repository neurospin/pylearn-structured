# -*- coding: utf-8 -*-
"""
The :mod:`parsimony` module includes several different parsimony machine
learning models for one, two or more blocks of data.

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: TBD.
"""

import algorithms
import datasets
import estimators
import functions
import utils
import start_vectors

__version__ = '0.1.0'

__all__ = ['algorithms', 'datasets', 'estimators', 'functions', 'utils',
           'start_vectors']