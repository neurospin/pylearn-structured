# -*- coding: utf-8 -*-
"""
The :mod:`parsimony` module includes several different parsimony machine
learning models for one, two or more blocks of data.

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD 3-clause.
"""
from . import algorithms
from . import estimators
from . import start_vectors

__version__ = "0.1.6"

__all__ = ["algorithms", "estimators", "start_vectors"]