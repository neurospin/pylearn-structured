# -*- coding: utf-8 -*-
"""
The :mod:`parsimony` module includes several different parsimony machine
learning models for one, two or more blocks of data.

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD 3-clause.
"""

import parsimony.algorithms as algorithms
import parsimony.estimators as estimators
import parsimony.start_vectors as start_vectors

__version__ = "0.1.2"

__all__ = ["algorithms", "estimators", "start_vectors"]