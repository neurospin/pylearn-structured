# -*- coding: utf-8 -*-
"""
The :mod:`strukturerad` module includes several different parsimony machine
learning models.

@author: Tommy Löfstedt
@email:  tommy.loefstedt@cea.fr
"""

# Directories
import datasets
import tests
import utils

# Modules
import algorithms
import loss_functions
import models
import start_vectors

__version__ = '0.0.1'

           # Directories
__all__ = ['datasets', 'tests', 'utils',
           # Modules in this directory
           'algorithms', 'loss_functions', 'models', 'start_vectors']