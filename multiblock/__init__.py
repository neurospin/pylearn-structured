# -*- coding: utf-8 -*-
"""
The :mod:`multiblock` module includes several different projection based latent
variable methods for one, two or more blocks of data.

@author: Tommy Löfstedt <tommy.loefstedt@cea.fr>
"""

from .methods import PCA
from .methods import SVD
from .methods import PLSR
from .methods import PLSC
from .methods import O2PLS
from .methods import RGCCA

import algorithms
import preprocess
import prox_ops
import tests
import utils

__all__ = ['PCA', 'SVD', 'PLSR', 'PLSC', 'O2PLS', 'RGCCA',
           'prox_ops', 'algorithms', 'preprocess',
           'tests', 'utils']