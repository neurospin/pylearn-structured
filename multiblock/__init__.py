# -*- coding: utf-8 -*-
"""
The :mod:`multiblock` module includes several different projection based latent
variable methods for one, two or more blocks of data.

@author: Tommy Löfstedt <tommy.loefstedt@cea.fr>
"""

from .methods import PCA
from .methods import SVD
from .methods import EIGSym
from .methods import PLSR
from .methods import PLSC
from .methods import O2PLS
from .methods import center
from .methods import scale
from .methods import direct

import algorithms
import preprocess
import prox_op
import tests
import utils

__all__ = ['PCA', 'SVD', 'EIGSym', 'PLSR', 'PLSC', 'O2PLS',
           'center', 'scale', 'direct', 'prox_op', 'algorithms', 'preprocess',
           'tests', 'utils']