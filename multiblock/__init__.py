# -*- coding: utf-8 -*-
"""
The :mod:`multiblock` module includes several different projection based latent
variable methods for one, two or more blocks of data.

@author: Tommy Löfstedt <tommy.loefstedt@cea.fr>
"""

from .methods import PCA
from .methods import SVD
from .methods import PLSR
from .methods import TuckerFactorAnalysis
from .methods import PLSC
from .methods import O2PLS
from .methods import RGCCA
from .methods import LinearRegression
from .methods import LinearRegressionTV
from .methods import LogisticRegression
from .methods import RidgeRegressionTV

import algorithms
import preprocess
import prox_ops
import tests
import utils
import loss_functions
import start_vectors

__all__ = ['PCA', 'SVD', 'PLSR', 'TuckerFactorAnalysis', 'PLSC', 'O2PLS',
           'RGCCA',
           'LinearRegression', 'LinearRegressionTV', 'RidgeRegressionTV',
           'LogisticRegression',
           'prox_ops', 'algorithms', 'preprocess',
           'tests', 'utils', 'loss_functions', 'start_vectors']