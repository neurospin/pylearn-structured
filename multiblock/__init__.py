# -*- coding: utf-8 -*-
"""
The :mod:`multiblock` module includes several different projection based latent
variable models for one, two or more blocks of data.

@author: Tommy Löfstedt <tommy.loefstedt@cea.fr>
"""

from .models import PCA
from .models import SVD
from .models import PLSR
from .models import TuckerFactorAnalysis
from .models import PLSC
from .models import O2PLS
from .models import RGCCA
from .models import LinearRegression
from .models import LinearRegressionTV
from .models import LogisticRegression
from .models import RidgeRegressionTV

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
