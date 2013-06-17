# -*- coding: utf-8 -*-
"""
The :mod:`structured` module includes several different structured machine
learning models for one, two or more blocks of data.

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
from .models import Lasso
from .models import RidgeRegression
from .models import ElasticNet
from .models import LinearRegressionTV
from .models import LinearRegressionL1TV
from .models import LinearRegressionElasticNetTV
from .models import LogisticRegression

from .models import EGMRidgeRegression
from .models import EGMLinearRegressionL1L2
from .models import EGMRidgeRegressionTV
from .models import EGMLinearRegressionL1L2TV

import algorithms
import preprocess
import prox_ops
import tests
import utils
import loss_functions
import start_vectors

__version__ = '0.0.9'

__all__ = ['PCA', 'SVD', 'PLSR', 'TuckerFactorAnalysis', 'PLSC', 'O2PLS',
           'RGCCA',

           'LinearRegression', 'Lasso', 'RidgeRegression', 'ElasticNet',
           'LinearRegressionTV', 'LinearRegressionL1TV',
           'LinearRegressionElasticNetTV', 'LogisticRegression',

           'EGMRidgeRegression', 'EGMLinearRegressionL1L2',
           'EGMRidgeRegressionTV', 'EGMLinearRegressionL1L2TV',

           'prox_ops', 'algorithms', 'preprocess',
           'tests', 'utils', 'loss_functions', 'start_vectors']