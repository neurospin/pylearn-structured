# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 14:54:28 2014

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
from . import interfaces
from . import losses
from . import penalties

from .combinedfunctions import CombinedFunction
from .combinedfunctions import LinearRegressionL1L2TV
from .combinedfunctions import LinearRegressionL1L2GL
from .combinedfunctions import LogisticRegressionL1L2TV
from .combinedfunctions import LogisticRegressionL1L2GL
from .combinedfunctions import RR_SmoothedL1TV
from .combinedfunctions import PCA_L1_TV

__all__ = ["interfaces", "losses", "penalties",

           "CombinedFunction",
           "LinearRegressionL1L2TV", "LinearRegressionL1L2GL",
           "LogisticRegressionL1L2TV", "LogisticRegressionL1L2GL",
           "RR_SmoothedL1TV",
           "PCA_L1_TV"]