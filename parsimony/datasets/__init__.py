# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:31:13 2013

@author:  Tommy Löfstedt and Edouard Duchesnay
@email:   tommy.loefstedt@cea.fr, edouard.duchesnay@cea.fr
@license: TBD
"""

import Russett
import simulate
import regression
import classification

#from .samples_generator_nostruct import make_classification
#from .samples_generator_struct import make_regression_struct
#from .samples_generator_struct import make_classification_struct

__all__ = ['Russett', 'simulate',
           'regression', 'classification']