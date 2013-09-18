# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 09:26:14 2013

@author: Tommy Löfstedt
@email: tommy.loefstedt@cea.fr
"""

import regression
import correlation_matrices

import lasso
import ridge
import l1_tv
#import l2_2D
#import l1_l2_tv
#import l1_l2_tv_2D

__all__ = ['regression', 'correlation_matrices',
           'lasso', 'ridge', 'l1_tv']