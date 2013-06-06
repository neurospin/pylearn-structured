# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 17:39:51 2013

@author: edouard.duchesnay@cea.fr
@author: fouad.hadjselem@cea.fr
@author: vincent.guillemot@cea.fr
@author: lofstedt.tommy@gmail.com
@author: vincent.frouin@cea.fr
"""
from abc import abstractmethod


class Optimizer:
    """Optimizer abstract class"""

    @abstractmethod
    def optimize(self, X, y, *args, **kwargs):
        """Do the optimization."""