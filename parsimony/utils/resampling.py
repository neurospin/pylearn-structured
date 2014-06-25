# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 10:00:06 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""

__all__ = ["k_fold"]


def k_fold(n, K=7):

    all_ids = set(range(n))
    for k in xrange(K):
        test = range(k, n, K)
        train = all_ids.difference(test)

        yield list(train), test