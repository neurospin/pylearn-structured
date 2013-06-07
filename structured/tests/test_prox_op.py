# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 13:46:35 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD Style
"""

import numpy as np
from multiblock.utils import *

def test_prox_op():
    import prox_op

    a = np.random.rand(100,1)
    a /= norm(a)
    op = prox_op.L1(0.1) # Remove variables smaller than 0.1
    b = op.prox(a, 0)
    print "Norm2:",norm(b)
    print "Norm1:",norm1(b)
    print "Norm0:",norm0(b)

    op = prox_op.L1_binsearch(0.7) # Enforces |x|_1 <= s = 1, by finding lambda
    b = op.prox(a, 0)
    print "Norm2:",norm(b)
    print "Norm1:",norm1(b)
    print "Norm0:",norm0(b)

    op = prox_op.L0_binsearch(40) # Keep 10 variables, by finding a lambda
    b = op.prox(a, 0)
    print "Norm2:",norm(b)
    print "Norm1:",norm1(b)
    print "Norm0:",norm0(b)

    op = prox_op.L0_by_count(40) # Keep the 10 absolute largest variables
    b = op.prox(a, 0)
    print "Norm2:",norm(b)
    print "Norm1:",norm1(b)
    print "Norm0:",norm0(b)

if __name__ == "__main__":
    test_prox_op()