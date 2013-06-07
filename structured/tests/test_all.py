# -*- coding: utf-8 -*-
"""
Created on Thu May 23 15:08:31 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD Style
"""

import test_SVD_PCA
import test_predictions
import test_o2pls
import test_regularisation
import test_multiblock


def test():
    test_SVD_PCA.test()
    test_predictions.test()
    test_o2pls.test()
    test_regularisation.test()
    test_multiblock.test()


if __name__ == "__main__":

    test()