# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 15:19:17 2013

@author: edouard.duchesnay@cea.fr
"""


class LinearRegressionL1L2TV:
    def __init__(self, k, l, g):
        self.k, self.l, self.g = k, l, g
        self.function_pgm = functions.OLSL2_L1_TV(k, l, g, shape)

    def fit(self, X, y):
        beta_dynamic, f_dynamic, t_dynamic, mu_dynamic, G_conesta \
            = algorithms.CONESTA(X, y, self.function_pgm,
                         betastart,
                         mu_start=mu_egm[0],
                         mumin=mu_zero,
                         sigma=2.0,
                         tau=0.5,
                         dynamic=True,
                         eps=eps,
                         conts=conts,
                         max_iter=maxit,
                         init_iter=init_iter)