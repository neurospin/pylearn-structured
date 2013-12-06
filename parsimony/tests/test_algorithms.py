# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:42:07 2013

@author: jinpeng.li@cea.fr
"""
import unittest


class TestFISTA(unittest.TestCase):
    def test_fista(self):
        import numpy as np
        import scipy.sparse as sparse
        import parsimony.estimators as estimators
        import parsimony.algorithms as algorithms
        import parsimony.tv
        shape = (4, 4, 1)
        num_samples = 10
        num_ft = shape[0] * shape[1] * shape[2]
        X = np.random.random((num_samples, num_ft))
        y = np.random.randint(0, 2, (num_samples, 1))
        k = 0  # ridge regression coefficient
        l = 0.05  # l1 coefficient
        g = 0  # tv coefficient
        # ================================================================
        # using spams
#        import spams
#        spams_X = np.asfortranarray(X)
#        spams_X = np.asfortranarray(spams_X - np.tile(
#                                    np.mean(spams_X, 0),
#                                   (spams_X.shape[0], 1)))
#        spams_X = spams.normalize(spams_X)
#        spams_Y = np.asfortranarray(y)
#        spams_Y = np.asfortranarray(spams_Y - np.tile(
#                                    np.mean(spams_Y, 0),
#                                    (spams_Y.shape[0], 1)))
#        spams_Y = spams.normalize(spams_Y)
#        W0 = np.zeros((spams_X.shape[1], spams_Y.shape[1]),
#                       dtype=np.float64,
#                       order="FORTRAN")
#        param = {'numThreads': 1, 'verbose': True,
#             'lambda1': l, 'it0': 10, 'max_it': 200,
#             'L0': 0.1, 'tol': 1e-3, 'intercept': False,
#             'pos': False}
#        param['compute_gram'] = True
#        param['loss'] = 'square'
#        param['regul'] = 'l1'
#        (W, optim_info) = spams.fistaFlat(spams_Y,
#                                          spams_X,
#                                          W0,
#                                          True,
#                                          **param)
#        print "spams_X =", repr(spams_X)
#        print "spams_Y =", repr(spams_Y)
#        print "W=", repr(W)
        # ================================================================
        # using pre-computed values
        spams_X = np.asarray([[-0.22669849,  0.2852286 ,  0.77555995, -0.17072452, -0.02503861,
                                 0.34005493, -0.38620549,  0.23769974,  0.48183272,  0.14304892,
                                 0.21320875, -0.41067527,  0.21459235,  0.29682488,  0.43760805,
                                -0.27367393],
                               [-0.16857183,  0.17607336, -0.15863386, -0.22696897, -0.43454013,
                                 0.09335672,  0.37582625,  0.14073385,  0.27006627,  0.31404945,
                                -0.29564747,  0.30579183,  0.16937423, -0.51058077, -0.36565302,
                                -0.42954914],
                               [ 0.24184494, -0.63420934,  0.13826861,  0.1855299 ,  0.17996426,
                                 0.4033453 , -0.6362587 ,  0.50457772, -0.39421451, -0.22488171,
                                 0.1930659 ,  0.37481986, -0.12453901,  0.34413859, -0.34093645,
                                 0.39261188],
                               [-0.37764785, -0.45043382, -0.4080217 , -0.32622956, -0.44993139,
                                -0.23713072, -0.29855795, -0.02502723,  0.31204208,  0.27981462,
                                -0.09355799, -0.315959  ,  0.23007904, -0.01782453,  0.45721548,
                                -0.12758744],
                               [-0.05613541,  0.08139135,  0.02230077, -0.17730469,  0.42012083,
                                -0.66613602,  0.26354841,  0.21183382,  0.33563657, -0.39885472,
                                -0.56007932, -0.25022924,  0.1526051 , -0.3269321 ,  0.06108786,
                                 0.35439118],
                               [ 0.69998675, -0.25802025, -0.36036607,  0.25941764,  0.34339356,
                                -0.11121503,  0.10279268, -0.30845008, -0.41595121,  0.31786093,
                                -0.21528263,  0.1037642 ,  0.30970969,  0.20390179, -0.20826949,
                                 0.10199785],
                               [-0.07001238,  0.04621471,  0.03945222,  0.38993868, -0.01058093,
                                 0.37073276,  0.32250556, -0.57223144, -0.26063618, -0.17099705,
                                 0.22723935,  0.44391192,  0.12804407,  0.24034856,  0.31708701,
                                 0.39284176],
                               [-0.38502068,  0.25817593,  0.15356289, -0.28950307,  0.01038645,
                                 0.10199073,  0.09092231,  0.26396855, -0.21490011,  0.17335655,
                                 0.46401081, -0.1221018 , -0.574307  , -0.47203429, -0.0801845 ,
                                -0.26372429],
                               [ 0.08674756,  0.34197461, -0.02419701,  0.6141446 , -0.39139451,
                                -0.07943399,  0.15219052, -0.34628654, -0.19084391,  0.19483056,
                                 0.34326354,  0.25456178,  0.11025607, -0.07139007, -0.41897953,
                                -0.3894433 ],
                               [ 0.2555074 ,  0.15360485, -0.17792578, -0.25830001,  0.35762048,
                                -0.21556467,  0.01323643, -0.10681839,  0.07696828, -0.62822755,
                                -0.27622093, -0.38388428, -0.61581453,  0.31354792,  0.14102459,
                                 0.24213543]])
        spams_Y = np.asarray([[-0.20701967],
                               [-0.20701967],
                               [-0.20701967],
                               [-0.20701967],
                               [-0.20701967],
                               [ 0.48304589],
                               [ 0.48304589],
                               [ 0.48304589],
                               [-0.20701967],
                               [-0.20701967]])
        W = np.asarray([[ 0.        ],
                       [ 0.        ],
                       [ 0.        ],
                       [-0.10297415],
                       [ 0.34178538],
                       [ 0.11020355],
                       [ 0.43096033],
                       [ 0.        ],
                       [-0.73844159],
                       [ 0.39383968],
                       [ 0.07912836],
                       [ 0.        ],
                       [ 0.        ],
                       [-0.07157305],
                       [ 0.48195605],
                       [ 0.        ]])
        # ================================================================
        # using pre-computed values
        Atv, n_compacts = parsimony.tv.A_from_shape(shape)
        tvl1l2_algorithms = []
        # Al1 = sparse.eye(num_ft, num_ft)
        tvl1l2_fista = estimators.RidgeRegression_L1_TV(
                                k, l, g,
                                Atv,
                                algorithm=algorithms.FISTA())
        tvl1l2_conesta_static = estimators.RidgeRegression_L1_TV(
                                k, l, g,
                                Atv,
                                algorithm=algorithms.CONESTA(dynamic=False))
        tvl1l2_conesta_dynamic = estimators.RidgeRegression_L1_TV(
                                k, l, g,
                                Atv,
                                algorithm=algorithms.CONESTA(dynamic=True))

        tvl1l2_algorithms.append(tvl1l2_fista)
        tvl1l2_algorithms.append(tvl1l2_conesta_static)
        tvl1l2_algorithms.append(tvl1l2_conesta_dynamic)

        for tvl1l2_algorithm in tvl1l2_algorithms:
            print str(tvl1l2_algorithm.algorithm)
            tvl1l2_algorithm.fit(spams_X, spams_Y)
            ## sometimes betas are different
            ## but lead to the same error (err1 and err2)
            # error = np.sum(np.absolute(tvl1l2_algorithm.beta - W))
            # self.assertTrue(error < 0.01)
            err1 = np.sum(np.absolute(
                          np.dot(spams_X, tvl1l2_algorithm.beta) - spams_Y))
            err2 = np.sum(np.absolute(np.dot(spams_X, W) - spams_Y))
            self.assertTrue(np.absolute(err1 - err2) < 0.01,
                            np.absolute(err1 - err2))

if __name__ == "__main__":
    unittest.main()
