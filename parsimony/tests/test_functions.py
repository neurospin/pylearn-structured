# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:42:07 2013

@author: jinpeng
"""
import unittest


class TestFunctions(unittest.TestCase):
#    def test_ridge_l1(self):
#        import numpy as np
#        import parsimony.estimators as estimators
#        import parsimony.algorithms as algorithms
#        import parsimony.tv
#        shape = (4, 4, 1)
#        num_samples = 10
#        num_ft = shape[0] * shape[1] * shape[2]
#        X = np.random.random((num_samples, num_ft))
#        y = np.random.randint(0, 2, (num_samples, 1))
#        k = 0.05  # ridge regression coefficient
#        l = 0  # l1 coefficient
#        g = 0  # tv coefficient
#        # ================================================================
#        # using spams
##        import spams
##        spams_X = np.asfortranarray(X)
##        spams_X = np.asfortranarray(spams_X - np.tile(
##                                    np.mean(spams_X, 0),
##                                   (spams_X.shape[0], 1)))
##        spams_X = spams.normalize(spams_X)
##        spams_Y = np.asfortranarray(y)
##        spams_Y = np.asfortranarray(spams_Y - np.tile(
##                                    np.mean(spams_Y, 0),
##                                    (spams_Y.shape[0], 1)))
##        spams_Y = spams.normalize(spams_Y)
##        W0 = np.zeros((spams_X.shape[1], spams_Y.shape[1]),
##                       dtype=np.float64,
##                       order="FORTRAN")
##        param = {'numThreads': 1, 'verbose': True,
##             'lambda1': k, 'it0': 10, 'max_it': 200,
##             'L0': 0.1, 'tol': 1e-3, 'intercept': False,
##             'pos': False}
##        param['compute_gram'] = True
##        param['loss'] = 'square'
##        param['regul'] = 'l2'
##        (W_ridge, optim_info) = spams.fistaFlat(spams_Y,
##                                              spams_X,
##                                              W0,
##                                              True,
##                                              **param)
##        param['regul'] = 'l1'
##        (W_l1, optim_info) = spams.fistaFlat(spams_Y,
##                                             spams_X,
##                                             W0,
##                                             True,
##                                             **param)
##        print "spams_X =", repr(spams_X)
##        print "spams_Y =", repr(spams_Y)
##        print "W_ridge =", repr(W_ridge)
##        print "W_l1 =", repr(W_l1)
#        # ================================================================
#        # using pre-computed values
#        spams_X = np.asarray([[-0.62020013,  0.39809644, -0.4354417 ,  0.1276226 ,  0.25042866,
#                -0.37853835,  0.2049136 ,  0.26038437, -0.26147759, -0.3468884 ,
#                -0.31256216, -0.33767948,  0.18188244, -0.46330705, -0.61563846,
#                 0.15202283],
#               [-0.17312953,  0.09848125,  0.52135967, -0.48213548,  0.43368937,
#                 0.19575179,  0.03878352, -0.39845962, -0.53109945,  0.6372815 ,
#                -0.46497223,  0.22836529,  0.25785729,  0.00636549, -0.01793828,
#                -0.55756495],
#               [-0.02609454, -0.19485542, -0.08859252,  0.35688521,  0.21252098,
#                -0.412504  ,  0.18681481,  0.02197316, -0.13867063, -0.10196259,
#                -0.11794445, -0.01940324, -0.42705233, -0.19205355, -0.35930676,
#                -0.0974568 ],
#               [-0.19790972,  0.16969336, -0.23183487, -0.53515508,  0.26139754,
#                 0.22712531,  0.49107622,  0.58600579,  0.40461464, -0.1392028 ,
#                 0.3920103 , -0.09166105,  0.42427079,  0.51850803,  0.42394035,
#                 0.39358843],
#               [ 0.36195964,  0.64907472,  0.30817127, -0.09021277,  0.17335893,
#                 0.20634251,  0.47436517,  0.24174525,  0.22825588, -0.3859298 ,
#                 0.17496794,  0.64333359, -0.22540255, -0.30735041, -0.16699589,
#                -0.25533835],
#               [ 0.37255952, -0.33469109, -0.38968882,  0.05322303, -0.45421723,
#                -0.3787753 , -0.09619131, -0.16923193,  0.27941676,  0.48236463,
#                 0.36730602, -0.24614769, -0.07536402,  0.3437463 , -0.08556341,
#                -0.38946117],
#               [-0.28342847, -0.08002174, -0.2699718 ,  0.51183259,  0.00261475,
#                 0.02220215, -0.39377875, -0.36510762,  0.39913342, -0.10648414,
#                -0.51101333, -0.54958565, -0.21264646, -0.3747578 ,  0.47994559,
#                 0.23532864],
#               [-0.01697586, -0.03739086,  0.04770701,  0.21683097, -0.00927167,
#                -0.22504576, -0.26297358, -0.14943215, -0.38376856,  0.16970954,
#                 0.16277625,  0.08828773,  0.54808446,  0.32384488,  0.04362793,
#                -0.13082438],
#               [ 0.17975858, -0.35104813,  0.19804056, -0.12903055, -0.33440911,
#                 0.16787617, -0.4087631 , -0.31925976, -0.12037744, -0.09464076,
#                 0.24823202,  0.13061948, -0.12461543,  0.14704438,  0.14906396,
#                 0.30066957],
#               [ 0.40346051, -0.31733852,  0.3402512 , -0.02986053, -0.53611222,
#                 0.57556547, -0.23424659,  0.29138251,  0.12397298, -0.11424718,
#                 0.06119964,  0.15387102, -0.3470142 , -0.00204029,  0.14886499,
#                 0.34903618]])
#        spams_Y = np.asarray([[ 0.25819889],
#                               [ 0.25819889],
#                               [-0.38729833],
#                               [ 0.25819889],
#                               [ 0.25819889],
#                               [-0.38729833],
#                               [ 0.25819889],
#                               [-0.38729833],
#                               [-0.38729833],
#                               [ 0.25819889]])
#        W_ridge = np.asarray([[-0.08305595],
#                           [ 0.34645966],
#                           [ 0.02759184],
#                           [-0.17568114],
#                           [-0.16459901],
#                           [ 0.48650463],
#                           [ 0.04353418],
#                           [ 0.34172603],
#                           [ 0.2223307 ],
#                           [ 0.15972512],
#                           [-0.42766711],
#                           [-0.19276766],
#                           [ 0.12821563],
#                           [-0.29284748],
#                           [ 0.03788927],
#                           [-0.06673578]])
#        W_l1 = np.asarray([[ 0.        ],
#                           [ 0.3981345 ],
#                           [ 0.        ],
#                           [-0.10537759],
#                           [ 0.        ],
#                           [ 0.48124976],
#                           [ 0.        ],
#                           [ 0.2192977 ],
#                           [ 0.13872083],
#                           [ 0.        ],
#                           [-0.47542659],
#                           [-0.16684663],
#                           [ 0.        ],
#                           [ 0.        ],
#                           [ 0.        ],
#                           [ 0.        ]])
#
#        # ================================================================
#        # using pre-computed values
#        Atv, n_compacts = parsimony.tv.A_from_shape(shape)
#        k = 0.05  # ridge regression coefficient
#        l = 0  # l1 coefficient
#        g = 0  # tv coefficient
#        tvl1l2_fista_ridge = estimators.RidgeRegression_L1_TV(k, l, g,
#                                                  Atv,
#                                                  algorithm=algorithms.FISTA())
#        tvl1l2_fista_ridge.fit(spams_X, spams_Y)
#        k = 0  # ridge regression coefficient
#        l = 0.05  # l1 coefficient
#        g = 0  # tv coefficient
#        tvl1l2_fista_l1 = estimators.RidgeRegression_L1_TV(k, l, g,
#                                                  Atv,
#                                                  algorithm=algorithms.FISTA())
#        tvl1l2_fista_l1.fit(spams_X, spams_Y)
#        err1_ridge = np.sum(np.absolute(
#                          np.dot(spams_X, tvl1l2_fista_ridge.beta) - spams_Y))
#        err1_l1 = np.sum(np.absolute(
#                          np.dot(spams_X, tvl1l2_fista_l1.beta) - spams_Y))
#        err2_ridge = np.sum(np.absolute(np.dot(spams_X, W_ridge) - spams_Y))
#        err2_l1 = np.sum(np.absolute(np.dot(spams_X, W_l1) - spams_Y))
#        self.assertTrue(np.absolute(err1_ridge - err2_ridge) < 0.01)
#        self.assertTrue(np.absolute(err1_l1 - err2_l1) < 0.01)

    def test_smoothed_l1(self):
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
        k = 0.0  # ridge regression coefficient
        l = 0.05  # l1 coefficient
        g = 0.0  # tv coefficient
        # ================================================================
        # using spams
        # http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams006.html#toc21
        import spams
        spams_X = np.asfortranarray(X)
        spams_X = np.asfortranarray(spams_X - np.tile(
                                    np.mean(spams_X, 0),
                                   (spams_X.shape[0], 1)))
        spams_X = spams.normalize(spams_X)
        spams_Y = np.asfortranarray(y)
        spams_Y = np.asfortranarray(spams_Y - np.tile(
                                    np.mean(spams_Y, 0),
                                    (spams_Y.shape[0], 1)))
        spams_Y = spams.normalize(spams_Y)
        W0 = np.zeros((spams_X.shape[1], spams_Y.shape[1]),
                       dtype=np.float64,
                       order="FORTRAN")
        param = {'numThreads': 1, 'verbose': True,
             'lambda1': l, 'it0': 10, 'max_it': 200,
             'L0': 0.1, 'tol': 1e-3, 'intercept': False,
             'pos': False}
        param['compute_gram'] = True
        param['loss'] = 'square'
        param['regul'] = 'l1'
        (W_l1, optim_info) = spams.fistaFlat(spams_Y,
                                              spams_X,
                                              W0,
                                              True,
                                              **param)
        print "spams_X =", repr(spams_X)
        print "spams_Y =", repr(spams_Y)
        print "W_l1 =", repr(W_l1)
        # ================================================================
        # using pre-computed values


        # ================================================================
        # using pre-computed values
        Atv, n_compacts = parsimony.tv.A_from_shape(shape)
        Al1 = sparse.eye(num_ft, num_ft)
        k = 0.05  # ridge regression coefficient
        l = 0  # l1 coefficient
        g = 0  # tv coefficient
        tvl1l2_fista_ridge = estimators.RidgeRegression_SmoothedL1TV(
                    k, l, g,
                    Atv=Atv, Al1=Al1,
                    algorithm=algorithms.ExcessiveGapMethod(max_iter=1000))
        tvl1l2_fista_ridge.fit(spams_X, spams_Y)


if __name__ == "__main__":
    unittest.main()
