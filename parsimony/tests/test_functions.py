# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 10:42:07 2013

@author: jinpeng
"""
import unittest


class TestFunctions(unittest.TestCase):
    def test_ridge_l1(self):
        import numpy as np
        import parsimony.estimators as estimators
        import parsimony.algorithms as algorithms
        import parsimony.tv
        shape = (4, 4, 1)
        num_samples = 10
        num_ft = shape[0] * shape[1] * shape[2]
        X = np.random.random((num_samples, num_ft))
        y = np.random.randint(0, 2, (num_samples, 1))
        k = 0.05  # ridge regression coefficient
        l = 0  # l1 coefficient
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
#             'lambda1': k, 'it0': 10, 'max_it': 200,
#             'L0': 0.1, 'tol': 1e-3, 'intercept': False,
#             'pos': False}
#        param['compute_gram'] = True
#        param['loss'] = 'square'
#        param['regul'] = 'l2'
#        (W_ridge, optim_info) = spams.fistaFlat(spams_Y,
#                                              spams_X,
#                                              W0,
#                                              True,
#                                              **param)
#        param['regul'] = 'l1'
#        (W_l1, optim_info) = spams.fistaFlat(spams_Y,
#                                             spams_X,
#                                             W0,
#                                             True,
#                                             **param)
#        print "spams_X =", repr(spams_X)
#        print "spams_Y =", repr(spams_Y)
#        print "W_ridge =", repr(W_ridge)
#        print "W_l1 =", repr(W_l1)
        # ================================================================
        # using pre-computed values
        spams_X = np.asarray([[-0.62020013,  0.39809644, -0.4354417 ,  0.1276226 ,  0.25042866,
                -0.37853835,  0.2049136 ,  0.26038437, -0.26147759, -0.3468884 ,
                -0.31256216, -0.33767948,  0.18188244, -0.46330705, -0.61563846,
                 0.15202283],
               [-0.17312953,  0.09848125,  0.52135967, -0.48213548,  0.43368937,
                 0.19575179,  0.03878352, -0.39845962, -0.53109945,  0.6372815 ,
                -0.46497223,  0.22836529,  0.25785729,  0.00636549, -0.01793828,
                -0.55756495],
               [-0.02609454, -0.19485542, -0.08859252,  0.35688521,  0.21252098,
                -0.412504  ,  0.18681481,  0.02197316, -0.13867063, -0.10196259,
                -0.11794445, -0.01940324, -0.42705233, -0.19205355, -0.35930676,
                -0.0974568 ],
               [-0.19790972,  0.16969336, -0.23183487, -0.53515508,  0.26139754,
                 0.22712531,  0.49107622,  0.58600579,  0.40461464, -0.1392028 ,
                 0.3920103 , -0.09166105,  0.42427079,  0.51850803,  0.42394035,
                 0.39358843],
               [ 0.36195964,  0.64907472,  0.30817127, -0.09021277,  0.17335893,
                 0.20634251,  0.47436517,  0.24174525,  0.22825588, -0.3859298 ,
                 0.17496794,  0.64333359, -0.22540255, -0.30735041, -0.16699589,
                -0.25533835],
               [ 0.37255952, -0.33469109, -0.38968882,  0.05322303, -0.45421723,
                -0.3787753 , -0.09619131, -0.16923193,  0.27941676,  0.48236463,
                 0.36730602, -0.24614769, -0.07536402,  0.3437463 , -0.08556341,
                -0.38946117],
               [-0.28342847, -0.08002174, -0.2699718 ,  0.51183259,  0.00261475,
                 0.02220215, -0.39377875, -0.36510762,  0.39913342, -0.10648414,
                -0.51101333, -0.54958565, -0.21264646, -0.3747578 ,  0.47994559,
                 0.23532864],
               [-0.01697586, -0.03739086,  0.04770701,  0.21683097, -0.00927167,
                -0.22504576, -0.26297358, -0.14943215, -0.38376856,  0.16970954,
                 0.16277625,  0.08828773,  0.54808446,  0.32384488,  0.04362793,
                -0.13082438],
               [ 0.17975858, -0.35104813,  0.19804056, -0.12903055, -0.33440911,
                 0.16787617, -0.4087631 , -0.31925976, -0.12037744, -0.09464076,
                 0.24823202,  0.13061948, -0.12461543,  0.14704438,  0.14906396,
                 0.30066957],
               [ 0.40346051, -0.31733852,  0.3402512 , -0.02986053, -0.53611222,
                 0.57556547, -0.23424659,  0.29138251,  0.12397298, -0.11424718,
                 0.06119964,  0.15387102, -0.3470142 , -0.00204029,  0.14886499,
                 0.34903618]])
        spams_Y = np.asarray([[ 0.25819889],
                               [ 0.25819889],
                               [-0.38729833],
                               [ 0.25819889],
                               [ 0.25819889],
                               [-0.38729833],
                               [ 0.25819889],
                               [-0.38729833],
                               [-0.38729833],
                               [ 0.25819889]])
        W_ridge = np.asarray([[-0.08305595],
                           [ 0.34645966],
                           [ 0.02759184],
                           [-0.17568114],
                           [-0.16459901],
                           [ 0.48650463],
                           [ 0.04353418],
                           [ 0.34172603],
                           [ 0.2223307 ],
                           [ 0.15972512],
                           [-0.42766711],
                           [-0.19276766],
                           [ 0.12821563],
                           [-0.29284748],
                           [ 0.03788927],
                           [-0.06673578]])
        W_l1 = np.asarray([[ 0.        ],
                           [ 0.3981345 ],
                           [ 0.        ],
                           [-0.10537759],
                           [ 0.        ],
                           [ 0.48124976],
                           [ 0.        ],
                           [ 0.2192977 ],
                           [ 0.13872083],
                           [ 0.        ],
                           [-0.47542659],
                           [-0.16684663],
                           [ 0.        ],
                           [ 0.        ],
                           [ 0.        ],
                           [ 0.        ]])

        # ================================================================
        # using pre-computed values
        Atv, n_compacts = parsimony.tv.A_from_shape(shape)
        k = 0.05  # ridge regression coefficient
        l = 0  # l1 coefficient
        g = 0  # tv coefficient
        tvl1l2_fista_ridge = estimators.RidgeRegression_L1_TV(k, l, g,
                                                  Atv,
                                                  algorithm=algorithms.FISTA())
        tvl1l2_fista_ridge.fit(spams_X, spams_Y)
        k = 0  # ridge regression coefficient
        l = 0.05  # l1 coefficient
        g = 0  # tv coefficient
        tvl1l2_fista_l1 = estimators.RidgeRegression_L1_TV(k, l, g,
                                                  Atv,
                                                  algorithm=algorithms.FISTA())
        tvl1l2_fista_l1.fit(spams_X, spams_Y)
        err1_ridge = np.sum(np.absolute(
                          np.dot(spams_X, tvl1l2_fista_ridge.beta) - spams_Y))
        err1_l1 = np.sum(np.absolute(
                          np.dot(spams_X, tvl1l2_fista_l1.beta) - spams_Y))
        err2_ridge = np.sum(np.absolute(np.dot(spams_X, W_ridge) - spams_Y))
        err2_l1 = np.sum(np.absolute(np.dot(spams_X, W_l1) - spams_Y))
        self.assertTrue(np.absolute(err1_ridge - err2_ridge) < 0.01)
        self.assertTrue(np.absolute(err1_l1 - err2_l1) < 0.01)

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
#        (W_l1, optim_info) = spams.fistaFlat(spams_Y,
#                                              spams_X,
#                                              W0,
#                                              True,
#                                              **param)
#        param['regul'] = 'l2'
#        (W_l2, optim_info) = spams.fistaFlat(spams_Y,
#                                              spams_X,
#                                              W0,
#                                              True,
#                                              **param)
#        print "spams_X =", repr(spams_X)
#        print "spams_Y =", repr(spams_Y)
#        print "W_l1 =", repr(W_l1)
#        print "W_l2 =", repr(W_l2)
        # ================================================================
        # using pre-computed values
        spams_X =np.asarray([[  5.19856444e-01,   1.98565067e-02,  -1.06309916e-01,
                  5.30122459e-01,   3.69935878e-01,   3.76416443e-01,
                 -2.75350802e-01,  -2.88576527e-01,   1.81536696e-01,
                  4.86466087e-01,   2.98503534e-01,  -5.42962362e-02,
                 -3.39623886e-01,  -4.17748409e-01,  -5.73760251e-01,
                 -3.11696224e-01],
               [ -3.94913310e-01,   3.19148830e-01,   6.43585845e-01,
                  5.49464971e-02,  -1.10775436e-01,  -2.75163767e-02,
                 -2.02034877e-01,  -3.29979686e-01,  -5.12041453e-01,
                  5.81334752e-01,   1.67907704e-01,   2.46442195e-01,
                 -4.55725709e-01,  -3.27178329e-01,  -4.26616325e-01,
                  1.68637091e-01],
               [ -3.72266759e-02,  -7.10474145e-03,   1.42787906e-01,
                  1.91548868e-01,   3.69253836e-01,   4.73065994e-01,
                  1.11353590e-01,   6.10831910e-02,   2.08244357e-01,
                  2.50066524e-02,   3.24586006e-01,  -4.76911867e-01,
                  1.69032466e-01,   3.03174816e-01,   2.52657616e-01,
                 -5.37322535e-01],
               [  4.52838558e-01,  -1.27114912e-01,  -3.35896474e-01,
                 -2.32936688e-01,  -3.59032968e-01,  -2.78976102e-01,
                  1.46243199e-01,  -3.96277749e-01,   3.34488449e-01,
                 -3.61910011e-01,  -3.26741577e-01,   3.30131899e-01,
                  3.56461502e-01,  -3.87681439e-02,   4.23534702e-01,
                  1.53836289e-04],
               [  2.18819443e-01,   5.63504650e-01,  -4.43473156e-01,
                  5.02363938e-01,  -3.65050875e-01,   1.36012048e-01,
                 -5.12909327e-01,   1.39278314e-01,   2.66383330e-01,
                 -3.09367330e-01,  -4.04753248e-01,   2.33583752e-01,
                  1.78987067e-01,  -5.88413655e-02,   1.75993568e-01,
                  4.32252962e-01],
               [ -2.27411474e-01,  -1.27272384e-01,  -1.25057541e-01,
                 -2.36513812e-01,  -2.23241600e-02,  -4.49169611e-01,
                  2.16310732e-01,   2.06231913e-01,  -3.16904786e-01,
                  2.74856854e-02,  -8.78854596e-02,   3.15211150e-01,
                 -5.20870987e-01,  -2.71470677e-01,  -2.50429675e-01,
                  3.40254560e-01],
               [ -2.03663680e-01,  -3.41453268e-01,  -1.47585127e-01,
                 -4.35952948e-01,   4.62036983e-01,  -1.06573080e-01,
                  2.01262221e-01,  -9.27743079e-02,  -1.88653825e-01,
                  1.77845390e-02,   3.62285704e-01,  -4.30052620e-01,
                 -4.62360048e-02,   3.16073577e-02,  -7.48430152e-02,
                  3.02609967e-01],
               [ -4.50619036e-01,  -2.52575914e-01,   3.07222868e-01,
                 -3.55573390e-01,  -4.76314038e-01,   8.21010644e-02,
                  1.32610158e-05,   5.92303921e-01,   1.75775869e-01,
                  1.30670614e-01,  -4.62587543e-01,  -1.14250158e-01,
                  2.48787410e-01,  -1.82025241e-01,  -3.78943512e-02,
                 -1.80102707e-01],
               [  1.49841248e-01,   4.04475484e-01,   2.67640128e-01,
                 -1.36276496e-02,   2.89758941e-02,  -4.87710672e-01,
                 -3.16551882e-01,   3.85866923e-01,  -4.62137557e-01,
                 -2.71754334e-01,   3.34082498e-01,   3.26301249e-01,
                  1.71913626e-02,   6.41452619e-01,   1.71683064e-01,
                  1.53373837e-01],
               [ -2.75215161e-02,  -4.51464252e-01,  -2.02914532e-01,
                 -4.37727385e-03,   1.03294886e-01,   2.82350293e-01,
                  6.31663884e-01,  -2.77155992e-01,   3.13308920e-01,
                 -3.25716655e-01,  -2.05397620e-01,  -3.76159362e-01,
                  3.91996780e-01,   3.19797373e-01,   3.39674666e-01,
                 -3.68160788e-01]])
        spams_Y =np.asarray([[-0.20701967],
               [ 0.48304589],
               [-0.20701967],
               [ 0.48304589],
               [-0.20701967],
               [ 0.48304589],
               [-0.20701967],
               [-0.20701967],
               [-0.20701967],
               [-0.20701967]])
        # ================================================================
        # using pre-computed values
        Atv, n_compacts = parsimony.tv.A_from_shape(shape)
        Al1 = sparse.eye(num_ft, num_ft)
        k = 0.05  # ridge regression coefficient
        l = 0.05  # l1 coefficient
        g = 0.05  # tv coefficient
        rr = estimators.RidgeRegression_SmoothedL1TV(
                    k, l, g,
                    Atv=Atv, Al1=Al1,
                    algorithm=algorithms.ExcessiveGapMethod(max_iter=1000))
        res = rr.fit(spams_X, spams_Y)
        verified_beta = np.asarray([[ -3.06795144e-02],
                                   [ -1.15708593e-02],
                                   [ -1.09255970e-05],
                                   [ -2.38825088e-01],
                                   [ -6.74746378e-02],
                                   [ -5.50823531e-02],
                                   [  1.91713527e-05],
                                   [ -2.38855356e-01],
                                   [ -6.92107632e-02],
                                   [ -3.09792738e-02],
                                   [ -1.06060363e-02],
                                   [  2.68985405e-01],
                                   [ -1.37646869e-01],
                                   [ -1.37638461e-01],
                                   [  6.50395154e-06],
                                   [  2.09415248e-05]])
        err = np.sum(np.abs(verified_beta - rr.beta))
        self.assertTrue(err < 0.01)

if __name__ == "__main__":
    unittest.main()
