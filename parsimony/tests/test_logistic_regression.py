# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 10:33:40 2014

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import unittest
from nose.tools import assert_less, assert_almost_equal

import numpy as np

from tests import TestCase
import parsimony.utils.consts as consts


class TestLogisticRegression(TestCase):

    def test_logistic_regression(self):
        # Spams: http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams006.html#toc23

        import parsimony.functions.losses as losses
        import parsimony.algorithms.explicit as explicit
        import parsimony.start_vectors as start_vectors
        import parsimony.utils.maths as maths

        np.random.seed(42)

        start_vector = start_vectors.RandomStartVector(normalise=True)

        n, p = 50, 100

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        X = np.random.multivariate_normal(mean, Sigma, n)
        y = np.array(np.random.randint(0, 2, (n, 1)), dtype=X.dtype)

        eps = 1e-8
        max_iter = 10000

        gd = explicit.GradientDescent(eps=eps, max_iter=max_iter)
        lr = losses.LogisticRegression(X, y, mean=True)
        beta_start = start_vector.get_vector((p, 1))

        beta = gd.run(lr, beta_start)

        try:
            import spams

            params = {"loss": "logistic",
                      "regul": "l2",
                      "lambda1": 0.0,
                      "max_it": max_iter,
                      "tol": eps,
                      "ista": True,
                      "numThreads": -1,
                      "intercept": False,
                     }

            y_ = y.copy()
            y_[y_ == 0.0] = -1.0
            beta_spams, optim_info = \
                    spams.fistaFlat(Y=np.asfortranarray(y_),
                                    X=np.asfortranarray(X),
                                    W0=np.asfortranarray(beta_start),
                                    return_optim_info=True,
                                    **params)

        except ImportError:
            beta_spams = np.asarray(
                    [[-1.01027835], [0.04541433], [-0.36492995], [0.61694659],
                     [0.31352106], [-0.45514542], [-0.06516154], [0.16322194],
                     [1.57370712], [1.01770363], [-1.18053922], [0.94475217],
                     [1.09808833], [0.68986878], [-0.59109072], [-0.35280848],
                     [1.13403386], [1.60746596], [-0.51702108], [0.40298098],
                     [0.06973893], [1.38913351], [0.31263145], [0.25254335],
                     [1.65984054], [0.19778432], [1.20301178], [1.37325924],
                     [-0.81674455], [0.65926444], [-0.59679473], [0.38773981],
                     [-0.58720461], [-0.48949141], [0.55399832], [0.75475734],
                     [-0.93326053], [-0.55721247], [0.94682476], [0.59778018],
                     [0.08025665], [0.09741072], [-0.75940411], [1.00956122],
                     [0.86698692], [0.22242015], [0.08627418], [0.10561352],
                     [0.50026322], [1.54323783], [0.43163912], [1.81605466],
                     [-0.39638692], [0.08939998], [-1.10451808], [-0.93858999],
                     [-0.68055786], [-0.24316492], [0.60939807], [-1.20860996],
                     [0.96899678], [1.11948487], [0.64488373], [-0.65170164],
                     [0.33778775], [-0.1380265], [-0.04784483], [-0.02324114],
                     [0.55396485], [0.53091428], [-0.06142249], [-1.481608],
                     [-1.09504543], [-1.2885626], [0.4204063], [0.691696],
                     [-0.23795757], [1.13300874], [-0.02437729], [0.61871999],
                     [0.95775709], [1.586174], [1.0027941], [0.84361898],
                     [0.25268826], [-0.72440374], [-0.96623248], [-1.153285],
                     [0.60933874], [-0.35755106], [-0.30428306], [-0.11095325],
                     [0.84154014], [-0.27480962], [-0.95924936], [0.54482999],
                     [-1.41566245], [0.3111633], [-0.25463532], [0.8865664]])

        re = maths.norm(beta - beta_spams) / maths.norm(beta_spams)
#        print "re:", re
        assert_almost_equal(re, 0.063010,
                            msg="The found regression vector is not correct.",
                            places=5)

        f_parsimony = lr.f(beta)
        f_spams = lr.f(beta_spams)
        if abs(f_spams) > consts.TOLERANCE:
            err = abs(f_parsimony - f_spams) / f_spams
        else:
            err = abs(f_parsimony - f_spams)
#        print "err:", err
        assert_almost_equal(err, 0.783121,
                            msg="The found regression vector does not give " \
                                "the correct function value.",
                            places=5)

    def test_logistic_regression_l1(self):
        # Spams: http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams006.html#toc23

        from parsimony.functions import CombinedFunction
        import parsimony.functions.losses as losses
        import parsimony.functions.penalties as penalties
        import parsimony.algorithms.explicit as explicit
        import parsimony.start_vectors as start_vectors
        import parsimony.utils.maths as maths

        np.random.seed(42)

        start_vector = start_vectors.RandomStartVector(normalise=True)

        n, p = 50, 100

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        X = np.random.multivariate_normal(mean, Sigma, n)
        y = np.array(np.random.randint(0, 2, (n, 1)), dtype=X.dtype)

        eps = 1e-8
        max_iter = 10000

        l = 0.001
        k = 0.0
        g = 0.0

        algorithm = explicit.ISTA(eps=eps, max_iter=max_iter)
        function = CombinedFunction()
        function.add_function(losses.LogisticRegression(X, y, mean=True))
        function.add_prox(penalties.L1(l))
        beta_start = start_vector.get_vector((p, 1))

        beta = algorithm.run(function, beta_start)

        try:
            import spams

            params = {"loss": "logistic",
                      "regul": "l1",
                      "lambda1": l,
                      "max_it": max_iter,
                      "tol": eps,
                      "ista": True,
                      "numThreads": -1,
                      "intercept": False,
                     }

            y_ = y.copy()
            y_[y_ == 0.0] = -1.0
            beta_spams, optim_info = \
                    spams.fistaFlat(Y=np.asfortranarray(y_),
                                    X=np.asfortranarray(X),
                                    W0=np.asfortranarray(beta_start),
                                    return_optim_info=True,
                                    **params)

        except ImportError:
            beta_spams = np.asarray(
                    [[-1.15299513], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                     [1.6052226], [0.86846838], [-0.30706116], [0.31483119],
                     [0.91018498], [0.07378518], [0.], [0.], [0.91509995],
                     [0.79327426], [0.], [0.31663264], [0.], [0.74234398],
                     [0.], [0.], [1.36178271], [0.], [0.], [0.62679508],
                     [-0.63081276], [0.], [0.], [0.], [0.], [0.], [0.],
                     [0.74699032], [-0.23479909], [0.], [0.], [0.], [0.], [0.],
                     [-0.21076412], [0.], [0.], [0.], [0.], [0.], [0.],
                     [1.68165668], [0.], [2.09181095], [0.], [0.],
                     [-0.66720945], [-0.0173993], [-0.12982041], [0.], [0.],
                     [-0.90838118], [0.], [0.64843176], [0.], [0.], [0.], [0.],
                     [0.], [0.], [0.], [0.], [0.], [-0.94676745],
                     [-1.52904001], [-1.09095236], [0.], [0.], [0.],
                     [0.42959206], [0.], [0.], [0.23419629], [1.28258911],
                     [0.], [0.], [0.], [0.], [-1.00708153], [-0.58957659],
                     [0.10385635], [0.], [0.], [0.], [0.13107728], [0.], [0.],
                     [0.32741694], [-1.52896011], [0.], [0.], [0.55867497]])

        re = maths.norm(beta - beta_spams) / maths.norm(beta_spams)
#        print "re:", re
        assert_almost_equal(re, 0.070202,
                            msg="The found regression vector is not correct.",
                            places=5)

        f_parsimony = function.f(beta)
        f_spams = function.f(beta_spams)
        if abs(f_spams) > consts.TOLERANCE:
            err = abs(f_parsimony - f_spams) / f_spams
        else:
            err = abs(f_parsimony - f_spams)
#        print "err:", err
        assert_almost_equal(err, 0.001084,
                            msg="The found regression vector does not give " \
                                "the correct function value.",
                            places=5)

    def test_logistic_regression_l2(self):
        # Spams: http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams006.html#toc23

        from parsimony.functions import CombinedFunction
        import parsimony.functions.losses as losses
        import parsimony.functions.penalties as penalties
        import parsimony.algorithms.explicit as explicit
        import parsimony.start_vectors as start_vectors
        import parsimony.utils.maths as maths

        np.random.seed(42)

        start_vector = start_vectors.RandomStartVector(normalise=True)

        n, p = 50, 100

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        X = np.random.multivariate_normal(mean, Sigma, n)
        y = np.array(np.random.randint(0, 2, (n, 1)), dtype=X.dtype)

        eps = 1e-8
        max_iter = 10000

        l = 0.0
        k = 0.618
        g = 0.0

        gd = explicit.GradientDescent(eps=eps, max_iter=max_iter)
        function = CombinedFunction()
        function.add_function(losses.LogisticRegression(X, y, mean=True))
        function.add_penalty(penalties.L2(k))
        beta_start = start_vector.get_vector((p, 1))

        beta = gd.run(function, beta_start)

        try:
            import spams

            params = {"loss": "logistic",
                      "regul": "l2",
                      "lambda1": k,
                      "max_it": max_iter,
                      "tol": eps,
                      "ista": True,
                      "numThreads": -1,
                      "intercept": False,
                     }

            y_ = y.copy()
            y_[y_ == 0.0] = -1.0
            beta_spams, optim_info = \
                    spams.fistaFlat(Y=np.asfortranarray(y_),
                                    X=np.asfortranarray(X),
                                    W0=np.asfortranarray(beta_start),
                                    return_optim_info=True,
                                    **params)

        except ImportError:
            beta_spams = np.asarray(
                    [[-7.08364262e-02], [1.91036495e-02], [-1.75255424e-02],
                     [7.47128825e-02], [7.40632464e-02], [-3.75244731e-02],
                     [-2.92621926e-02], [-1.05369719e-02], [9.66112304e-02],
                     [7.51422863e-02], [-8.68237085e-02], [6.18339678e-02],
                     [6.71375958e-02], [8.51932542e-02], [-2.89076676e-02],
                     [-4.38200875e-02], [1.07707995e-01], [1.06084672e-01],
                     [-4.61642640e-02], [4.65185709e-02], [-1.62870583e-02],
                     [6.43003085e-02], [2.20164633e-02], [-1.72292280e-02],
                     [1.50433895e-01], [-3.13280748e-03], [8.17609235e-02],
                     [6.43320148e-02], [-1.05817876e-01], [2.91561184e-02],
                     [-5.76679119e-02], [2.69019465e-02], [-4.92060035e-02],
                     [-2.40326340e-02], [3.82109840e-02], [6.56241233e-02],
                     [-7.59938589e-02], [-4.01138454e-02], [6.76081041e-02],
                     [2.26736428e-02], [-4.16219986e-02], [9.21053628e-03],
                     [-6.27279544e-02], [4.35017254e-02], [7.09329235e-02],
                     [3.59671264e-02], [-2.61329331e-02], [-2.47750482e-05],
                     [4.16761269e-02], [1.15249546e-01], [2.79507760e-02],
                     [1.40411020e-01], [-3.56203099e-02], [7.41165779e-03],
                     [-6.20244160e-02], [-6.25563475e-02], [-4.49866882e-02],
                     [1.56899626e-02], [5.75579342e-02], [-8.78078022e-02],
                     [6.40786745e-02], [7.24929357e-02], [-1.91301750e-02],
                     [-6.35141022e-02], [-1.43111575e-02], [-5.08493775e-03],
                     [3.49268208e-02], [2.17157072e-02], [3.75927668e-02],
                     [2.43141126e-02], [1.90823190e-02], [-1.16637083e-01],
                     [-1.15662240e-01], [-1.04568929e-01], [5.47195183e-02],
                     [5.47161271e-02], [5.30288992e-03], [9.42356174e-02],
                     [6.65494972e-03], [6.48408209e-02], [4.97005251e-02],
                     [8.01004869e-02], [8.66977987e-02], [7.55207729e-02],
                     [3.12955645e-03], [-8.60849149e-02], [-6.89566685e-02],
                     [-1.01972189e-01], [-2.29940810e-03], [-4.11457076e-02],
                     [-1.23086609e-02], [-2.61123403e-03], [4.92835514e-02],
                     [-4.43815143e-02], [-3.55913310e-02], [5.64363199e-02],
                     [-6.96402195e-02], [1.73230866e-02], [1.90892359e-02],
                     [5.68265025e-02]])

        re = maths.norm(beta - beta_spams) / maths.norm(beta_spams)
#        print "re:", re
        assert_almost_equal(re, 2.926735e-09,
                            msg="The found regression vector is not correct.",
                            places=5)

        f_parsimony = function.f(beta)
        f_spams = function.f(beta_spams)
        if abs(f_spams) > consts.TOLERANCE:
            err = abs(f_parsimony - f_spams) / f_spams
        else:
            err = abs(f_parsimony - f_spams)
#        print "err:", err
        assert_almost_equal(err, 0.0,
                            msg="The found regression vector does not give " \
                                "the correct function value.",
                            places=5)

    def test_logistic_regression_gl(self):
        # Spams: http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams006.html#toc23

        from parsimony.functions import CombinedFunction
        import parsimony.functions.losses as losses
        import parsimony.functions.penalties as penalties
        import parsimony.algorithms.explicit as explicit
        import parsimony.start_vectors as start_vectors
        import parsimony.utils.maths as maths
        import parsimony.functions.nesterov.gl as gl

        np.random.seed(42)

        start_vector = start_vectors.RandomStartVector(normalise=True)

        # Note that p must be even!
        n, p = 50, 100
        groups = [range(0, p / 2), range(p / 2, p)]
#        weights = [1.5, 0.5]

        A = gl.A_from_groups(p, groups=groups)  # , weights=weights)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        X = np.random.multivariate_normal(mean, Sigma, n)
        y = np.array(np.random.randint(0, 2, (n, 1)), dtype=X.dtype)

        eps = 1e-8
        max_iter = 7000

        l = 0.0
        k = 0.0
        g = 0.001
        mu = 5e-4

        algorithm = explicit.ISTA(eps=eps, max_iter=max_iter)
        function = CombinedFunction()
        function.add_function(losses.LogisticRegression(X, y, mean=True))
        function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                  penalty_start=0))
        beta_start = start_vector.get_vector((p, 1))

        beta = algorithm.run(function, beta_start)

        try:
            import spams

            params = {"loss": "logistic",
                      "regul": "group-lasso-l2",
                      "groups": np.array([1] * (p / 2) + [2] * (p / 2),
                                         dtype=np.int32),
                      "lambda1": g,
                      "max_it": max_iter,
                      "tol": eps,
                      "ista": True,
                      "numThreads": -1,
                      "intercept": False,
                     }

            y_ = y.copy()
            y_[y_ == 0.0] = -1.0
            beta_spams, optim_info = \
                    spams.fistaFlat(Y=np.asfortranarray(y_),
                                    X=np.asfortranarray(X),
                                    W0=np.asfortranarray(beta_start),
                                    return_optim_info=True,
                                    **params)

        except ImportError:
            beta_spams = np.asarray(
                    [[-0.72542349], [0.02830505], [-0.21973781], [0.41495258],
                     [0.229409], [-0.32370782], [-0.15752327], [0.0632292],
                     [1.06252282], [0.66542057], [-0.84258213], [0.69489539],
                     [0.72518289], [0.46540807], [-0.34997616], [-0.34717853],
                     [0.78537712], [1.09381737], [-0.33570154], [0.25842894],
                     [-0.00959316], [0.92931029], [0.16074866], [0.11725611],
                     [1.18146773], [0.03350294], [0.8230971], [0.98554419],
                     [-0.61217155], [0.40936428], [-0.43282706], [0.19459689],
                     [-0.44080338], [-0.33548882], [0.32473485], [0.56413217],
                     [-0.66081985], [-0.43362073], [0.58328254], [0.41602645],
                     [-0.01677669], [0.06827701], [-0.57902052], [0.64755089],
                     [0.5010607], [0.09013846], [0.03085689], [0.0684073],
                     [0.2971785], [1.03409051], [0.2652446], [1.23882265],
                     [-0.27871008], [0.05570645], [-0.76659011], [-0.66016803],
                     [-0.51300177], [-0.2289061], [0.40504384], [-0.8754489],
                     [0.65528664], [0.76493272], [0.45700299], [-0.43729913],
                     [0.16797076], [-0.12563883], [-0.05556865], [0.01500861],
                     [0.27430934], [0.36472081], [-0.12008283], [-1.04799662],
                     [-0.78768917], [-0.93620521], [0.21787308], [0.44862306],
                     [-0.20981051], [0.75096296], [-0.0357571], [0.40723417],
                     [0.65944272], [1.12012117], [0.70820101], [0.57642298],
                     [0.12019244], [-0.54588467], [-0.68402079], [-0.86922667],
                     [0.41024387], [-0.28984963], [-0.22063841], [-0.06986448],
                     [0.5727723], [-0.24701453], [-0.73092213], [0.31178252],
                     [-1.05972579], [0.19986263], [-0.1638552], [0.6232789]])

        re = maths.norm(beta - beta_spams) / maths.norm(beta_spams)
#        print "re:", re
        assert_almost_equal(re, 0.065260,
                            msg="The found regression vector is not correct.",
                            places=5)

        f_parsimony = function.f(beta)
        f_spams = function.f(beta_spams)
        if abs(f_spams) > consts.TOLERANCE:
            err = abs(f_parsimony - f_spams) / f_spams
        else:
            err = abs(f_parsimony - f_spams)
#        print "err:", err
        assert_almost_equal(err, 0.003466,
                            msg="The found regression vector does not give " \
                                "the correct function value.",
                            places=5)

    def test_logistic_regression_l1_l2(self):
        # Spams: http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams006.html#toc23

        from parsimony.functions import CombinedFunction
        import parsimony.functions.losses as losses
        import parsimony.functions.penalties as penalties
        import parsimony.algorithms.explicit as explicit
        import parsimony.start_vectors as start_vectors
        import parsimony.utils.maths as maths

        np.random.seed(42)

        start_vector = start_vectors.RandomStartVector(normalise=True)

        n, p = 50, 100

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        X = np.random.multivariate_normal(mean, Sigma, n)
        y = np.array(np.random.randint(0, 2, (n, 1)), dtype=X.dtype)

        eps = 1e-8
        max_iter = 10000

        l = 0.0318
        k = 1.0 - l
        g = 0.0

        algorithm = explicit.ISTA(eps=eps, max_iter=max_iter)
        function = CombinedFunction()
        function.add_function(losses.LogisticRegression(X, y, mean=True))
        function.add_penalty(penalties.L2(k))
        function.add_prox(penalties.L1(l))
        beta_start = start_vector.get_vector((p, 1))

        beta = algorithm.run(function, beta_start)

        try:
            import spams

            params = {"loss": "logistic",
                      "regul": "elastic-net",
                      "lambda1": l,
                      "lambda2": k,
                      "max_it": max_iter,
                      "tol": eps,
                      "ista": True,
                      "numThreads": -1,
                      "intercept": False,
                     }

            y_ = y.copy()
            y_[y_ == 0.0] = -1.0
            beta_spams, optim_info = \
                    spams.fistaFlat(Y=np.asfortranarray(y_),
                                    X=np.asfortranarray(X),
                                    W0=np.asfortranarray(beta_start),
                                    return_optim_info=True,
                                    **params)

        except ImportError:
            beta_spams = np.asarray(
                    [[-0.02529761], [0.], [0.], [0.0334721], [0.05321362],
                     [-0.01287593], [0.], [0.], [0.05285862], [0.04136545],
                     [-0.03874372], [0.02097468], [0.02078901], [0.05317634],
                     [0.], [-0.01038235], [0.06953451], [0.05629869],
                     [-0.01385761], [0.0197789], [0.], [0.01531954], [0.],
                     [0.], [0.0979065], [0.], [0.03202502], [0.00465224],
                     [-0.07322855], [0.], [-0.02333038], [0.], [-0.0219676],
                     [0.], [0.00073144], [0.02900314], [-0.0364214],
                     [-0.00497983], [0.02181465], [0.], [-0.02216263], [0.],
                     [-0.02871861], [0.00014289], [0.02549944], [0.01219949],
                     [0.], [0.], [0.01284684], [0.07015268], [0.],
                     [0.09246748], [-0.004762], [0.], [-0.01817149],
                     [-0.02119763], [-0.00932684], [0.], [0.02950021],
                     [-0.04073497], [0.02537134], [0.02957037], [-0.00782987],
                     [-0.03301447], [0.], [0.], [0.01734085], [0.00280076],
                     [0.], [0.], [0.], [-0.06911439], [-0.08260008],
                     [-0.06254957], [0.02925817], [0.02162837], [0.],
                     [0.04903786], [0.], [0.03930223], [0.00839264],
                     [0.03145341], [0.04055641], [0.0379336], [0.],
                     [-0.05476343], [-0.02684736], [-0.05647316], [0.],
                     [-0.0181865], [0.], [0.], [0.0096403], [-0.01322636],
                     [0.], [0.02832135], [-0.01693505], [0.], [0.],
                     [0.02531008]])

        re = maths.norm(beta - beta_spams) / maths.norm(beta_spams)
#        print "re:", re
        assert_almost_equal(re, 1.143472e-08,
                            msg="The found regression vector is not correct.",
                            places=5)

        f_parsimony = function.f(beta)
        f_spams = function.f(beta_spams)
        if abs(f_spams) > consts.TOLERANCE:
            err = abs(f_parsimony - f_spams) / f_spams
        else:
            err = abs(f_parsimony - f_spams)
#        print "err:", err
        assert_almost_equal(err, 3.644088e-16,
                            msg="The found regression vector does not give " \
                                "the correct function value.",
                            places=5)

    def test_logistic_regression_l1_gl(self):
        # Spams: http://spams-devel.gforge.inria.fr/doc-python/html/doc_spams006.html#toc23

        from parsimony.functions import CombinedFunction
        import parsimony.functions.losses as losses
        import parsimony.functions.penalties as penalties
        import parsimony.algorithms.explicit as explicit
        import parsimony.start_vectors as start_vectors
        import parsimony.utils.maths as maths
        import parsimony.functions.nesterov.gl as gl

        np.random.seed(42)

        start_vector = start_vectors.RandomStartVector(normalise=True)

        # Note that p must be even!
        n, p = 50, 100
        groups = [range(0, p / 2), range(p / 2, p)]
#        weights = [1.5, 0.5]

        A = gl.A_from_groups(p, groups=groups)  # , weights=weights)

        alpha = 0.9
        Sigma = alpha * np.eye(p, p) \
              + (1.0 - alpha) * np.random.randn(p, p)
        mean = np.zeros(p)
        X = np.random.multivariate_normal(mean, Sigma, n)
        y = np.array(np.random.randint(0, 2, (n, 1)), dtype=X.dtype)

        eps = 1e-8
        max_iter = 6600

        l = 0.01
        k = 0.0
        g = 0.001
        mu = 5e-4

        algorithm = explicit.ISTA(eps=eps, max_iter=max_iter)
        function = CombinedFunction()
        function.add_function(losses.LogisticRegression(X, y, mean=True))
        function.add_penalty(gl.GroupLassoOverlap(l=g, A=A, mu=mu,
                                                  penalty_start=0))
        function.add_prox(penalties.L1(l))
        beta_start = start_vector.get_vector((p, 1))

        beta = algorithm.run(function, beta_start)

        try:
            import spams

            params = {"loss": "logistic",
                      "regul": "sparse-group-lasso-l2",
                      "groups": np.array([1] * (p / 2) + [2] * (p / 2),
                                         dtype=np.int32),
                      "lambda1": g,
                      "lambda2": l,
                      "max_it": max_iter,
                      "tol": eps,
                      "ista": True,
                      "numThreads": -1,
                      "intercept": False,
                     }

            y_ = y.copy()
            y_[y_ == 0.0] = -1.0
            beta_spams, optim_info = \
                    spams.fistaFlat(Y=np.asfortranarray(y_),
                                    X=np.asfortranarray(X),
                                    W0=np.asfortranarray(beta_start),
                                    return_optim_info=True,
                                    **params)

        except ImportError:
            beta_spams = np.asarray(
                    [[-0.49445071], [0.], [0.], [0.], [0.], [0.], [0.], [0.],
                     [0.90020246], [0.40967343], [-0.17363366], [0.],
                     [0.4458841], [0.07978072], [0.], [0.], [0.56516372],
                     [0.3811369], [0.], [0.07324983], [0.], [0.41067348], [0.],
                     [0.], [0.79465353], [0.], [0.], [0.22514379],
                     [-0.28391624], [0.], [0.], [0.], [0.], [0.], [0.],
                     [0.57412006], [-0.08485725], [0.], [0.], [0.], [0.], [0.],
                     [-0.16013528], [0.], [0.], [0.], [0.], [0.], [0.],
                     [1.01262503], [0.], [1.24327631], [0.], [0.],
                     [-0.35373743], [0.], [-0.02456871], [0.], [0.],
                     [-0.44805359], [0.], [0.39618791], [0.], [0.], [0.], [0.],
                     [0.], [0.], [0.], [0.], [0.], [-0.4650603], [-0.86402976],
                     [-0.64165934], [0.], [0.], [0.], [0.24080178], [0.], [0.],
                     [0.02534903], [0.57627445], [0.], [0.], [0.],
                     [-0.03991855], [-0.35161357], [-0.35708467], [0.], [0.],
                     [0.], [0.], [0.], [0.], [0.], [0.26739579], [-0.6467167],
                     [0.], [0.], [0.19439507]])

        re = maths.norm(beta - beta_spams) / maths.norm(beta_spams)
#        print "re:", re
        assert_almost_equal(re, 0.000915,
                            msg="The found regression vector is not correct.",
                            places=5)

        f_parsimony = function.f(beta)
        f_spams = function.f(beta_spams)
        if abs(f_spams) > consts.TOLERANCE:
            err = abs(f_parsimony - f_spams) / f_spams
        else:
            err = abs(f_parsimony - f_spams)
#        print "err:", err
        assert_almost_equal(err, 5.848802e-08,
                            msg="The found regression vector does not give " \
                                "the correct function value.",
                            places=5)

if __name__ == "__main__":
    unittest.main()