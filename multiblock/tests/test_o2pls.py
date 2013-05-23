# -*- coding: utf-8 -*-
"""
Created on Thu May 23 09:34:47 2013

@author: Tommy Löfstedt
"""

import numpy as np
import preprocess
import methods
from utils.testing import assert_array_almost_equal
from utils import corr, TOLERANCE, direct


def test():

    np.random.seed(42)

    # 000011100
    p = np.vstack((np.zeros((4, 1)), np.ones((3, 1)), np.zeros((2, 1))))
    # 001110000
    q = np.vstack((np.zeros((2, 1)), np.ones((3, 1)), np.zeros((4, 1))))
    t = np.random.randn(10, 1)

    # 111111000
    po = np.vstack((np.ones((4, 1)), np.ones((2, 1)), np.zeros((3, 1))))
    to = np.random.randn(10, 1)

    # 000111111
    qo = np.vstack((np.zeros((3, 1)), np.ones((2, 1)), np.ones((4, 1))))
    uo = np.random.randn(10, 1)

    Q, R = np.linalg.qr(np.hstack((t, to, uo)))
    t = Q[:, [0]]
    to = Q[:, [1]]
    uo = Q[:, [2]]

    X = np.dot(t, p.T) + np.dot(to, po.T)
    Y = np.dot(t, q.T) + np.dot(uo, qo.T)

    svd = methods.SVD(num_comp=1)
    svd.fit(np.dot(X.T, Y))
    t_svd = np.dot(X, svd.U)
    u_svd = np.dot(Y, svd.V)

    o2pls = methods.O2PLS(num_comp=[1, 1, 1])
    o2pls.fit(X, Y)

    Xhat = np.dot(o2pls.T, o2pls.P.T) + np.dot(o2pls.To, o2pls.Po.T)
    assert_array_almost_equal(X, Xhat, decimal=5, err_msg="O2PLS does not" \
            " give a correct reconstruction of X")
    Yhat = np.dot(o2pls.U, o2pls.Q.T) + np.dot(o2pls.Uo, o2pls.Qo.T)
    assert_array_almost_equal(Y, Yhat, decimal=5, err_msg="O2PLS does not" \
            " give a correct reconstruction of Y")

    assert np.abs(corr(o2pls.T, o2pls.U)) > np.abs(corr(t_svd, u_svd))
    assert np.abs(corr(o2pls.T, o2pls.U)) > np.abs(corr(t_svd, u_svd))
    assert np.abs(corr(o2pls.T, t)) > np.abs(corr(t_svd, t))
    assert np.abs(corr(o2pls.U, t)) > np.abs(corr(u_svd, t))

    assert ((p > TOLERANCE) == (np.abs(o2pls.W) > TOLERANCE)).all()
    assert ((p > TOLERANCE) == (np.abs(o2pls.P) > TOLERANCE)).all()
    assert ((po > TOLERANCE) == (np.abs(o2pls.Po) > TOLERANCE)).all()
    assert np.dot(o2pls.W.T, o2pls.Wo) < TOLERANCE

    assert ((q > TOLERANCE) == (np.abs(o2pls.C) > TOLERANCE)).all()
    assert ((q > TOLERANCE) == (np.abs(o2pls.Q) > TOLERANCE)).all()
    assert ((qo > TOLERANCE) == (np.abs(o2pls.Qo) > TOLERANCE)).all()
    assert np.dot(o2pls.C.T, o2pls.Co) < TOLERANCE

    # Compare to known solution
    W = np.asarray([[-0.000000003587], [-0.000000003587], [-0.000000003587],
                    [-0.000000003587], [0.5773502687753], [0.5773502687753],
                    [0.5773502700183], [0],               [0]])
    W, o2pls.W = direct(W, o2pls.W, compare=True)
    assert_array_almost_equal(W, o2pls.W, decimal=5, err_msg="O2PLS does not" \
            " give the correct weights in X")

    T = np.asarray([[-0.3320724686729], [0.0924349866604], [-0.433004639499],
                    [-1.0182038851347], [0.1565405190876], [0.1565295366198],
                    [-1.0557643604012], [-0.513059567452], [0.3138616360639],
                    [-0.3627222076846]])
    T, o2pls.T = direct(T, o2pls.T, compare=True)
    assert_array_almost_equal(T, o2pls.T, decimal=5, err_msg="O2PLS does not" \
            " give the correct scores in X")

    P = np.asarray([[-0.000000003587], [-0.000000003587], [-0.000000003587],
                    [-0.000000003587], [0.5773502687753], [0.5773502687753],
                    [0.5773502700183], [0],               [0]])
    P, o2pls.P = direct(P, o2pls.P, compare=True)
    assert_array_almost_equal(P, o2pls.P, decimal=5, err_msg="O2PLS does not" \
            " give the correct loadings in X")

    Wo = np.asarray([[-0.462910049661], [-0.4629100496613], [-0.4629100496613],
                     [-0.462910049661], [-0.1543033544685], [-0.1543033544685],
                     [0.3086066967678], [0],                [0]])
    Wo, o2pls.Wo = direct(Wo, o2pls.Wo, compare=True)
    assert_array_almost_equal(Wo, o2pls.Wo, decimal=5, err_msg="O2PLS does " \
            "not give the correct unique weights in X")

    To = np.asarray([[0.1166363558019], [0.3982081656371], [-0.460718789788],
                     [0.7141661638895], [1.3523258395355], [0.5103916850288],
                     [0.0373342153684], [-0.565848461012], [0.8644955981846],
                     [0.7835700349552]])
    To, o2pls.To = direct(To, o2pls.To, compare=True)
    assert_array_almost_equal(To, o2pls.To, decimal=5, err_msg="O2PLS does " \
            "not give the correct unique scores in X")

    Po = np.asarray([[-0.462910047927], [-0.4629100479271], [-0.4629100479271],
                     [-0.462910047927], [-0.4629100494572], [-0.4629100494572],
                     [0.0000000000150], [0],                [0]])
    Po, o2pls.Po = direct(Po, o2pls.Po, compare=True)
    assert_array_almost_equal(Po, o2pls.Po, decimal=5, err_msg="O2PLS does " \
            "not give the correct unique loadings in X")

    C = np.asarray([[0],               [0],               [0.5773502653499],
                    [0.5773502711095], [0.5773502711095], [-0.000000000646],
                    [-0.000000000646], [-0.000000000646], [-0.000000000646]])
    C, o2pls.C = direct(C, o2pls.C, compare=True)
    assert_array_almost_equal(C, o2pls.C, decimal=5, err_msg="O2PLS does not" \
            " give the correct weights in Y")

    U = np.asarray([[-0.3320724688962], [0.0924349962939], [-0.433004640520],
                    [-1.0182039005573], [0.1565405177895], [0.1565295466182],
                    [-1.0557643651281], [-0.513059568417], [0.3138616457148],
                    [-0.3627222001620]])
    U, o2pls.U = direct(U, o2pls.U, compare=True)
    assert_array_almost_equal(U, o2pls.U, decimal=5, err_msg="O2PLS does not" \
            " give the correct scores in Y")

    Q = np.asarray([[0],               [0],               [0.5773502653499],
                    [0.5773502711095], [0.5773502711095], [-0.000000000646],
                    [-0.000000000646], [-0.000000000646], [-0.000000000646]])
    Q, o2pls.Q = direct(Q, o2pls.Q, compare=True)
    assert_array_almost_equal(Q, o2pls.Q, decimal=5, err_msg="O2PLS does not" \
            " give the correct loadings in Y")

    Co = np.asarray([[-0.000000000000], [-0.0000000000000], [0.3086067007710],
                     [-0.154303349882], [-0.1543033498817], [-0.462910049756],
                     [-0.462910049756], [-0.4629100497585], [-0.462910049756]])
    Co, o2pls.Co = direct(Co, o2pls.Co, compare=True)
    assert_array_almost_equal(Co, o2pls.Co, decimal=5, err_msg="O2PLS does " \
            "not give the correct unique weights in Y")

    Uo = np.asarray([[-1.895421907630], [0.0598668575527], [-0.069696302091],
                     [0.4526578568357], [-0.143031804891], [-0.337137833530],
                     [0.5494330695544], [-0.394204878737], [0.3248866786239],
                     [-0.404667969696]])
    Uo, o2pls.Uo = direct(Uo, o2pls.Uo, compare=True)
    assert_array_almost_equal(Uo, o2pls.Uo, decimal=5, err_msg="O2PLS does " \
            "not give the correct unique scores in Y")

    Qo = np.asarray([[0],                [0],                [0.0000000008807],
                     [-0.4629100498755], [-0.462910049876], [-0.462910049909],
                     [-0.4629100499092], [-0.462910049909], [-0.462910049909]])
    Qo, o2pls.Qo = direct(Qo, o2pls.Qo, compare=True)
    assert_array_almost_equal(Qo, o2pls.Qo, decimal=5, err_msg="O2PLS does " \
            "not give the correct unique loadings in Y")

    # O2PLS with random dataset, compare to known solution on the data
    np.random.seed(42)

    # Test primal version
    X = np.random.rand(10, 5)
    Y = np.random.rand(10, 5)

    preprocX = preprocess.PreprocessQueue([preprocess.Center()], X)
    preprocY = preprocess.PreprocessQueue([preprocess.Center()], Y)
    X = preprocX.process(X)
    Y = preprocY.process(Y)

#    print X
#    print Y

    o2pls = methods.O2PLS(num_comp=[3, 2, 2])
    o2pls.set_tolerance(5e-12)
    o2pls.fit(X, Y)

    Xhat = np.dot(o2pls.T, o2pls.P.T) + np.dot(o2pls.To, o2pls.Po.T)
    assert_array_almost_equal(X, Xhat, decimal=5, err_msg="O2PLS does not" \
            " give a correct reconstruction of X")
    Yhat = np.dot(o2pls.U, o2pls.Q.T) + np.dot(o2pls.Uo, o2pls.Qo.T)
    assert_array_almost_equal(Y, Yhat, decimal=5, err_msg="O2PLS does not" \
            " give a correct reconstruction of Y")

    W = np.asarray([[0.28574946790251, 0.80549463591904, 0.46033811220009],
                    [-0.8206917792876, 0.03104339272344, 0.41068049575333],
                    [0.23145983182932, -0.4575178927710, 0.78402240214039],
                    [-0.0476345720024, 0.32258037076527, -0.0472056956547],
                    [0.43470626726892, -0.1919218108116, 0.05010836360956]])
    W, o2pls.W = direct(W, o2pls.W, compare=True)
    assert_array_almost_equal(W, o2pls.W, decimal=5, err_msg="O2PLS does not" \
            " give the correct weights in X")

    C = np.asarray([[0.33983623057749, -0.0864374467592, 0.61598479965533],
                    [-0.2152253235407, 0.76365073948393, -0.0489442226022],
                    [-0.8064484759235, -0.4011334093529, 0.35039634003963],
                    [-0.3919315842703, 0.44086245939825, 0.16315472674870],
                    [-0.1849861763096, -0.2325906182065, -0.6846678973735]])
    C, o2pls.C = direct(C, o2pls.C, compare=True)
    assert_array_almost_equal(C, o2pls.C, decimal=5, err_msg="O2PLS does not" \
            " give the correct weights in Y")

    Wo = np.asarray([[0.09275972900527, 0.22138222202790],
                     [-0.3237766587847, 0.22806033590264],
                     [0.13719457299956, -0.3218543865251],
                     [-0.4806324686954, -0.8126726919742],
                     [-0.7979563816879, 0.36735710766490]])
    Wo, o2pls.Wo = direct(Wo, o2pls.Wo, compare=True)
    assert_array_almost_equal(Wo, o2pls.Wo, decimal=5, err_msg="O2PLS does " \
            "not give the correct unique weights in X")

    Co = np.asarray([[-0.2860251435011, -0.6448195468993],
                     [0.41206605980471, -0.4453331714833],
                     [0.21923094429525, -0.1337648740582],
                     [-0.7577522025752, 0.22632291041065],
                     [-0.3551627404418, -0.5628241439426]])
    Co, o2pls.Co = direct(Co, o2pls.Co, compare=True)
    assert_array_almost_equal(Co, o2pls.Co, decimal=5, err_msg="O2PLS does " \
            "not give the correct unique weights in Y")

    # O2PLS with random dataset, compare to known solution on the data
    np.random.seed(43)

    # Test dual version
    X = np.random.rand(5, 10)
    Y = np.random.rand(5, 10)

    preprocX = preprocess.PreprocessQueue([preprocess.Center()], X)
    preprocY = preprocess.PreprocessQueue([preprocess.Center()], Y)
    X = preprocX.process(X)
    Y = preprocY.process(Y)

    o2pls = methods.O2PLS(num_comp=[3, 2, 2])
    o2pls.set_tolerance(5e-12)
    o2pls.fit(X, Y)

    Xhat = np.dot(o2pls.T, o2pls.P.T) + np.dot(o2pls.To, o2pls.Po.T)
    assert_array_almost_equal(X, Xhat, decimal=5, err_msg="O2PLS does not" \
            " give a correct reconstruction of X")
    Yhat = np.dot(o2pls.U, o2pls.Q.T) + np.dot(o2pls.Uo, o2pls.Qo.T)
    assert_array_almost_equal(Y, Yhat, decimal=5, err_msg="O2PLS does not" \
            " give a correct reconstruction of Y")

    W = np.asarray([[0.660803597278207, 0.283434091506369, -0.10111733853777],
                    [0.199245987632961, -0.37178287065057, -0.10558062870063],
                    [0.049056826578120, 0.510007873722491, -0.05258764260954],
                    [0.227075734276342, 0.321380074146550, 0.386973037543944],
                    [-0.36636167268872, 0.047162463451118, -0.51320512393785],
                    [-0.23102837136786, -0.05368842048677, 0.613919597967758],
                    [-0.44150306809316, 0.531948476266181, 0.122002207828563],
                    [0.155295133662634, -0.24320395140635, 0.327486426731878],
                    [0.177637796765300, 0.210455027475178, 0.022448447240951],
                    [0.177420328030713, 0.162892673626251, -0.25139972066789]])
    W, o2pls.W = direct(W, o2pls.W, compare=True)
    assert_array_almost_equal(W, o2pls.W, decimal=5, err_msg="O2PLS does not" \
            " give the correct weights in X")

    C = np.asarray([[0.029463187629622, 0.343448566391335, 0.439488969936543],
                    [-0.30851463695679, 0.567738611008308, 0.288894827826079],
                    [-0.04136803272166, 0.240281903256231, -0.16506192502899],
                    [0.507488061231666, 0.220214013827959, -0.19774980994540],
                    [-0.03283135387270, -0.45687225108843, 0.330712836574687],
                    [-0.08846573694626, 0.137459131512888, -0.19971688576644],
                    [0.575982215268074, 0.265756602507649, 0.284152520405287],
                    [-0.17475865049036, -0.23679836102887, 0.559859131854712],
                    [0.474892087711923, -0.15622059920436, 0.286373509842812],
                    [-0.21902628912052, 0.273412086540475, 0.177725330447690]])
    C, o2pls.C = direct(C, o2pls.C, compare=True)
    assert_array_almost_equal(C, o2pls.C, decimal=5, err_msg="O2PLS does not" \
            " give the correct weights in Y")

    Wo = np.asarray([[-0.07261963606155, 0],
                     [-0.18613960916330, 0],
                     [-0.19334537702829, 0],
                     [-0.41655633915778, 0],
                     [-0.55615535411133, 0],
                     [0.042355493010538, 0],
                     [0.170141007666872, 0],
                     [-0.15834677472031, 0],
                     [-0.07965438708092, 0],
                     [0.614579177338253, 0]])
    Wo, o2pls.Wo = direct(Wo, o2pls.Wo, compare=True)
    assert_array_almost_equal(Wo, o2pls.Wo, decimal=5, err_msg="O2PLS does " \
            "not give the correct unique weights in X")

    Co = np.asarray([[-0.14337088725157, 0],
                     [0.268379847579299, 0],
                     [0.538254868418912, 0],
                     [-0.07676572699854, 0],
                     [0.011841641465690, 0],
                     [-0.68216820966895, 0],
                     [-0.04763233629597, 0],
                     [-0.03243171228586, 0],
                     [0.059518204830295, 0],
                     [-0.37342871207122, 0]])
    Co, o2pls.Co = direct(Co, o2pls.Co, compare=True)
    assert_array_almost_equal(Co, o2pls.Co, decimal=5, err_msg="O2PLS does " \
            "not give the correct unique weights in Y")


if __name__ == "__main__":

    test()