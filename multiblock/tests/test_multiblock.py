import numpy as np
from numpy import dot
import multiblock.utils as utils
from multiblock.utils import direct, TOLERANCE, MAX_ITER, corr, cov, norm, normI
from multiblock.utils import rand, zeros, optimal_shrinkage
from multiblock.utils.testing import assert_array_almost_equal
from multiblock import *
import multiblock.start_vectors as start_vectors
import multiblock.prox_ops as prox_ops
import multiblock.schemes as schemes
import multiblock.error_functions as error_functions
from sklearn.datasets import load_linnerud
from numpy import ones, eye
from numpy.linalg import eig
import scipy.sparse as sparse
from time import time

from math import log
from sklearn.pls import PLSRegression
from sklearn.pls import PLSCanonical
#from sklearn.pls import CCA
#from sklearn.pls import PLSSVD
#from sklearn.pls import _center_scale_xy

# this is a dumb comment 
# not as dumb as this! ;-)


def check_ortho(M, err_msg):
    K = np.dot(M.T, M)
    assert_array_almost_equal(K, np.diag(np.diag(K)), err_msg=err_msg)


def test_multiblock():

    test_rgcca()

#    n = 10
#    X, Y = orth_matrix(n)
#    SSX = np.sum(X**2)
#    SSY = np.sum(Y**2)
#    tol = 5e-12
#    miter = 1000
#    center = True
#    scale  = False
#    num_comp = 1
#    l = 0.5
#
#    nonzeroW = []
#    nonzeroWo = []
#    so2pls4 = O2PLS(num_comp = [num_comp, 1, 0], center = center, scale = scale,
#                    tolerance = tol, max_iter = miter,
#                    prox_op = prox_op.L0_by_count([1,1], [100,100], normaliser = [norm, norm]))
#    so2pls4.fit(X, Y)
#    Yhat4    = so2pls4.predict(X)
#    SSYdiff4 = np.sum((Y-Yhat4)**2)
#    R2Yhat4  = 1 - (SSYdiff4 / SSY)
#    nonzeroW.append(np.count_nonzero(so2pls4.W[:,0]))
#    if so2pls4.Wo.shape[1] > 0:
#        nonzeroWo.append(np.count_nonzero(so2pls4.Wo[:,0]))
#    else:
#        nonzeroWo.append(0)
#    print "sO2PLS: n = %3d, R2Yhat = %.6f, num_comp = %d, nonzeroW: %2d, nonzeroWo: %2d" \
#            % (n, R2Yhat4, num_comp, nonzeroW[-1], nonzeroWo[-1])
#
#    print so2pls4.W
#    print so2pls4.Wo
#    print so2pls4.To

#    for l in [0.00, 0.10, 0.20, 0.30, 0.40]:
#        so2pls = O2PLS(num_comp = [num_comp, 2, 0], center = center, scale = scale,
#                       tolerance = tol, max_iter = miter,
#                       prox_op = prox_op.L1([l, l], [0, 0]))
#        so2pls.fit(X, Y)
#        Yhat = so2pls.predict(X)
#        SSYdiff = np.sum((Y-Yhat)**2)
#        R2Yhat = 1 - (SSYdiff / SSY)
#        print
#        print "sO2PLS: l=%.2f, R2Yhat=%.6f, num_comp=%d" % (l, R2Yhat, num_comp)
#
#        print so2pls.W
##        print so2pls.T
##        print so2pls.P
#
##        print "Transform:"
##        print so2pls.transform(X)
#        print
#        print so2pls.Wo
##        print so2pls.To
##        print so2pls.Po
##        print Y
##        print Yhat


#def test_eigsym():
#
#    d = load_linnerud()
#    X = d.data
#
#    n = 3
#    X = dot(X.T, X)
#
#    eig = EIGSym(num_comp = n, tolerance = 5e-12)
#    eig.fit(X)
#    Xhat = dot(eig.V, dot(eig.D, eig.V.T))
#
#    assert_array_almost_equal(X, Xhat, decimal=4, err_msg="EIGSym does not" \
#            " give the correct reconstruction of the matrix")
#
#    [D,V] = np.linalg.eig(X)
#    # linalg.eig does not return the eigenvalues in order, so need to sort
#    idx = np.argsort(D, axis=None).tolist()[::-1]
#    D = D[idx]
#    V = V[:,idx]
#
#    Xhat = dot(V, dot(np.diag(D), V.T))
#
#    V, eig.V = direct(V, eig.V, compare = True)
#    assert_array_almost_equal(V, eig.V, decimal=5, err_msg="EIGSym does not" \
#            " give the correct eigenvectors")


def test_SVD_PCA():

    # Assure same answer every time
    np.random.seed(42)

    # Compare SVD with and without sparsity constraint to numpy.linalg.svd
    Xtr = np.random.rand(6, 6)
    #Xte = np.random.rand(2,6)
    num_comp = 3
    tol = 5e-12

    preproc = preprocess.PreprocessQueue([preprocess.Center(),
                                          preprocess.Scale()], Xtr)
    Xtr = preproc.process(Xtr)

    for st in [0.1, 0.01, 0.001, 0.0001, 0]:
        svd = SVD(num_comp=num_comp)
        svd.set_prox_op(prox_ops.L1(st))
        svd.set_tolerance(tol)
        svd.set_max_iter(1000)
        svd.fit(Xtr)

        U, S, V = np.linalg.svd(Xtr)
        V = V.T
        S = np.diag(S)
        U = U[:, 0:num_comp]
        S = S[:, 0:num_comp]
        V = V[:, 0:num_comp]

        if st < tol:
            num_decimals = 5
        else:
            num_decimals = int(log(1. / st, 10) + 0.5)
        svd.V, V = direct(svd.V, V, compare=True)
        assert_array_almost_equal(svd.V, V, decimal=num_decimals - 2,
                err_msg="sklearn.NIPALS.SVD and numpy.linalg.svd " \
                "implementations lead to different loadings")

    # Compare PCA with sparsity constraint to numpy.linalg.svd
    Xtr = np.random.rand(5, 5)
    num_comp = 5
    tol = 5e-9

    preproc = preprocess.PreprocessQueue([preprocess.Center(),
                                          preprocess.Scale()], Xtr)
    Xtr = preproc.process(Xtr)

    for st in [0.1, 0.01, 0.001, 0.0001, 0]:
        pca = PCA(num_comp=num_comp)
        svd.set_prox_op(prox_ops.L1(st))
        svd.set_tolerance(tol)
        svd.set_max_iter(1000)
        pca.fit(Xtr)

        Tte = pca.transform(Xtr)
        U, S, V = np.linalg.svd(Xtr)
        V = V.T
        US = dot(U, np.diag(S))
        US = US[:, 0:num_comp]
        V = V[:, 0:num_comp]

        if st < tol:
            num_decimals = 5
        else:
            num_decimals = int(log(1. / st, 10) + 0.5)
        assert_array_almost_equal(Xtr, dot(pca.T, pca.P.T),
                                  decimal=num_decimals - 1,
                                  err_msg="Model does not equal the matrices")

    # Compare PCA without the sparsity constraint to numpy.linalg.svd
    Xtr = np.random.rand(50, 50)
    Xte = np.random.rand(20, 50)
    num_comp = 3

    preproc = preprocess.PreprocessQueue([preprocess.Center(),
                                          preprocess.Scale()], Xtr)
    Xtr = preproc.process(Xtr)
    Xte = preproc.process(Xte)

    pca = PCA(num_comp=num_comp)
    pca.set_tolerance(tol)
    pca.set_max_iter(1000)
    pca.fit(Xtr)

    pca.P, pca.T = direct(pca.P, pca.T)
    Tte = pca.transform(Xte)

    U, S, V = np.linalg.svd(Xtr)
    V = V.T
    US = dot(U, np.diag(S))
    US = US[:, 0:num_comp]
    V = V[:, 0:num_comp]
    V, US = direct(V, US)
    SVDte = dot(Xte, V)

    assert_array_almost_equal(pca.P, V, decimal=2, err_msg="NIPALS PCA and "
            "numpy.linalg.svd implementations lead to different loadings")

    assert_array_almost_equal(pca.T, US, decimal=2, err_msg="NIPALS PCA and "
            "numpy.linalg.svd implementations lead to different scores")

    assert_array_almost_equal(Tte, SVDte, decimal=2, err_msg="NIPALS PCA and "
            "numpy.linalg.svd implementations lead to different scores")

    # Compare PCA without the sparsity constraint to numpy.linalg.svd
    X = np.random.rand(50, 100)
    num_comp = 50

    preproc = preprocess.PreprocessQueue([preprocess.Center(),
                                          preprocess.Scale()], X)
    X = preproc.process(X)

    pca = PCA(num_comp=num_comp)
    pca.set_tolerance(tol)
    pca.set_max_iter(1000)
    pca.fit(X)

    Xhat_1 = dot(pca.T, pca.P.T)

    U, S, V = np.linalg.svd(X, full_matrices=False)
    Xhat_2 = dot(U, dot(np.diag(S), V))

    assert_array_almost_equal(X, Xhat_1, decimal=2, err_msg="PCA performs "
            " a faulty reconstruction of X")

    assert_array_almost_equal(Xhat_1, Xhat_2, decimal=2, err_msg="PCA and " \
            "numpy.linalg.svd implementations lead to different " \
            "reconstructions")

    # Compare PCA without the sparsity constraint to numpy.linalg.svd
    X = np.random.rand(100, 50)
    num_comp = 50

    preproc = preprocess.PreprocessQueue([preprocess.Center(),
                                          preprocess.Scale()], X)
    X = preproc.process(X)

    pca = PCA(num_comp=num_comp)
    pca.set_prox_op(prox_ops.L1(st))
    pca.set_tolerance(tol)
    pca.set_max_iter(1500)
    pca.fit(X)
    Xhat_1 = dot(pca.T, pca.P.T)

    U, S, V = np.linalg.svd(X, full_matrices=False)
    Xhat_2 = dot(U, dot(np.diag(S), V))

    assert_array_almost_equal(X, Xhat_1, decimal=2, err_msg="PCA performs a " \
        "faulty reconstruction of X")

    assert_array_almost_equal(Xhat_1, Xhat_2, decimal=2, err_msg="PCA and "
            "numpy.linalg.svd implementations lead to different " \
            "reconstructions")


def test_predictions():

    d = load_linnerud()
    Xorig = d.data
    Yorig = d.target
    tol = 5e-12
    miter = 1000
    num_comp = 2
    SSY = np.sum(Yorig ** 2)
#    SSX = np.sum(Xorig ** 2)
#    center = True
    scale = False

    if scale:
        preprocX = preprocess.PreprocessQueue([preprocess.Center(),
                                               preprocess.Scale()], Xorig)
        preprocY = preprocess.PreprocessQueue([preprocess.Center(),
                                               preprocess.Scale()], Yorig)
    else:
        preprocX = preprocess.PreprocessQueue([preprocess.Center()], Xorig)
        preprocY = preprocess.PreprocessQueue([preprocess.Center()], Yorig)

    X = Xorig.copy()
    Y = Yorig.copy()
    pls1 = PLSRegression(n_components=num_comp, scale=scale,
                 tol=tol, max_iter=miter, copy=True)
    pls1.fit(X, Y)
    Yhat1 = pls1.predict(X)

    SSYdiff1 = np.sum((Y - Yhat1) ** 2)
    utils.debug("PLSRegression: R2Yhat = %.4f" % (1 - (SSYdiff1 / SSY)))

    # Compare sklearn.PLSRegression and PLSR
    X = preprocX.process(Xorig)
    Y = preprocY.process(Yorig)
    pls3 = PLSR(num_comp=num_comp)
    pls3.set_max_iter(miter)
    pls3.set_tolerance(tol)

    pls3.fit(X, Y)
    Yhat3 = pls3.predict(X)
    Yhat3 = preprocY.revert(Yhat3)

    SSYdiff3 = np.sum((Yorig - Yhat3) ** 2)
    utils.debug("PLSR         : R2Yhat = %.4f" % (1 - (SSYdiff3 / SSY)))

    assert abs(SSYdiff1 - SSYdiff3) < 0.00005

    assert_array_almost_equal(Yhat1, Yhat3, decimal=5,
            err_msg="PLSR gives wrong prediction")

    # Compare sklearn.PLSCanonical and PLSC
    X = Xorig.copy()
    Y = Yorig.copy()
    pls2 = PLSCanonical(n_components=num_comp, scale=scale,
                        tol=tol, max_iter=miter, copy=True)
    pls2.fit(X, Y)
    Yhat2 = pls2.predict(X)

    SSYdiff2 = np.sum((Yorig - Yhat2) ** 2)
    utils.debug("PLSCanonical : R2Yhat = %.4f" % (1 - (SSYdiff2 / SSY)))

    # Compare PLSC and sklearn.PLSCanonical
    X = preprocX.process(Xorig)
    Y = preprocY.process(Yorig)
    pls4 = PLSC(num_comp=num_comp)
    pls4.set_max_iter(miter)
    pls4.set_tolerance(tol)
    pls4.fit(X, Y)

    Yhat4 = pls4.predict(X)
    Yhat4 = preprocY.revert(Yhat4)

    SSYdiff4 = np.sum((Yorig - Yhat4) ** 2)
    utils.debug("PLSC         : R2Yhat = %.4f" % (1 - (SSYdiff4 / SSY)))

    # Compare O2PLS and sklearn.PLSCanonical
    X = preprocX.process(Xorig)
    Y = preprocY.process(Yorig)
    pls5 = O2PLS(num_comp=[num_comp, 1, 0])
    pls5.set_max_iter(miter)
    pls5.set_tolerance(tol)
    pls5.fit(X, Y)

    Yhat5 = pls5.predict(X)
    Yhat5 = preprocY.revert(Yhat5)

    SSYdiff5 = np.sum((Yorig - Yhat5) ** 2)
    utils.debug("O2PLS X-Y    : R2Yhat = %.4f" % (1 - (SSYdiff5 / SSY)))

    assert SSYdiff2 > SSYdiff4
    assert SSYdiff2 > SSYdiff5

    # Make sure O2PLS is symmetric!
    X = preprocX.process(Xorig)
    Y = preprocY.process(Yorig)
    pls6 = O2PLS(num_comp=[num_comp, 0, 1])
    pls6.set_max_iter(miter)
    pls6.set_tolerance(tol)
    pls6.fit(Y, X)

    Yhat6 = pls6.predict(Y=X)
    Yhat6 = preprocY.revert(Yhat6)

    pls5.W, pls6.C = direct(pls5.W, pls6.C, compare=True)
    assert_array_almost_equal(pls5.W, pls6.C, decimal=5,
            err_msg="O2PLS is not symmetic")
    pls5.T, pls6.U = direct(pls5.T, pls6.U, compare=True)
    assert_array_almost_equal(pls5.T, pls6.U, decimal=5,
            err_msg="O2PLS is not symmetic")
    pls5.P, pls6.Q = direct(pls5.P, pls6.Q, compare=True)
    assert_array_almost_equal(pls5.P, pls6.Q, decimal=5,
            err_msg="O2PLS is not symmetic")

    pls5.C, pls6.W = direct(pls5.C, pls6.W, compare=True)
    assert_array_almost_equal(pls5.C, pls6.W, decimal=5,
            err_msg="O2PLS is not symmetic")
    pls5.U, pls6.T = direct(pls5.U, pls6.T, compare=True)
    assert_array_almost_equal(pls5.U, pls6.T, decimal=5,
            err_msg="O2PLS is not symmetic")
    pls5.Q, pls6.P = direct(pls5.Q, pls6.P, compare=True)
    assert_array_almost_equal(pls5.Q, pls6.P, decimal=5,
            err_msg="O2PLS is not symmetic")

    assert_array_almost_equal(pls5.Bx, pls6.By, decimal=5,
            err_msg="O2PLS is not symmetic")
    assert_array_almost_equal(pls5.By, pls6.Bx, decimal=5,
            err_msg="O2PLS is not symmetic")

    SSYdiff6 = np.sum((Yorig - Yhat6) ** 2)
    utils.debug("O2PLS Y-X    : R2Yhat = %.4f" % (1 - (SSYdiff6 / SSY)))

    assert abs(SSYdiff6 - SSYdiff5) < TOLERANCE


def test_o2pls():

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

    X = dot(t, p.T) + dot(to, po.T)
    Y = dot(t, q.T) + dot(uo, qo.T)

    svd = SVD(num_comp=1)
    svd.fit(dot(X.T, Y))
    t_svd = dot(X, svd.U)
    u_svd = dot(Y, svd.V)

    o2pls = O2PLS(num_comp=[1, 1, 1])
    o2pls.fit(X, Y)

    Xhat = dot(o2pls.T, o2pls.P.T) + dot(o2pls.To, o2pls.Po.T)
    assert_array_almost_equal(X, Xhat, decimal=5, err_msg="O2PLS does not" \
            " give a correct reconstruction of X")
    Yhat = dot(o2pls.U, o2pls.Q.T) + dot(o2pls.Uo, o2pls.Qo.T)
    assert_array_almost_equal(Y, Yhat, decimal=5, err_msg="O2PLS does not" \
            " give a correct reconstruction of Y")

    assert np.abs(corr(o2pls.T, o2pls.U)) > np.abs(corr(t_svd, u_svd))
    assert np.abs(corr(o2pls.T, o2pls.U)) > np.abs(corr(t_svd, u_svd))
    assert np.abs(corr(o2pls.T, t)) > np.abs(corr(t_svd, t))
    assert np.abs(corr(o2pls.U, t)) > np.abs(corr(u_svd, t))

    assert ((p > TOLERANCE) == (np.abs(o2pls.W) > TOLERANCE)).all()
    assert ((p > TOLERANCE) == (np.abs(o2pls.P) > TOLERANCE)).all()
    assert ((po > TOLERANCE) == (np.abs(o2pls.Po) > TOLERANCE)).all()
    assert dot(o2pls.W.T, o2pls.Wo) < TOLERANCE

    assert ((q > TOLERANCE) == (np.abs(o2pls.C) > TOLERANCE)).all()
    assert ((q > TOLERANCE) == (np.abs(o2pls.Q) > TOLERANCE)).all()
    assert ((qo > TOLERANCE) == (np.abs(o2pls.Qo) > TOLERANCE)).all()
    assert dot(o2pls.C.T, o2pls.Co) < TOLERANCE

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

    o2pls = O2PLS(num_comp=[3, 2, 2])
    o2pls.set_tolerance(5e-12)
    o2pls.fit(X, Y)

    Xhat = dot(o2pls.T, o2pls.P.T) + dot(o2pls.To, o2pls.Po.T)
    assert_array_almost_equal(X, Xhat, decimal=5, err_msg="O2PLS does not" \
            " give a correct reconstruction of X")
    Yhat = dot(o2pls.U, o2pls.Q.T) + dot(o2pls.Uo, o2pls.Qo.T)
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

    o2pls = O2PLS(num_comp=[3, 2, 2])
    o2pls.set_tolerance(5e-12)
    o2pls.fit(X, Y)

    Xhat = dot(o2pls.T, o2pls.P.T) + dot(o2pls.To, o2pls.Po.T)
    assert_array_almost_equal(X, Xhat, decimal=5, err_msg="O2PLS does not" \
            " give a correct reconstruction of X")
    Yhat = dot(o2pls.U, o2pls.Q.T) + dot(o2pls.Uo, o2pls.Qo.T)
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


def test_regularisation():

    d = load_linnerud()
    Xorig = d.data
    Yorig = d.target
#    print X.shape
#    print Y.shape
    tol = 5e-12
    miter = 1000
#    Xorig = X.copy()
#    Yorig = Y.copy()
    center = True
    scale = True
    inf = 2 ** 30
    SSY = np.sum(Yorig ** 2)
    num_comp = 2

    preprocX = preprocess.PreprocessQueue([], Xorig)
    preprocY = preprocess.PreprocessQueue([], Yorig)
    if center:
        preprocX.push(preprocess.Center())
        preprocY.push(preprocess.Center())
    if scale:
        preprocX.push(preprocess.Scale())
        preprocY.push(preprocess.Scale())

    # Test first with PLSR
    X = preprocX.process(Xorig)
    Y = preprocY.process(Yorig)
    pls = PLSR(num_comp=num_comp)
    pls.set_tolerance(tol)
    pls.set_max_iter(miter)
    pls.fit(X, Y)
    Yhat = pls.predict(X)
    Yhat = preprocY.revert(Yhat)
    SSYdiff = np.sum((Yorig - Yhat) ** 2)
    R2Yhat = (1 - (SSYdiff / SSY))
    utils.debug("PLS : R2Yhat = %.6f" % R2Yhat)

    # Test sPLS methods when keeping all variables
    spls1 = PLSR(num_comp=num_comp)
    spls1.set_tolerance(tol)
    spls1.set_max_iter(miter)
    spls1.set_prox_op(prox_ops.L1(0., 0., normaliser=[norm, normI]))
    spls1.fit(X, Y)
    Yhat1 = spls1.predict(X)
    Yhat1 = preprocY.revert(Yhat1)
    SSYdiff1 = np.sum((Yorig - Yhat1) ** 2)
    utils.debug("sPLS: R2Yhat = %.6f" % (1 - (SSYdiff1 / SSY)))
    assert abs(R2Yhat - (1 - (SSYdiff1 / SSY))) < TOLERANCE
    assert_array_almost_equal(Yhat, Yhat1, decimal=5,
            err_msg="Sparse PLS with no thresholding does not give correct " \
                    "result")

#    spls2 = PLSR(num_comp=num_comp)
##    alg = spls2.get_algorithm()
#    spls2.set_tolerance(tol)
#    spls2.set_max_iter(miter)
#    spls2.set_prox_op(prox_ops.L1_binsearch(float('Inf'), float('Inf'),
#                      normaliser=[norm, normI]))
#    spls2.fit(X, Y)
#    Yhat2 = spls2.predict(X)
#    Yhat2 = preprocY.revert(Yhat2)
#    SSYdiff2 = np.sum((Yorig - Yhat2) ** 2)
#    utils.debug("sPLS: R2Yhat = %.6f" % (1 - (SSYdiff2 / SSY)))
#    assert abs(R2Yhat - (1 - (SSYdiff1 / SSY))) < TOLERANCE
#    assert_array_almost_equal(Yhat, Yhat2, decimal=5,
#            err_msg="Sparse PLS with no thresholding does not give correct " \
#                    "result")
#
#    spls3 = PLSR(num_comp=num_comp)
#    alg = spls3.get_algorithm()
#    alg.set_tolerance(tol)
#    alg.set_max_iter(miter)
#    alg.set_prox_op(prox_ops.L0_binsearch(inf, inf, normaliser=[norm, normI]))
#    spls3.fit(X, Y)
#    Yhat3 = spls3.predict(X)
#    Yhat3 = preprocY.revert(Yhat3)
#    SSYdiff3 = np.sum((Yorig - Yhat3) ** 2)
#    utils.debug("sPLS: R2Yhat = %.6f" % (1 - (SSYdiff3 / SSY)))
#    assert_array_almost_equal(Yhat, Yhat3, decimal=5,
#            err_msg="Sparse PLS with no thresholding does not give correct " \
#                    "result")
#
#    spls4 = PLSR(num_comp=num_comp)
#    alg = spls4.get_algorithm()
#    alg.set_tolerance(tol)
#    alg.set_max_iter(miter)
#    alg.set_prox_op(prox_ops.L0_by_count(inf, inf, normaliser=[norm, normI]))
#    spls4.fit(X, Y)
#    Yhat4 = spls4.predict(X)
#    Yhat4 = preprocY.revert(Yhat4)
#    SSYdiff4 = np.sum((Yorig - Yhat4) ** 2)
#    utils.debug("sPLS: R2Yhat = %.6f" % (1 - (SSYdiff4 / SSY)))
#    assert_array_almost_equal(Yhat, Yhat4, decimal=5,
#            err_msg="Sparse PLS with no thresholding does not give correct " \
#                    "result")

    # Create a matrix X (10,11) with variables with
    # correlation 1 throught 0 to a single y variable
#    np.random.seed(38) # 15, 22, 32, 38, 40
    n_sz = 10
    Xorig, Yorig = orth_matrix(n_sz)
    SSX = np.sum(Xorig ** 2)
    SSY = np.sum(Yorig ** 2)

    preprocX = preprocess.PreprocessQueue([], Xorig)
    preprocY = preprocess.PreprocessQueue([], Yorig)
    if center:
        preprocX.push(preprocess.Center())
        preprocY.push(preprocess.Center())
    if scale:
        preprocX.push(preprocess.Scale())
        preprocY.push(preprocess.Scale())
    X = preprocX.process(Xorig)
    Y = preprocY.process(Yorig)

    utils.debug()
    num_comp = n_sz - 1
    # Analyse with PLSR
    pls = PLSR(num_comp=num_comp)
    pls.set_tolerance(tol)
    pls.set_max_iter(miter)
    pls.fit(X, Y)
    Yhat = pls.predict(X)
    Yhat = preprocY.revert(Yhat)
    SSYdiff = np.sum((Yorig - Yhat) ** 2)
    R2Yhat = (1 - (SSYdiff / SSY))
    utils.debug("PLS :         R2Yhat=%.6f, num_comp=%d" % (R2Yhat, num_comp))

    num_comp = 1
    # Analyse with PLSR
    pls = PLSR(num_comp=num_comp)
    pls.set_tolerance(tol)
    pls.set_max_iter(miter)
    pls.fit(X, Y)
    Yhat = pls.predict(X)
    Yhat = preprocY.revert(Yhat)
    SSYdiff = np.sum((Yorig - Yhat) ** 2)
    R2Yhat = (1 - (SSYdiff / SSY))
    utils.debug("PLS :         R2Yhat=%.6f, num_comp=%d" % (R2Yhat, num_comp))

    # Analyse with Sparse PLSR (L1)
    nonzero = []
    for l in np.linspace(0, 0.9, 10).tolist():
        spls1 = PLSR(num_comp=num_comp)
#        alg = spls1.get_algorithm()
        spls1.set_tolerance(tol)
        spls1.set_max_iter(miter)
        spls1.set_prox_op(prox_ops.L1(l, 0, normaliser=[norm, normI]))
        spls1.fit(X, Y)
        Yhat1 = spls1.predict(X)
        Yhat1 = preprocY.revert(Yhat1)
        SSYdiff1 = np.sum((Yorig - Yhat1) ** 2)
        R2Yhat1 = 1 - (SSYdiff1 / SSY)
        nonzero.append(np.count_nonzero(spls1.W))
        utils.debug("sPLS: l=%.2f, R2Yhat=%.6f, num_comp=%d, nonzero=%d" \
                % (l, R2Yhat1, num_comp, nonzero[-1]))

        assert all(x <= y for x, y in zip(np.abs(spls1.W)[:, 0],
                                          (np.abs(spls1.W)[:, 0])[1:]))
        assert all(x >= y for x, y in zip(nonzero, nonzero[1:]))
    assert abs(R2Yhat1 - 1) < TOLERANCE

#    # Analyse with Sparse PLSR (L1_binsearch)
#    utils.debug()
#    utils.debug("PLS :         R2Yhat=%.6f, num_comp=%d" % (R2Yhat, num_comp))
#    nonzero = []
#    for s in [float('Inf'), 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.25, 0.125, 0.0625,
#              0.03125, 0]:
#        spls2 = PLSR(num_comp=num_comp)
#        alg = spls2.get_algorithm()
#        alg.set_tolerance(tol)
#        alg.set_max_iter(miter)
#        alg.set_prox_op(prox_ops.L1_binsearch(s, float('Inf'),
#                                              normaliser=[norm, normI]))
#        spls2.fit(X, Y)
#        Yhat2 = spls2.predict(X)
#        Yhat2 = preprocY.revert(Yhat2)
#        SSYdiff2 = np.sum((Yorig - Yhat2) ** 2)
#        R2Yhat2 = 1 - (SSYdiff2 / SSY)
#        nonzero.append(np.count_nonzero(spls2.W))
#        utils.debug("sPLS: s=%-4.2f, R2Yhat=%.6f, num_comp=%d, nonzero=%d" \
#                % (s, R2Yhat2, num_comp, nonzero[-1]))
#
#        assert all(x <= y for x, y in zip(np.abs(spls2.W)[:, 0],
#                                          (np.abs(spls2.W)[:, 0])[1:]))
#        assert all(x >= y for x, y in zip(nonzero, nonzero[1:]))
#    assert abs(R2Yhat2 - 1) < TOLERANCE
#
#    # Analyse with Sparse PLSR (L0_binsearch)
#    utils.debug()
#    utils.debug("PLS :        R2Yhat=%.6f, num_comp=%d" % (R2Yhat, num_comp))
#    nonzero = []
#    for n in [100, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]:
#        spls3 = PLSR(num_comp=num_comp)
#        alg = spls3.get_algorithm()
#        alg.set_tolerance(tol)
#        alg.set_max_iter(miter)
#        alg.set_prox_op(prox_ops.L0_binsearch(n, 100,
#                                              normaliser=[norm, normI]))
#        spls3.fit(X, Y)
#        Yhat3 = spls3.predict(X)
#        Yhat3 = preprocY.revert(Yhat3)
#        SSYdiff3 = np.sum((Yorig - Yhat3) ** 2)
#        R2Yhat3 = 1 - (SSYdiff3 / SSY)
#        nonzero.append(np.count_nonzero(spls3.W))
#        utils.debug("sPLS: n=%3d, R2Yhat=%.6f, num_comp=%d, nonzero=%d" \
#                % (n, R2Yhat3, num_comp, nonzero[-1]))
#
#        assert all(x <= y for x, y in zip(np.abs(spls3.W)[:, 0],
#                                          (np.abs(spls3.W)[:, 0])[1:]))
#        assert all(x >= y for x, y in zip(nonzero, nonzero[1:]))
#    assert abs(R2Yhat3 - 1) < TOLERANCE
#
#    # Analyse with Sparse PLSR (L0_by_count)
#    utils.debug()
#    utils.debug("PLS :        R2Yhat=%.6f, num_comp=%d" % (R2Yhat, num_comp))
#    nonzero = []
#    for n in [100] + range(11, -1, -1):
#        spls4 = PLSR(num_comp=num_comp)
#        alg = spls4.get_algorithm()
#        alg.set_tolerance(tol)
#        alg.set_max_iter(miter)
#        alg.set_prox_op(prox_ops.L0_by_count(n, 100, normaliser=[norm, normI]))
#        spls4.fit(X, Y)
#        Yhat4 = spls4.predict(X)
#        Yhat4 = preprocY.revert(Yhat4)
#        SSYdiff4 = np.sum((Yorig - Yhat4) ** 2)
#        R2Yhat4 = 1 - (SSYdiff4 / SSY)
#        nonzero.append(np.count_nonzero(spls4.W))
#        utils.debug("sPLS: n=%3d, R2Yhat=%.6f, num_comp=%d, nonzero=%d" \
#                % (n, R2Yhat4, num_comp, nonzero[-1]))
#
#        assert all(x <= y for x, y in zip(np.abs(spls4.W)[:, 0],
#                                          (np.abs(spls4.W)[:, 0])[1:]))
#        assert all(x >= y for x, y in zip(nonzero, nonzero[1:]))
#    assert abs(R2Yhat4 - 1) < TOLERANCE

    # Analyse with O2PLS
    o2pls = O2PLS(num_comp=[num_comp, 8, 0])
    o2pls.set_tolerance(tol)
    o2pls.set_max_iter(miter)
    o2pls.fit(X, Y)
    Yhat = o2pls.predict(X)
    Yhat = preprocY.revert(Yhat)
    SSYdiff = np.sum((Yorig - Yhat) ** 2)
    R2Yhat = (1 - (SSYdiff / SSY))
    utils.debug()
    utils.debug("O2PLS :         R2Yhat=%.5f, num_comp=%d" \
            % (R2Yhat, num_comp))

    # Analyse with Sparse O2PLS (L1)
    nonzeroW = []
    nonzeroWo = []
    n_cp = 1
    for l in np.linspace(0, 0.9, 10).tolist():
        num_orth = max(n_sz - n_cp, 0)
        so2pls1 = O2PLS(num_comp=[num_comp, num_orth, 0])
        so2pls1.set_tolerance(tol)
        so2pls1.set_max_iter(miter)
        so2pls1.set_prox_op(prox_ops.L1([l, l], [0, 0]))
        so2pls1.fit(X, Y)
        Yhat1 = so2pls1.predict(X)
        Yhat1 = preprocY.revert(Yhat1)
        SSYdiff1 = np.sum((Yorig - Yhat1) ** 2)
        R2Yhat1 = 1 - (SSYdiff1 / SSY)
        nonzeroW.append(np.count_nonzero(so2pls1.W[:, [0]]))
        if so2pls1.Wo.shape[1] > 0:
            nonzeroWo.append(np.count_nonzero(so2pls1.Wo[:, [0]]))
        else:
            nonzeroWo.append(0)
        utils.debug("sO2PLS: l=%.2f, R2Yhat=%.5f, num_comp=%d, num_orth=%2d, "\
              "nonzeroW=%2d, nonzeroWo=%2d" \
              % (l, R2Yhat1, num_comp, num_orth, nonzeroW[-1], nonzeroWo[-1]))
        n_cp += 1

#        assert all(x <= y for x, y in zip(np.abs(so2pls1.W)[:,0], (np.abs(so2pls1.W)[:,0])[1:]))
#        assert all(x >= y for x, y in zip(nonzeroW, nonzeroW[1:]))
#        assert all(x <= y for x, y in zip(nonzeroWo, nonzeroWo[1:]))
    assert abs(R2Yhat1 - 1) < 0.0005  # TOLERANCE

#    # Analyse with Sparse O2PLS (L1_binsearch)
#    utils.debug()
#    utils.debug("O2PLS :         R2Yhat=%.5f, num_comp=%d" \
#            % (R2Yhat, num_comp))
#    nonzeroW = []
#    nonzeroWo = []
#    n_cp = 0
#    for s in [float('Inf'), 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.25, 0.125, 0.0625,
#              0.03125, 0]:
#        num_orth = max(n_sz - n_cp, 0)
#        so2pls2 = O2PLS(num_comp=[num_comp, num_orth, 0])
#        alg = so2pls2.get_algorithm()
#        alg.set_tolerance(tol)
#        alg.set_max_iter(miter)
#        alg.set_prox_op(prox_ops.L1_binsearch([s, s], [float('Inf')] * 2))
#        so2pls2.fit(X, Y)
#        Yhat2 = so2pls2.predict(X)
#        Yhat2 = preprocY.revert(Yhat2)
#        SSYdiff2 = np.sum((Yorig - Yhat2) ** 2)
#        R2Yhat2 = 1 - (SSYdiff2 / SSY)
#        nonzeroW.append(np.count_nonzero(so2pls2.W[:, [0]]))
#        if so2pls2.Wo.shape[1] > 0:
#            nonzeroWo.append(np.count_nonzero(so2pls2.Wo[:, [0]]))
#        else:
#            nonzeroWo.append(0)
#        utils.debug("sO2PLS: s=%4.2f, R2Yhat=%.5f, num_comp=%d, " \
#                    "num_orth=%2d, nonzeroW=%2d, nonzeroWo=%2d" \
#                    % (s, R2Yhat2, num_comp, num_orth, nonzeroW[-1],
#                       nonzeroWo[-1]))
#        n_cp += 1
#
##        assert all(x <= y for x, y in zip(np.abs(so2pls2.W)[:,0], (np.abs(so2pls2.W)[:,0])[1:]))
##        assert all(x >= y for x, y in zip(nonzeroW, nonzeroW[1:]))
##        assert all(x <= y for x, y in zip(nonzeroWo, nonzeroWo[1:]))
#    assert abs(R2Yhat2 - 1) < 0.0005  # TOLERANCE
#
#    # Analyse with Sparse O2PLS (L0_binsearch)
#    utils.debug()
#    utils.debug("O2PLS :        R2Yhat=%.6f, num_comp=%d" % (R2Yhat, num_comp))
#    nonzeroW = []
#    nonzeroWo = []
#    n_cp = 0
#    for n in [100, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]:
#        num_orth = max(n_sz - n_cp, 0)
#        so2pls3 = O2PLS(num_comp=[num_comp, num_orth, 0])
#        alg = so2pls3.get_algorithm()
#        alg.set_tolerance(tol)
#        alg.set_max_iter(miter)
#        alg.set_prox_op(prox_ops.L0_binsearch([n, n], [100, 100]))
#        so2pls3.fit(X, Y)
#        Yhat3 = so2pls3.predict(X)
#        Yhat3 = preprocY.revert(Yhat3)
#        SSYdiff3 = np.sum((Yorig - Yhat3) ** 2)
#        R2Yhat3 = 1 - (SSYdiff3 / SSY)
#        nonzeroW.append(np.count_nonzero(so2pls3.W[:, 0]))
#        if so2pls3.Wo.shape[1] > 0:
#            nonzeroWo.append(np.count_nonzero(so2pls3.Wo[:, 0]))
#        else:
#            nonzeroWo.append(0)
#        utils.debug("sO2PLS: n=%3d, R2Yhat=%.6f, num_comp=%d, num_orth=%2d, " \
#                    "nonzeroW=%2d, nonzeroWo=%2d" \
#                    % (n, R2Yhat3, num_comp, num_orth, nonzeroW[-1],
#                       nonzeroWo[-1]))
#        n_cp += 1
#
##        assert all(x <= y for x, y in zip(np.abs(so2pls3.W)[:,0], (np.abs(so2pls3.W)[:,0])[1:]))
#        assert all(x >= y for x, y in zip(nonzeroW, nonzeroW[1:]))
##        assert all(x <= y for x, y in zip(nonzeroWo, nonzeroWo[1:]))
#    assert abs(R2Yhat3 - 1) < TOLERANCE
#
#    utils.debug()
#    utils.debug("O2PLS :          R2Yhat = %.6f, num_comp = %d" \
#            % (R2Yhat, num_comp))
#    nonzeroW = []
#    nonzeroWo = []
#    n_cp = 0
#    for n in [100, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]:
#        num_orth = max(n_sz - n_cp, 0)
#        so2pls4 = O2PLS(num_comp=[num_comp, num_orth, 0])
#        alg = so2pls4.get_algorithm()
#        alg.set_tolerance(tol)
#        alg.set_max_iter(miter)
#        alg.set_prox_op(prox_ops.L0_by_count([n, n], [100, 100],
#                                             normaliser=[norm, norm]))
#        so2pls4.fit(X, Y)
#        Yhat4 = so2pls4.predict(X)
#        Yhat4 = preprocY.revert(Yhat4)
#        SSYdiff4 = np.sum((Yorig - Yhat4) ** 2)
#        R2Yhat4 = 1 - (SSYdiff4 / SSY)
#        nonzeroW.append(np.count_nonzero(so2pls4.W[:, 0]))
#        if so2pls4.Wo.shape[1] > 0:
#            nonzeroWo.append(np.count_nonzero(so2pls4.Wo[:, 0]))
#        else:
#            nonzeroWo.append(0)
#        utils.debug("sO2PLS: n = %3d, R2Yhat = %.6f, num_comp = %d, " \
#                    "nonzeroW: %2d, nonzeroWo: %2d" \
#                    % (n, R2Yhat4, num_comp, nonzeroW[-1], nonzeroWo[-1]))
#        n_cp += 1
#
##        assert all(x <= y for x, y in zip(np.abs(so2pls4.W)[:,0], (np.abs(so2pls4.W)[:,0])[1:]))
#        assert all(x >= y for x, y in zip(nonzeroW, nonzeroW[1:]))
##        assert all(x <= y for x, y in zip(nonzeroWo, nonzeroWo[1:]))
#    assert abs(R2Yhat4 - 1) < TOLERANCE

    # Testing agreement of different runs by using Fleiss kappa
    utils.debug()
    np.random.seed(15)
    n = 10
    X, Y = orth_matrix(n)
    X = np.hstack((X, rand(n, n)))
    Y = np.hstack((Y, rand(n, n)))

    preprocX = preprocess.PreprocessQueue([], X)
    preprocY = preprocess.PreprocessQueue([], Y)
    if center:
        preprocX.push(preprocess.Center())
        preprocY.push(preprocess.Center())
    if scale:
        preprocX.push(preprocess.Scale())
        preprocY.push(preprocess.Scale())
    X = preprocX.process(X)
    Y = preprocY.process(Y)

    num_comp = 1
    nonzero = []
    for l in [0, 0.1, 0.2, 0.3]:
        num = 10
        A = zeros(X.shape[1], num)
        B = zeros(Y.shape[1], num)
        for run in xrange(num):
            spls1 = PLSR(num_comp=num_comp)
#            alg = spls1.get_algorithm()
            spls1.set_tolerance(tol)
            spls1.set_max_iter(miter)
            spls1.set_prox_op(prox_ops.L1(l, l, normaliser=[norm, normI]))
            spls1.set_start_vector(start_vectors.RandomStartVector())
            spls1.fit(X, Y)
            Yhat1 = spls1.predict(X)
            Yhat1 = preprocY.revert(Yhat1)
            SSYdiff1 = np.sum((Yorig - Yhat1) ** 2)
            R2Yhat1 = 1 - (SSYdiff1 / SSY)
            nonzero.append(np.count_nonzero(spls1.W))
            utils.debug("sPLS: l = %.2f, R2Yhat = %.6f, num_comp = %d, " \
                        "nonzero: %d" % (l, R2Yhat1, num_comp, nonzero[-1]))

            A[:, run] = np.abs(spls1.W[:, 0])
            B[:, run] = np.abs(spls1.C[:, 0])

        A[A > 0.01] = 1
        B[B > 0.01] = 1
        kappaA = fleiss_kappa(A, 2)
        kappaB = fleiss_kappa(B, 2)
        utils.debug("Kappa X:", kappaA)
        utils.debug("Kappa Y:", kappaB)
        assert kappaA > 0.1
        assert kappaB > 0.1

    utils.debug()
    num_comp = 1
    nonzeroW = []
    nonzeroWo = []
    for l in [0, 0.1, 0.2, 0.3]:
        num = 10
        A = zeros(X.shape[1], num)
        B = zeros(Y.shape[1], num)
        Ao = zeros(X.shape[1], num)
        Bo = zeros(Y.shape[1], num)
        for run in xrange(num):
            so2pls1 = O2PLS(num_comp=[num_comp, 2, 2])
            so2pls1.set_tolerance(tol)
            so2pls1.set_max_iter(miter)
            so2pls1.set_prox_op(prox_ops.L1([l, l], [l, l]))
            so2pls1.set_start_vector(start_vectors.RandomStartVector())
            so2pls1.fit(X, Y)
            Yhat1 = so2pls1.predict(X)
            Yhat1 = preprocY.revert(Yhat1)
            SSYdiff1 = np.sum((Yorig - Yhat1) ** 2)
            R2Yhat1 = 1 - (SSYdiff1 / SSY)
            nonzeroW.append(np.count_nonzero(so2pls1.W[:, 0]))
            nonzeroWo.append(np.count_nonzero(so2pls1.Wo[:, [0]]))
            utils.debug("O2PLS: l = %.2f, R2Yhat = %.6f, num_comp = %d, " \
                  "nonzeroW: %d, nonzeroWo: %d" \
                  % (l, R2Yhat1, num_comp, nonzeroW[-1], nonzeroWo[-1]))

            A[:, run] = np.abs(so2pls1.W[:, 0])
            B[:, run] = np.abs(so2pls1.C[:, 0])
            Ao[:, run] = np.abs(so2pls1.Wo[:, 0])
            Bo[:, run] = np.abs(so2pls1.Co[:, 0])

        A[A > 0] = 1
        B[B > 0] = 1
        Ao[Ao > 0] = 1
        Bo[Bo > 0] = 1
        kappaA = fleiss_kappa(A, 2)
        kappaB = fleiss_kappa(B, 2)
        kappaAo = fleiss_kappa(Ao, 2)
        kappaBo = fleiss_kappa(Bo, 2)
        utils.debug("Kappa X:", kappaA)
        utils.debug("Kappa Y:", kappaB)
        utils.debug("Kappa Xo:", kappaAo)
        utils.debug("Kappa Yo:", kappaBo)
        assert kappaA > 0.1
        assert kappaB > 0.1
        assert kappaAo > 0.1
        assert kappaBo > 0.1


def test_rgcca():

    # Assure same answers every time
    np.random.seed(42)

    X, Y = orth_matrix(10)
    Z = np.random.rand(10, 10)

    import tests.data.Russett as Russett
    X, Y, Z = Russett.load()

    preprocX = preprocess.PreprocessQueue([preprocess.Center(),
                                           preprocess.Scale()], X)
    preprocY = preprocess.PreprocessQueue([preprocess.Center(),
                                           preprocess.Scale()], Y)
    preprocZ = preprocess.PreprocessQueue([preprocess.Center(),
                                           preprocess.Scale()], Z)
    X = preprocX.process(X)
    Y = preprocY.process(Y)
    Z = preprocZ.process(Z)

    rgcca = RGCCA(num_comp=1, tau=optimal_shrinkage(X, Y, Z))
    rgcca.set_start_vector(start_vectors.OnesStartVector())
    rgcca.set_scheme(schemes.Factorial())
    rgcca.set_max_iter(10000)
    rgcca.set_tolerance(5e-12)
    rgcca.set_adjacency_matrix([[0, 0, 1],
                                [0, 0, 1],
                                [1, 1, 0]])
    rgcca.fit(X, Y, Z)

#    result = RGCCA::rgccak(A, C, tau=tau, scheme="factorial", scale=TRUE,
#                           verbose=TRUE, init="svd", bias=FALSE,
#                           tol = .Machine$double.eps)
#    rgcca_tau = [0.08853216, 0.02703256, 0.03224638]
    rgcca_tau = [0.135580, 0.073970, 0.053935]  # Old tau!

    rgcca_Y = np.asarray([[-0.32340371, 0.33270814, -0.2585368],
                          [-0.73426622, 1.37049777, 1.4075053],
                          [-0.89010881, 0.20859333, -0.2585368],
                          [2.09054709, 1.65449412, 1.4075053],
                          [-0.61193567, -1.40522464, -0.9005069],
                          [-1.19884504, -0.71624869, -0.2585368],
                          [1.13849159, 1.65473163, 1.4075053],
                          [-1.16985077, -0.12472963, -0.2585368],
                          [-0.96284882, -0.52743972, -0.2585368],
                          [-1.61949645, -0.55348628, -0.2585368],
                          [0.09888414, -0.21523725, -0.9005069],
                          [1.27874171, 0.75017443, 1.4075053],
                          [-0.64688605, -0.72669019, -0.9005069],
                          [-1.02430151, -0.66475374, -0.9005069],
                          [-0.94675324, -1.02088613, -0.9005069],
                          [-0.91375603, -0.77326348, -0.9005069],
                          [-0.02544715, 0.04906805, -0.2585368],
                          [1.45639909, 0.67467371, -0.2585368],
                          [-0.98557246, -0.97119447, -0.9005069],
                          [-0.86715474, -0.50199132, -0.2585368],
                          [-0.61205884, -1.04069155, -0.9005069],
                          [1.84009355, -1.35109072, 1.4075053],
                          [0.06674219, -1.11691358, -0.9005069],
                          [0.05596441, -0.03711167, 1.4075053],
                          [-0.48457109, 0.24112295, -0.2585368],
                          [0.84573249, -0.31618197, -0.2585368],
                          [-0.35813231, -1.33116656, -0.9005069],
                          [0.97819243, 0.85063973, 1.4075053],
                          [1.91195763, 1.41731296, 1.4075053],
                          [-0.13816504, 1.25118587, 1.4075053],
                          [-0.79518746, -0.84096166, -0.9005069],
                          [0.40515487, 0.64862715, 1.4075053],
                          [-0.43759076, -0.48446861, -0.9005069],
                          [-0.76503468, -0.93002225, -0.9005069],
                          [1.39139769, -0.78574577, -0.9005069],
                          [0.48031498, -0.42817514, -0.9005069],
                          [-0.03816222, -1.03120888, -0.9005069],
                          [-0.29094954, -0.52095667, -0.9005069],
                          [1.07795848, 1.43819595, 1.4075053],
                          [1.96709022, 1.72519193, 1.4075053],
                          [0.51025252, -0.76653852, -0.9005069],
                          [0.66861847, 2.36304238, 1.4075053],
                          [-0.15828123, 1.96705285, 1.4075053],
                          [-0.02253762, 0.08640057, 1.4075053],
                          [-0.82006407, 0.06383304, -0.9005069],
                          [-0.59086633, 1.19933465, -0.2585368],
                          [0.16969434, -0.76450212, -0.9005069]])

    rgcca_T = [rgcca_Y[:, [0]], rgcca_Y[:, [1]], rgcca_Y[:, [2]]]

    for i in xrange(len(rgcca_T)):
        rgcca_T[i], rgcca.T[i] = direct(rgcca_T[i], rgcca.T[i], compare=True)
        assert_array_almost_equal(rgcca_T[i], rgcca.T[i], decimal=4,
                err_msg="RGCCA does not give the correct scores")

    rgcca_a = [np.asarray([[-0.05801702],
                           [-1.00898259],
                           [0.66649974]]),
               np.asarray([[0.3320846],
                           [-0.7253266]]),
               np.asarray([[0.7850165],
                           [-0.3208365]])]

    for i in xrange(len(rgcca_a)):
        rgcca_a[i], rgcca.W[i] = direct(rgcca_a[i], rgcca.W[i], compare=True)
        assert_array_almost_equal(rgcca_a[i], rgcca.W[i], decimal=4,
                err_msg="RGCCA does not give the correct weights")

    tau = [1.0, 0.0, 1.0]
    rgcca = RGCCA(num_comp=1, tau=tau)
    rgcca.set_start_vector(start_vectors.OnesStartVector())
    rgcca.set_scheme(schemes.Horst())
    rgcca.set_max_iter(10000)
    rgcca.set_tolerance(5e-12)
    rgcca.set_adjacency_matrix([[0, 0, 1],
                                [0, 0, 1],
                                [1, 1, 0]])
    rgcca.fit(X, Y, Z)

#    print sum_corr(*rgcca.T)
#    print sum_cov(*rgcca.T)
#    print rgcca.T

#    result = RGCCA::rgccak(A, C, tau=tau, scheme="horst", scale=TRUE,
#                           verbose=TRUE, init="svd", bias=FALSE,
#                           tol = .Machine$double.eps)
    rgcca_Y = np.asarray([[-1.08408558, 0.35278204, -0.01135023],
                          [-1.57769579, 1.34673245, 1.66322570],
                          [-0.67885386, 0.19281535, -0.01135023],
                          [1.63670136, 1.65997366, 1.66322570],
                          [-1.44809300, -1.33191583, -1.24060913],
                          [-1.25175556, -0.72096975, -0.01135023],
                          [1.93643358, 1.61922048, 1.66322570],
                          [-1.74964016, -0.06812103, -0.01135023],
                          [-1.20537403, -0.54139279, -0.01135023],
                          [-1.60063396, -0.56302733, -0.01135023],
                          [-0.63542847, -0.22064846, -1.24060913],
                          [2.39014638, 0.71773412, 1.66322570],
                          [-0.91443039, -0.71144237, -1.24060913],
                          [-1.36929006, -0.64634715, -1.24060913],
                          [-0.74388080, -0.98537993, -1.24060913],
                          [-1.15149312, -0.77515191, -1.24060913],
                          [0.93337257, -0.02158894, -0.01135023],
                          [1.44735541, 0.62772212, -0.01135023],
                          [-1.36890662, -0.95775567, -1.24060913],
                          [-0.83997221, -0.48840473, -0.01135023],
                          [-0.66883376, -1.00865559, -1.24060913],
                          [1.75973093, -1.28240158, 1.66322570],
                          [-1.12166571, -1.11746692, -1.24060913],
                          [0.98905487, -0.06132036, 1.66322570],
                          [-0.86589534, 0.24258504, -0.01135023],
                          [2.06096612, -0.29311908, -0.01135023],
                          [-0.05106828, -1.27950251, -1.24060913],
                          [0.96631211, 0.80118165, 1.66322570],
                          [1.48137565, 1.44021825, 1.66322570],
                          [-0.46581874, 1.21805541, 1.66322570],
                          [-0.65310876, -0.84958293, -1.24060913],
                          [0.68277373, 0.60608758, 1.66322570],
                          [-0.36783934, -0.50115048, -1.24060913],
                          [-1.17342500, -0.89398227, -1.24060913],
                          [1.36256563, -0.77186968, -1.24060913],
                          [2.30111869, -0.46576766, -1.24060913],
                          [0.05318331, -0.99622914, -1.24060913],
                          [-0.81678994, -0.51325764, -1.24060913],
                          [1.28572683, 1.42116347, 1.66322570],
                          [2.27135202, 1.71869600, 1.66322570],
                          [0.32770887, -0.71724051, -1.24060913],
                          [0.17940265, 2.40547801, 1.66322570],
                          [-0.17980389, 1.91958821, 1.66322570],
                          [-0.69258809, 0.05947040, 1.66322570],
                          [-1.49818264, 0.01115025, -1.24060913],
                          [-0.01012787, 1.20456257, -0.01135023],
                          [2.11940027, -0.78152479, -1.24060913]])

    rgcca_T = [rgcca_Y[:, [0]], rgcca_Y[:, [1]], rgcca_Y[:, [2]]]

    for i in xrange(len(rgcca_T)):
        rgcca_T[i], rgcca.T[i] = direct(rgcca_T[i], rgcca.T[i], compare=True)
        assert_array_almost_equal(rgcca_T[i], rgcca.T[i], decimal=4,
                err_msg="RGCCA does not give the correct scores")

    rgcca_a = [np.asarray([[-0.6207702],
                           [-0.7594084],
                           [0.1947902]]),
               np.asarray([[0.2758329],
                           [-0.7623184]]),
               np.asarray([[0.7890375],
                           [-0.6143451]])]

    for i in xrange(len(rgcca_a)):
        rgcca_a[i], rgcca.W[i] = direct(rgcca_a[i], rgcca.W[i], compare=True)
        assert_array_almost_equal(rgcca_a[i], rgcca.W[i], decimal=4,
                err_msg="RGCCA does not give the correct weights")

    tau = [0.5, 1.0, 0.5]
    rgcca = RGCCA(num_comp=1, tau=tau)
    rgcca.set_start_vector(start_vectors.OnesStartVector())
    rgcca.set_scheme(schemes.Centroid())
    rgcca.set_max_iter(10000)
    rgcca.set_tolerance(5e-12)
    rgcca.set_adjacency_matrix([[0, 0, 1],
                                [0, 0, 1],
                                [1, 1, 0]])
    rgcca.fit(X, Y, Z)

#    result = RGCCA::rgccak(A, C, tau=tau, scheme="centroid", scale=TRUE,
#                           verbose=TRUE, init="svd", bias=FALSE,
#                           tol = .Machine$double.eps)

    rgcca_Y = np.asarray([[-0.63758064, 0.32572337, -0.09147646],
                          [-1.10918357, 1.82531118, 1.48710404],
                          [-0.71496356, 0.33305975, -0.09147646],
                          [1.84597188, 2.04833385, 1.48710404],
                          [-1.04151881, -2.09388644, -1.06044215],
                          [-1.19508109, -0.87607593, -0.09147646],
                          [1.41018757, 2.23483839, 1.48710404],
                          [-1.44460513, -0.41345100, -0.09147646],
                          [-1.05733654, -0.59754389, -0.09147646],
                          [-1.61365967, -0.65022480, -0.09147646],
                          [-0.14670461, -0.24512974, -1.06044215],
                          [1.62151955, 1.08739948, 1.48710404],
                          [-0.68923163, -0.97987196, -1.06044215],
                          [-1.14359046, -0.91660939, -1.06044215],
                          [-0.75695434, -1.44055217, -1.06044215],
                          [-0.96809137, -0.96038812, -1.06044215],
                          [0.30058863, 0.38245751, -0.09147646],
                          [1.40264187, 1.05871033, -0.09147646],
                          [-1.10616401, -1.27803940, -1.06044215],
                          [-0.72139769, -0.69075827, -0.09147646],
                          [-0.56126831, -1.44960592, -1.06044215],
                          [1.83685500, -2.00506631, 1.48710404],
                          [-0.40876917, -1.39707658, -1.06044215],
                          [0.36228965, 0.06346779, 1.48710404],
                          [-0.59333879, 0.29550631, -0.09147646],
                          [1.27548368, -0.50097057, -0.09147646],
                          [-0.21926025, -1.90276029, -1.06044215],
                          [0.91162524, 1.29059729, 1.48710404],
                          [1.66975408, 1.67196583, 1.48710404],
                          [-0.26085870, 1.71834567, 1.48710404],
                          [-0.69002328, -1.01463483, -1.06044215],
                          [0.42029059, 1.00602942, 1.48710404],
                          [-0.37840608, -0.53130130, -1.06044215],
                          [-0.98227198, -1.32911662, -1.06044215],
                          [1.40202935, -1.04764245, -1.06044215],
                          [1.07543988, -0.36577060, -1.06044215],
                          [0.08287892, -1.45109593, -1.06044215],
                          [-0.39720025, -0.68777824, -1.06044215],
                          [1.14567544, 1.87955789, 1.48710404],
                          [2.00282063, 2.19132490, 1.48710404],
                          [0.52631171, -1.18448399, -1.06044215],
                          [0.48729836, 2.76832751, 1.48710404],
                          [-0.09480629, 2.68050481, 1.48710404],
                          [-0.29717888, 0.23060217, 1.48710404],
                          [-1.11686321, 0.31930862, -1.06044215],
                          [-0.31671152, 1.47912232, -0.09147646],
                          [0.88335781, -0.88065964, -1.06044215]])

    rgcca_T = [rgcca_Y[:, [0]], rgcca_Y[:, [1]], rgcca_Y[:, [2]]]

    for i in xrange(len(rgcca_T)):
        rgcca_T[i], rgcca.T[i] = direct(rgcca_T[i], rgcca.T[i], compare=True)
        assert_array_almost_equal(rgcca_T[i], rgcca.T[i], decimal=4,
                err_msg="RGCCA does not give the correct scores")

    rgcca_a = [np.asarray([[-0.4233274],
                           [-0.7356075],
                           [0.4764125]]),
               np.asarray([[0.6716633],
                           [-0.7408565]]),
               np.asarray([[0.7438057],
                           [-0.4842587]])]

    for i in xrange(len(rgcca_a)):
        rgcca_a[i], rgcca.W[i] = direct(rgcca_a[i], rgcca.W[i], compare=True)
        assert_array_almost_equal(rgcca_a[i], rgcca.W[i], decimal=4,
                err_msg="RGCCA does not give the correct weights")


def test_ista():
    n = 26
    p = 24
    X = np.random.randn(n, p)
    betastar = np.concatenate((np.zeros((p / 2, 1)),
                               np.random.randn(p / 2, 1)))
    y = np.dot(X, betastar)
    D, V = eig(np.dot(X.T, X))
    t = 0.95 / np.max(D.real)

    lr = LinearRegression()
    alg = lr.get_algorithm()
    alg.set_max_iter(10000)
    alg.set_tolerance(5e-10)
    lr.fit(X, y, t=t)
    alg = lr.get_algorithm()

    print norm(lr.beta - betastar)
    print alg.iterations

    lr = LinearRegression()
    alg = lr.get_algorithm()
    alg.set_max_iter(10000)
    alg.set_tolerance(5e-10)
    h = error_functions.L1(10)
    lr.fit(X, y, h=h, t=t)

    print norm(lr.beta - betastar)
    print alg.iterations
    print lr.beta

#    print betastar
#    print lr.beta
#    print (lr.beta - betastar)

    import pylab
    pylab.plot(betastar[:, 0], '-', lr.beta[:, 0], '*')
    pylab.title("the iteration number is equal to " + str(alg.iterations))
    pylab.show()

#    xi = [log(n) for n in range(1, (len(alg.f) + 1))]
#    pylab.plot(np.log(xi), ista.f_beta_k, '-')
#    pylab.show()
    #xf = [log(n) for n in range(1, (len(fista.crit) + 1))]
    #xi = [log(n) for n in range(1, (len(ista.crit) + 1))]
    #xfm = [log(n) for n in range(1, (len(fistam.crit) + 1))]
    #pylab.plot(xf, fista.crit, '--r', xi, ista.crit, '-b', xfm, fistam.crit, ':k')
    #pylab.show()


def test_tv():

    import pylab

    np.random.seed(42)

    eps = 0.1
    maxit = 10000
    M = 300
    N = 1
    O = 1
    p = M * N * O  # Must be even!
    n = 50
    X = np.random.randn(n, p)
    betastar = np.concatenate((np.zeros((p / 2, 1)),
                               np.random.randn(p / 2, 1)))
    betastar = np.sort(np.abs(betastar), axis=0)
    y = np.dot(X, betastar)

#    np.savetxt('test.txt', np.vstack((X, betastar.T)), delimiter='\t')

#    # "Regular" linear regression with L1 regularisation
#    lr = LinearRegression()
#    alg = lr.get_algorithm()
#    alg.set_max_iter(maxit)
#    alg.set_tolerance(eps)
#    h = error_functions.L1(1)
#    lr.fit(X, y, t=t)
#
#    print norm(lr.beta - betastar)
#    print alg.iterations
#    print lr.beta
#    print np.reshape(lr.beta, (O, M, N))  # pz, py, px
#
#    pylab.subplot(4, 2, 1)
#    pylab.plot(betastar[:, 0], '-', lr.beta[:, 0], '*')
#    pylab.title("Iterations: " + str(alg.iterations))
#    pylab.subplot(4, 2, 2)
#    pylab.plot(alg.f)

    gamma = 1
    l = 0.1
    r = 0
    for i in xrange(X.shape[1]):
        r = max(r, abs(utils.cov(X[:, [i]], y)))
    mus = [r * 0.5 ** i for i in xrange(5)]

    total_start = time()
    # Linear regression with total variation regularisation
    lr = LinearRegression(algorithm=algorithms.ISTARegression())
    alg = lr.get_algorithm()
    alg.set_max_iter(maxit)
    alg.set_tolerance(eps)

    g1 = error_functions.SumSqRegressionError(X, y)
    start = time()
    g2 = error_functions.TotalVariation((M, N, O), gamma, mus[0])
    print "time TV init:", time() - start
    g = error_functions.CombinedNesterovErrorFunction(g1, g2, mus)
    h = error_functions.L1(l)

    lr.fit(X, y, g=g, h=h)
    print "Total time:", (time()-total_start)

#    print norm(lr.beta - betastar)
#    print alg.iterations
#    print lr.beta
#    print np.reshape(lr.beta, (O, M, N))

#    pylab.subplot(4, 2, 3)
#    pylab.plot(betastar[:, 0], '-', lr.beta[:, 0], '*')
#    pylab.title("Iterations: " + str(alg.iterations))
#    pylab.subplot(4, 2, 4)
#    pylab.plot(alg.f)
#    pylab.title("Iterations: " + str(alg.iterations))

#    pylab.subplot(2, 1, 1)
    pylab.plot(betastar[:, 0], '-', lr.beta[:, 0], '*')
    pylab.title("Iterations: " + str(alg.iterations))
    pylab.show()
#
    pylab.plot(alg.f, '-b')
    pylab.show()

    return

    # Linear regression with total variation regularisation
    lr = LinearRegression(algorithm=algorithms.FISTARegression())
    alg = lr.get_algorithm()
    alg.set_max_iter(maxit)
    alg.set_tolerance(eps)

    g1 = error_functions.SumSqRegressionError(X, y)
    g2 = error_functions.TotalVariation((M, N, O), gamma, mu)
    g = error_functions.CombinedDifferentiableErrorFunction(g1, g2)
    h = error_functions.L1(l)

    lr.fit(X, y, g=g, h=h, t=t)

    print norm(lr.beta - betastar)
    print alg.iterations
    print lr.beta
    print np.reshape(lr.beta, (O, M, N))

    pylab.subplot(4, 2, 5)
    pylab.plot(betastar[:, 0], '-', lr.beta[:, 0], '*')
    pylab.title("Iterations: " + str(alg.iterations))
    pylab.subplot(4, 2, 6)
    pylab.plot(alg.f)
#    pylab.title("Iterations: " + str(alg.iterations))
    gamma_small_beta = lr.beta



    # Linear regression with total variation regularisation
    lr = LinearRegression(algorithm=algorithms.MonotoneFISTARegression())
    alg = lr.get_algorithm()
    alg.set_max_iter(maxit)
    alg.set_tolerance(eps)

    g1 = error_functions.SumSqRegressionError(X, y)
    g2 = error_functions.TotalVariation((M, N, O), gamma, mu)
    g = error_functions.CombinedDifferentiableErrorFunction(g1, g2)
    h = error_functions.L1(l)

    lr.fit(X, y, g=g, h=h, t=t)

    print norm(lr.beta - betastar)
    print alg.iterations
    print lr.beta
    print np.reshape(lr.beta, (O, M, N))

    print "diff:", norm(gamma_small_beta - lr.beta)

    pylab.subplot(4, 2, 7)
    pylab.plot(betastar[:, 0], '-', lr.beta[:, 0], '*')
    pylab.title("Iterations: " + str(alg.iterations))
    pylab.subplot(4, 2, 8)
    pylab.plot(alg.f)
#    pylab.title("Iterations: " + str(alg.iterations))
    pylab.show()

#    xi = [log(n) for n in range(1, (len(alg.f) + 1))]
#    pylab.plot(np.log(xi), ista.f_beta_k, '-')
#    pylab.show()
    #xf = [log(n) for n in range(1, (len(fista.crit) + 1))]
    #xi = [log(n) for n in range(1, (len(ista.crit) + 1))]
    #xfm = [log(n) for n in range(1, (len(fistam.crit) + 1))]
    #pylab.plot(xf, fista.crit, '--r', xi, ista.crit, '-b', xfm, fistam.crit, ':k')
    #pylab.show()



def test_gl():

    import pylab
    from time import time

    np.random.seed(42)

    eps = 0.1
    maxit = 10000
    p = 100  # Must be even!
    n = 100
    X = np.random.randn(n, p)
    betastar = np.concatenate((np.zeros((p / 2, 1)),
                               np.random.randn(p / 2, 1)))
    y = np.dot(X, betastar)

    gamma = 1
    l = 1

    r = 0
    for i in xrange(X.shape[1]):
        r = max(r, abs(utils.cov(X[:, [i]], y)))
    mus = [r * 0.5 ** i for i in xrange(5)]

    # Linear regression with total variation regularisation
    lr = LinearRegression(algorithm=algorithms.MonotoneFISTARegression())
    alg = lr.get_algorithm()
    alg.set_max_iter(maxit)
    alg.set_tolerance(eps)

    groups = [range(p / 2), range(p / 2, p)]
    g1 = error_functions.SumSqRegressionError(X, y)
    start = time()
    g2 = error_functions.GroupLassoOverlap(p, groups, gamma, mus[0])

    g = error_functions.CombinedNesterovErrorFunction(g1, g2, mus)
    h = error_functions.L1(l)
    lr.fit(X, y, g=g, h=h)
    print "time:", time() - start

    pylab.subplot(2, 1, 1)
    pylab.plot(betastar[:, 0], '-', lr.beta[:, 0], '*')
    pylab.title("Iterations: " + str(alg.iterations))
    pylab.subplot(2, 1, 2)
    pylab.plot(alg.f, '.')
    pylab.show()


def test_regression():

    n = 2000
    p = 50
    # generate a Gaussian dataset
    X = np.random.randn(n, p)
    # generate a beta with "overlapping groups" of coefficients
    beta1 = beta2 = beta3 = np.zeros((p, 1))
    beta1[0:20] = np.random.randn(20, 1)
    beta2[15:35] = np.random.randn(20, 1)
    beta3[27:50] = np.random.randn(23, 1)
    beta = beta1 + beta2 + beta3

    # compute X beta
    combi = np.dot(X, beta)

    # compute the class of each individual
    proba = 1 / (1 + np.exp(-combi))
    y = np.zeros((n, 1))
    for i in xrange(n):
        y[i] = np.random.binomial(1, proba[i], 1)

    eps = 0.0001
    maxit = 50000

    lr = LogisticRegression()
    alg = lr.get_algorithm()
    alg.set_max_iter(maxit)
    alg.set_tolerance(eps)

    gamma = 1
    r = 0
    for i in xrange(X.shape[1]):
        r = max(r, abs(utils.cov(X[:, [i]], y)))
    mus = [r * 0.5 ** i for i in xrange(5)]
    g1 = error_functions.LogisticRegressionError(X, y)
    g2 = error_functions.TotalVariation((50, 1, 1), gamma, mus[0])
    g = error_functions.CombinedNesterovErrorFunction(g1, g2, mus)
#    h = error_functions.L1(l)

    lr.fit(X, y, g=g)

    import pylab
    pylab.subplot(2, 1, 1)
    print norm(lr.beta)
    print norm(beta)
#    lr.beta /= norm(lr.beta)
#    beta /= norm(beta)
    pylab.plot(beta[:, 0], '-', lr.beta[:, 0], '*:')
    pylab.title("Iterations: " + str(alg.iterations))
    pylab.subplot(2, 1, 2)
    pylab.plot(alg.f, '.')
    pylab.show()


############################################################################
############################################################################


def sum_corr(*T, **kwargs):

    n = len(T)

    adj_matrix = kwargs.pop('adj_matrix', None)

    if adj_matrix == None:
        adj_matrix = ones((n, n)) - eye(n, n)

    cr = 0
    for i in xrange(n):
        Ti = T[i]
        for j in xrange(n):
            Tj = T[j]
            if adj_matrix[i, j] != 0:
                for k in xrange(Tj.shape[1]):
                    print i, j, k, corr(Ti[:, [k]], Tj[:, [k]])
                    cr += corr(Ti[:, [k]], Tj[:, [k]])
    return cr


def sum_cov(*T, **kwargs):

    n = len(T)

    adj_matrix = kwargs.pop('adj_matrix', None)

    if adj_matrix == None:
        adj_matrix = ones((n, n)) - eye(n, n)

    cv = 0
    for i in xrange(n):
        Ti = T[i]
        for j in xrange(n):
            Tj = T[j]
            if adj_matrix[i, j] != 0 or adj_matrix[j, i] != 0:
                for k in xrange(Tj.shape[1]):
                    print i, j, k, cov(Ti[:, [k]], Tj[:, [k]])
                    cv += cov(Ti[:, [k]], Tj[:, [k]])
    return cv


def fleiss_kappa(W, k):
    """Computes Fleiss' kappa for a set of variables classified into k
    categories by a number of different raters.

    W is a matrix with shape (variables, raters) with k categories between
    0,...,k.
    """
    N, n = W.shape
    if n <= 1:
        raise ValueError("At least two ratings are needed")
    A = zeros(N, k)
    Nn = N * n
    p = [0] * k
    for j in xrange(k):
        A[:, j] = np.sum(W == j, axis=1)

        p[j] = np.sum(A[:, j]) / float(Nn)

    P = [0] * N
    for i in xrange(N):
        for j in xrange(k):
            P[i] += A[i, j] ** 2
        P[i] -= n
        P[i] /= float(n * (n - 1))

    P_ = sum(P) / float(N)
    Pe = sum([pj ** 2 for pj in p])

    if abs(Pe - 1) < TOLERANCE:
        kappa = 1
    else:
        kappa = (P_ - Pe) / (1.0 - Pe)
    if kappa > 1:
        kappa = 1

    return kappa


def orth_matrix(n=10):
    Y = rand(n, 1)
    X = zeros(n, n)
    if n > 2:
        for j in xrange(n - 1):
            x = rand(n, 1)
            while abs(abs(corr(x, Y)) - j / (n - 1.0)) > 0.005:
                x = rand(n, 1)
            if corr(x, Y) < 0:
                x *= -1
            X[:, j] = x.ravel()

    X[:, n - 1] = Y.ravel()

    return X, Y


#    # Check PLS properties (with n_components=X.shape[1])
#    # ---------------------------------------------------
#    plsca = PLSCanonical(n_components=X.shape[1])
#    plsca.fit(X, Y)
#    pls_byNIPALS = pls.PLSC(num_comp=X.shape[1], tolerance=tol, max_iter=1000)
#    pls_byNIPALS.fit(X, Y)
#
#    T = plsca.x_scores_
#    P = plsca.x_loadings_
#    Wx = plsca.x_weights_
#    U = plsca.y_scores_
#    Q = plsca.y_loadings_
#    Wy = plsca.y_weights_
#
#    def check_ortho(M, err_msg):
#        K = np.dot(M.T, M)
#        assert_array_almost_equal(K, np.diag(np.diag(K)), err_msg=err_msg)
#
#    # Orthogonality of weights
#    # ~~~~~~~~~~~~~~~~~~~~~~~~
#    check_ortho(Wx, "X weights are not orthogonal")
#    check_ortho(Wy, "Y weights are not orthogonal")
#    check_ortho(pls_byNIPALS.W, "X weights are not orthogonal")
#    check_ortho(pls_byNIPALS.C, "Y weights are not orthogonal")
#    print "Testing orthogonality of weights ... OK!"
#
#    # Orthogonality of latent scores
#    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#    check_ortho(T, "X scores are not orthogonal")
#    check_ortho(U, "Y scores are not orthogonal")
#    check_ortho(pls_byNIPALS.T, "X scores are not orthogonal")
#    check_ortho(pls_byNIPALS.U, "Y scores are not orthogonal")
#    print "Testing orthogonality of scores ... OK!"
#
#
#    # Check X = TP' and Y = UQ' (with (p == q) components)
#    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#    # center scale X, Y
#    Xc, Yc, x_mean, y_mean, x_std, y_std = \
#        _center_scale_xy(X.copy(), Y.copy(), scale=True)
#    assert_array_almost_equal(Xc, dot(T, P.T), err_msg="X != TP'")
#    assert_array_almost_equal(Yc, dot(U, Q.T), err_msg="Y != UQ'")
#    print "Testing equality of matriecs and their models ... OK!"
#
#    Xc, mX = pls.center(X, return_means = True)
#    Xc, sX = pls.scale(Xc, return_stds = True)
#    Yc, mY = pls.center(Y, return_means = True)
#    Yc, sY = pls.scale(Yc, return_stds = True)
#
#    assert_array_almost_equal(Xc, dot(pls_byNIPALS.T, pls_byNIPALS.P.T), err_msg="X != TP'")
#    assert_array_almost_equal(Yc, dot(pls_byNIPALS.U, pls_byNIPALS.Q.T), err_msg="Y != UQ'")
#    print "Testing equality of matriecs and their models ... OK!"
#
#
#    # Check that rotations on training data lead to scores
#    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#    plsca = PLSCanonical(n_components=X.shape[1])
#    plsca.fit(X, Y)
#    pls_byNIPALS = pls.PLSC(num_comp=X.shape[1], tolerance=tol, max_iter=1000)
#    pls_byNIPALS.fit(X, Y)
#
#    Xr = plsca.transform(X)
#    assert_array_almost_equal(Xr, plsca.x_scores_,
#                              err_msg="Rotation of X failed")
#    Xr = pls_byNIPALS.transform(X)
#    assert_array_almost_equal(Xr, pls_byNIPALS.T,
#                              err_msg="Rotation of X failed")
#    print "Testing equality of computed X scores and the transform ... OK!"
#    Xr, Yr = plsca.transform(X, Y)
#    assert_array_almost_equal(Xr, plsca.x_scores_,
#                              err_msg="Rotation of X failed")
#    assert_array_almost_equal(Yr, plsca.y_scores_,
#                              err_msg="Rotation of Y failed")
#    print "Testing equality of computed X and Y scores and the transform ... OK!"
#
#
#    # "Non regression test" on canonical PLS
#    # --------------------------------------
#    # The results were checked against the R-package plspm
#    pls_ca = PLSCanonical(n_components=X.shape[1])
#    pls_ca.fit(X, Y)
#    pls_byNIPALS = pls.PLSC(num_comp=X.shape[1], tolerance=tol, max_iter=1000)
#    pls_byNIPALS.fit(X, Y)
#
#    x_weights = np.array(
#        [[-0.61330704,  0.25616119, -0.74715187],
#         [-0.74697144,  0.11930791,  0.65406368],
#         [-0.25668686, -0.95924297, -0.11817271]])
#    x_weights, pls_ca.x_weights_ = \
#            pls.direct(x_weights, pls_ca.x_weights_, compare = True)
#    assert_array_almost_equal(x_weights, pls_ca.x_weights_, decimal = 5)
#    x_weights, pls_byNIPALS.W = \
#            pls.direct(x_weights, pls_byNIPALS.W, compare = True)
#    assert_array_almost_equal(x_weights, pls_byNIPALS.W, decimal = 5)
#    print "Testing equality of X weights ... OK!"
#
#    x_rotations = np.array(
#        [[-0.61330704,  0.41591889, -0.62297525],
#         [-0.74697144,  0.31388326,  0.77368233],
#         [-0.25668686, -0.89237972, -0.24121788]])
#    x_rotations, pls_ca.x_rotations_ = \
#            pls.direct(x_rotations, pls_ca.x_rotations_, compare = True)
#    assert_array_almost_equal(x_rotations, pls_ca.x_rotations_, decimal = 5)
#    x_rotations, pls_byNIPALS.Ws = \
#            pls.direct(x_rotations, pls_byNIPALS.Ws, compare = True)
#    assert_array_almost_equal(x_rotations, pls_byNIPALS.Ws, decimal = 5)
#    print "Testing equality of X loadings weights ... OK!"
#
#    y_weights = np.array(
#        [[+0.58989127,  0.7890047,   0.1717553],
#         [+0.77134053, -0.61351791,  0.16920272],
#         [-0.23887670, -0.03267062,  0.97050016]])
#    y_weights, pls_ca.y_weights_ = \
#            pls.direct(y_weights, pls_ca.y_weights_, compare = True)
#    assert_array_almost_equal(y_weights, pls_ca.y_weights_, decimal = 5)
#    y_weights, pls_byNIPALS.C = \
#            pls.direct(y_weights, pls_byNIPALS.C, compare = True)
#    assert_array_almost_equal(y_weights, pls_byNIPALS.C, decimal = 5)
#    print "Testing equality of Y weights ... OK!"
#
#    y_rotations = np.array(
#        [[+0.58989127,  0.7168115,  0.30665872],
#         [+0.77134053, -0.70791757,  0.19786539],
#         [-0.23887670, -0.00343595,  0.94162826]])
#    pls_ca.y_rotations_, y_rotations = \
#            pls.direct(pls_ca.y_rotations_, y_rotations, compare = True)
#    assert_array_almost_equal(pls_ca.y_rotations_, y_rotations)
#    y_rotations, pls_byNIPALS.Cs = \
#            pls.direct(y_rotations, pls_byNIPALS.Cs, compare = True)
#    assert_array_almost_equal(y_rotations, pls_byNIPALS.Cs, decimal = 5)
#    print "Testing equality of Y loadings weights ... OK!"
#
#    assert_array_almost_equal(X, Xorig, decimal = 5, err_msg = "X and Xorig are not equal!!")
#    assert_array_almost_equal(Y, Yorig, decimal = 5, err_msg = "Y and Yorig are not equal!!")
#
#
#    # 2) Regression PLS (PLS2): "Non regression test"
#    # ===============================================
#    # The results were checked against the R-packages plspm, misOmics and pls
#    pls_2 = PLSRegression(n_components=X.shape[1])
#    pls_2.fit(X, Y)
#
#    pls_NIPALS = pls.PLSR(num_comp = X.shape[1],
#                          center = True, scale = True,
#                          tolerance=tol, max_iter=1000)
#    pls_NIPALS.fit(X, Y)
#
#    x_weights = np.array(
#        [[-0.61330704, -0.00443647,  0.78983213],
#         [-0.74697144, -0.32172099, -0.58183269],
#         [-0.25668686,  0.94682413, -0.19399983]])
#    x_weights, pls_NIPALS.W = pls.direct(x_weights, pls_NIPALS.W, compare = True)
#    assert_array_almost_equal(pls_NIPALS.W, x_weights, decimal=5,
#            err_msg="sklearn.NIPALS.PLSR and sklearn.pls.PLSRegression " \
#                    "implementations lead to different X weights")
#    print "Comparing X weights of sklearn.NIPALS.PLSR and " \
#            "sklearn.pls.PLSRegression ... OK!"
#
#    x_loadings = np.array(
#        [[-0.61470416, -0.24574278,  0.78983213],
#         [-0.65625755, -0.14396183, -0.58183269],
#         [-0.51733059,  1.00609417, -0.19399983]])
#    x_loadings, pls_NIPALS.P = pls.direct(x_loadings, pls_NIPALS.P, compare = True)
#    assert_array_almost_equal(pls_NIPALS.P, x_loadings, decimal=5,
#            err_msg="sklearn.NIPALS.PLSR and sklearn.pls.PLSRegression " \
#                    "implementations lead to different X loadings")
#    print "Comparing X loadings of sklearn.NIPALS.PLSR and " \
#            "sklearn.pls.PLSRegression ... OK!"
#
#    y_weights = np.array(
#        [[+0.32456184,  0.29892183,  0.20316322],
#         [+0.42439636,  0.61970543,  0.19320542],
#         [-0.13143144, -0.26348971, -0.17092916]])
#    y_weights, pls_NIPALS.C = pls.direct(y_weights, pls_NIPALS.C, compare = True)
#    assert_array_almost_equal(pls_NIPALS.C, y_weights, decimal=5,
#            err_msg="sklearn.NIPALS.PLSR and sklearn.pls.PLSRegression " \
#                    "implementations lead to different Y weights")
#    print "Comparing Y weights of sklearn.NIPALS.PLSR and " \
#            "sklearn.pls.PLSRegression ... OK!"
#
#    X_, m = pls.center(X, return_means = True)
#    X_, s = pls.scale(X_, return_stds = True, centered = True)
#    t1 = dot(X_, x_weights[:,[0]])
#    t1, pls_NIPALS.T[:,[0]] = pls.direct(t1, pls_NIPALS.T[:,[0]], compare = True)
#    assert_array_almost_equal(t1, pls_NIPALS.T[:,[0]], decimal=5,
#            err_msg="sklearn.NIPALS.PLSR and sklearn.pls.PLSRegression " \
#                    "implementations lead to different X scores")
#    print "Comparing scores of sklearn.NIPALS.PLSR and " \
#            "sklearn.pls.PLSRegression ... OK!"
#
#    y_loadings = np.array(
#        [[+0.32456184,  0.29892183,  0.20316322],
#         [+0.42439636,  0.61970543,  0.19320542],
#         [-0.13143144, -0.26348971, -0.17092916]])
#    y_loadings, pls_NIPALS.C = pls.direct(y_loadings, pls_NIPALS.C, compare = True)
#    assert_array_almost_equal(pls_NIPALS.C, y_loadings)
#
#
#    # 3) Another non-regression test of Canonical PLS on random dataset
#    # =================================================================
#    # The results were checked against the R-package plspm
#    #
#    # Warning! This example is not stable, and the reference weights have
#    # not converged properly!
#    #
#    n = 500
#    p_noise = 10
#    q_noise = 5
#    # 2 latents vars:
#    np.random.seed(11)
#    l1 = np.random.normal(size=n)
#    l2 = np.random.normal(size=n)
#    latents = np.array([l1, l1, l2, l2]).T
#    X = latents + np.random.normal(size=4 * n).reshape((n, 4))
#    Y = latents + np.random.normal(size=4 * n).reshape((n, 4))
#    X = np.concatenate(
#        (X, np.random.normal(size=p_noise * n).reshape(n, p_noise)), axis=1)
#    Y = np.concatenate(
#        (Y, np.random.normal(size=q_noise * n).reshape(n, q_noise)), axis=1)
#    np.random.seed(None)
#    x_weights = np.array(
#        [[ 0.65803719,  0.19197924,  0.21769083],
#         [ 0.7009113,   0.13303969, -0.15376699],
#         [ 0.13528197, -0.68636408,  0.13856546],
#         [ 0.16854574, -0.66788088, -0.12485304],
#         [-0.03232333, -0.04189855,  0.40690153],
#         [ 0.1148816,  -0.09643158,  0.1613305 ],
#         [ 0.04792138, -0.02384992,  0.17175319],
#         [-0.06781,    -0.01666137, -0.18556747],
#         [-0.00266945, -0.00160224,  0.11893098],
#         [-0.00849528, -0.07706095,  0.1570547 ],
#         [-0.00949471, -0.02964127,  0.34657036],
#         [-0.03572177,  0.0945091,   0.3414855 ],
#         [ 0.05584937, -0.02028961, -0.57682568],
#         [ 0.05744254, -0.01482333, -0.17431274]])
#    tols = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]#, 5e-7, 5e-8, 5e-9, 5e-10, 5e-11]
#    for i in tols:
#        pls_ca = PLSCanonical(n_components=3, max_iter=1000, tol=i)
#        pls_ca.fit(X, Y)
#
#        x_weights, pls_ca.x_weights_ = pls.direct(x_weights, pls_ca.x_weights_, compare = True)
#        print "tolerance: "+str(i).rjust(6)+", error:", np.max(pls_ca.x_weights_ - x_weights)
#
#    assert_array_almost_equal(pls_ca.x_weights_, x_weights, decimal = 4,
#            err_msg="sklearn.pls.PLSCanonical does not give the same " \
#                    "X weights as the reference model")
#    print "Comparing X weights of sklearn.pls.PLSCanonical ... OK!"
#
#    for i in tols:
#        pls_NIPALS = pls.PLSC(num_comp=3, tolerance=i, max_iter=1000)
#        pls_NIPALS.fit(X, Y)
#
#        x_weights, pls_NIPALS.W = pls.direct(x_weights, pls_NIPALS.W, compare = True)
#        print "tolerance: "+str(i).rjust(6)+", error:", np.max(x_weights - pls_NIPALS.W)
#
#    assert_array_almost_equal(pls_NIPALS.W, x_weights, decimal = 4,
#            err_msg="sklearn.NIPALS.PLSC does not give the same " \
#                    "X weights as the reference model")
#    print "Comparing X weights of sklearn.NIPALS.PLSC ... OK! "
#
#    x_loadings = np.array(
#        [[ 0.65649254,  0.1847647,   0.15270699],
#         [ 0.67554234,  0.15237508, -0.09182247],
#         [ 0.19219925, -0.67750975,  0.08673128],
#         [ 0.2133631,  -0.67034809, -0.08835483],
#         [-0.03178912, -0.06668336,  0.43395268],
#         [ 0.15684588, -0.13350241,  0.20578984],
#         [ 0.03337736, -0.03807306,  0.09871553],
#         [-0.06199844,  0.01559854, -0.1881785 ],
#         [ 0.00406146, -0.00587025,  0.16413253],
#         [-0.00374239, -0.05848466,  0.19140336],
#         [ 0.00139214, -0.01033161,  0.32239136],
#         [-0.05292828,  0.0953533,   0.31916881],
#         [ 0.04031924, -0.01961045, -0.65174036],
#         [ 0.06172484, -0.06597366, -0.1244497]])
#    pls_ca.x_loadings_, x_loadings = pls.direct(pls_ca.x_loadings_, x_loadings, compare = True)
#    assert_array_almost_equal(pls_ca.x_loadings_, x_loadings, decimal = 4,
#            err_msg="sklearn.pls.PLSCanonical does not give the same " \
#                    "X loadings as the reference model")
#    print "Comparing X loadings of sklearn.pls.PLSCanonical ... OK!"
#
#    pls_NIPALS.P, x_loadings = pls.direct(pls_NIPALS.P, x_loadings, compare = True)
#    assert_array_almost_equal(pls_NIPALS.P, x_loadings, decimal = 4,
#            err_msg="sklearn.NIPALS.PLSC does not give the same " \
#                    "loadings as the reference model")
#    print "Comparing X loadings of sklearn.NIPALS.PLSC ... OK! "
#
#    y_weights = np.array(
#        [[0.66101097,  0.18672553,  0.22826092],
#         [0.69347861,  0.18463471, -0.23995597],
#         [0.14462724, -0.66504085,  0.17082434],
#         [0.22247955, -0.6932605, -0.09832993],
#         [0.07035859,  0.00714283,  0.67810124],
#         [0.07765351, -0.0105204, -0.44108074],
#         [-0.00917056,  0.04322147,  0.10062478],
#         [-0.01909512,  0.06182718,  0.28830475],
#         [0.01756709,  0.04797666,  0.32225745]])
#    pls_ca.y_weights_, y_weights = pls.direct(pls_ca.y_weights_, y_weights, compare = True)
#    assert_array_almost_equal(pls_ca.y_weights_, y_weights, decimal = 4,
#            err_msg="sklearn.pls.PLSCanonical does not give the same " \
#                    "Y weights as the reference model")
#    print "Comparing Y weights of sklearn.pls.PLSCanonical ... OK!"
#
#    pls_NIPALS.C, y_weights = pls.direct(pls_NIPALS.C, y_weights, compare = True)
#    assert_array_almost_equal(pls_NIPALS.C, y_weights, decimal = 4,
#            err_msg="sklearn.NIPALS.PLSC does not give the same " \
#                    "loadings as the reference model")
#    print "Comparing Y weights of sklearn.NIPALS.PLSC ... OK! "
#
#    y_loadings = np.array(
#        [[0.68568625,   0.1674376,   0.0969508 ],
#         [0.68782064,   0.20375837, -0.1164448 ],
#         [0.11712173,  -0.68046903,  0.12001505],
#         [0.17860457,  -0.6798319,  -0.05089681],
#         [0.06265739,  -0.0277703,   0.74729584],
#         [0.0914178,    0.00403751, -0.5135078 ],
#         [-0.02196918, -0.01377169,  0.09564505],
#         [-0.03288952,  0.09039729,  0.31858973],
#         [0.04287624,   0.05254676,  0.27836841]])
#    pls_ca.y_loadings_, y_loadings = pls.direct(pls_ca.y_loadings_, y_loadings, compare = True)
#    assert_array_almost_equal(pls_ca.y_loadings_, y_loadings, decimal = 4,
#            err_msg="sklearn.pls.PLSCanonical does not give the same " \
#                    "Y loadings as the reference model")
#    print "Comparing Y loadings of sklearn.pls.PLSCanonical ... OK!"
#
#    pls_NIPALS.Q, y_loadings = pls.direct(pls_NIPALS.Q, y_loadings, compare = True)
#    assert_array_almost_equal(pls_NIPALS.Q, y_loadings, decimal = 4,
#            err_msg="sklearn.NIPALS.PLSC does not give the same " \
#                    "Y loadings as the reference model")
#    print "Comparing Y loadings of sklearn.NIPALS.PLSC ... OK!"
#
#    # Orthogonality of weights
#    # ~~~~~~~~~~~~~~~~~~~~~~~~
#    check_ortho(pls_ca.x_weights_, "X weights are not orthogonal in sklearn.pls.PLSCanonical")
#    check_ortho(pls_ca.y_weights_, "Y weights are not orthogonal in sklearn.pls.PLSCanonical")
#    print "Confirming orthogonality of weights in sklearn.pls.PLSCanonical ... OK!"
#    check_ortho(pls_NIPALS.W, "X weights are not orthogonal in sklearn.NIPALS.PLSC")
#    check_ortho(pls_NIPALS.C, "Y weights are not orthogonal in sklearn.NIPALS.PLSC")
#    print "Confirming orthogonality of weights in sklearn.NIPALS.PLSC ... OK!"
#
#    # Orthogonality of latent scores
#    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#    check_ortho(pls_ca.x_scores_, "X scores are not orthogonal in sklearn.pls.PLSCanonical")
#    check_ortho(pls_ca.y_scores_, "Y scores are not orthogonal in sklearn.pls.PLSCanonical")
#    print "Confirming orthogonality of scores in sklearn.pls.PLSCanonical ... OK!"
#    check_ortho(pls_NIPALS.T, "X scores are not orthogonal in sklearn.NIPALS.PLSC")
#    check_ortho(pls_NIPALS.U, "Y scores are not orthogonal in sklearn.NIPALS.PLSC")
#    print "Confirming orthogonality of scores in sklearn.NIPALS.PLSC ... OK!"
#
#
#    # Compare sparse sklearn.NIPALS.PLSR and sklearn.pls.PLSRegression
#
#    d = load_linnerud()
#    X = np.asarray(d.data)
#    Y = d.target
#    num_comp = 3
#    tol = 5e-12
#
#    for st in [0.1, 0.01, 0.001, 0.0001, 0]:
#
#        plsr = pls.PLSR(num_comp = num_comp, center = True, scale = True,
#                        tolerance=tol, max_iter=1000, soft_threshold = st)
#        plsr.fit(X, Y)
#        Yhat = plsr.predict(X)
#
#        pls2 = PLSRegression(n_components=num_comp, scale=True,
#                     max_iter=1000, tol=tol, copy=True)
#        pls2.fit(X, Y)
#        Yhat_ = pls2.predict(X)
#
#        SSY     = np.sum(Y**2)
#        SSYdiff = np.sum((Y-Yhat)**2)
##        print np.sum(1 - (SSYdiff/SSY))
#        SSY     = np.sum(Y**2)
#        SSYdiff = np.sum((Y-Yhat_)**2)
##        print np.sum(1 - (SSYdiff/SSY))
#
#        if st < tol:
#            num_decimals = 5
#        else:
#            num_decimals = int(log(1./st, 10) + 0.5)
#        assert_array_almost_equal(Yhat, Yhat_, decimal=num_decimals-2,
#                err_msg="NIPALS SVD and numpy.linalg.svd implementations " \
#                "lead to different loadings")
#        print "Comparing loadings of PLSR and sklearn.pls ... OK!" \
#                " (err=%.4f, threshold=%0.4f)" % (np.sum((Yhat-Yhat_)**2), st)
#
#


def test_scale():

    pass
#    d = load_linnerud()
#    X = d.data
#    Y = d.target
#
#    # causes X[:, -1].std() to be zero
#    X[:, -1] = 1.0

#    methods = [PLSCanonical(), PLSRegression(), CCA(), PLSSVD(),
#               pls.PCA(), pls.SVD(), pls.PLSR(), pls.PLSC()]
#    names   = ["PLSCanonical", "PLSRegression", "CCA", "PLSSVD",
#               "pls.PCA", "pls.SVD", "pls.PLSR", "pls.PLSC"]
#    for i in xrange(len(methods)):
#        clf = methods[i]
#        print "Testing scale of "+names[i]
##        clf.set_params(scale=True)
#        clf.scale = True
#        clf.fit(X, Y)


if __name__ == "__main__":

    test_SVD_PCA()
    test_predictions()
    test_o2pls()
    test_regularisation()
    test_multiblock()
#    test_ista()
#    test_tv()
#    test_gl()

#    test_regression()

#    np.random.seed(42)
#    A = sparse.rand(10000, 10000, 0.01, format="lil").tocsr()
#    B = A.todense()
#
#    start = time()
#    svd = SVD(num_comp=1).fit(B)
#    u = svd.U
#    s = svd.S
#    v = svd.V
#    print s
#    print "time:", (time() - start), svd.get_algorithm().iterations
#    B = None
#
#    start = time()
#    v = algorithms.SparseSVD(max_iter=10).run(A)
#    u = A.dot(v)
#    s = np.sqrt(np.sum(u ** 2.0))
#    print s
#    print "time:", (time() - start)

#    start = time()
#    u, s, v = algorithms.SparseSVD(max_iter=50).run(A)
#    print s
#    print "time:", (time() - start)

#    import cProfile
#    import pstats
#    cProfile.run('test_tv()', 'prof_output')
#    p = pstats.Stats('prof_output')
#    p.sort_stats('calls').print_stats(20)

#    test_scale()