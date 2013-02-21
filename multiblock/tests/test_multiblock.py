import numpy as np
from numpy import dot
from multiblock.utils import *
from multiblock.utils.testing import *
from multiblock import *
import multiblock.prox_op as prox_op
from sklearn.datasets import load_linnerud

from math import log
from sklearn.pls import PLSRegression
from sklearn.pls import PLSCanonical
from sklearn.pls import CCA
from sklearn.pls import PLSSVD
from sklearn.pls import _center_scale_xy

def check_ortho(M, err_msg):
    K = np.dot(M.T, M)
    assert_array_almost_equal(K, np.diag(np.diag(K)), err_msg=err_msg)



def test_multiblock():

#    test_SVD_PCA()
#    test_eigsym()
#    test_o2pls()
#    test_predictions()
    test_L1_regularisation()



def test_o2pls():

    np.random.seed(42)

    # 000011100
    p  = np.vstack((np.zeros((4,1)), np.ones((3,1)), np.zeros((2,1))))
    # 001110000
    q  = np.vstack((np.zeros((2,1)), np.ones((3,1)), np.zeros((4,1))))
    t  = np.random.randn(10,1)

    # 111111000
    po = np.vstack((np.ones((4,1)), np.ones((2,1)), np.zeros((3,1))))
    to = np.random.randn(10,1)

    # 000111111
    qo = np.vstack((np.zeros((3,1)), np.ones((2,1)), np.ones((4,1))))
    uo = np.random.randn(10,1)

    Q, R = np.linalg.qr(np.hstack((t, to, uo)))
    t  = Q[:,[0]]
    to = Q[:,[1]]
    uo = Q[:,[2]]

    X = dot(t, p.T) + dot(to, po.T)
    Y = dot(t, q.T) + dot(uo, qo.T)

    svd = SVD(num_comp = 1)
    svd.fit(dot(X.T, Y))
    t_svd = dot(X, svd.U)
    u_svd = dot(Y, svd.V)

    o2pls = O2PLS(num_comp = [1, 1, 1], center = False, scale = False)
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

    W  = np.asarray([[-0.0000000035873],[-0.0000000035873],[-0.0000000035873],
                     [-0.0000000035873],[ 0.5773502687753],[ 0.5773502687753],
                     [ 0.5773502700183],[               0],[               0]])
    W, o2pls.W = direct(W, o2pls.W, compare = True)
    assert_array_almost_equal(W, o2pls.W, decimal=5, err_msg="O2PLS does not" \
            " give the correct weights in X")
    
    T  = np.asarray([[-0.3320724686729],[ 0.0924349866604],[-0.4330046394993],
                     [-1.0182038851347],[ 0.1565405190876],[ 0.1565295366198],
                     [-1.0557643604012],[-0.5130595674519],[ 0.3138616360639],
                     [-0.3627222076846]])
    T, o2pls.T = direct(T, o2pls.T, compare = True)
    assert_array_almost_equal(T, o2pls.T, decimal=5, err_msg="O2PLS does not" \
            " give the correct scores in X")

    P  = np.asarray([[-0.0000000035873],[-0.0000000035873],[-0.0000000035873],
                     [-0.0000000035873],[ 0.5773502687753],[ 0.5773502687753],
                     [ 0.5773502700183],[               0],[               0]])
    P, o2pls.P = direct(P, o2pls.P, compare = True)
    assert_array_almost_equal(P, o2pls.P, decimal=5, err_msg="O2PLS does not" \
            " give the correct loadings in X")

    Wo = np.asarray([[-0.4629100496613],[-0.4629100496613],[-0.4629100496613],
                     [-0.4629100496613],[-0.1543033544685],[-0.1543033544685],
                     [ 0.3086066967678],[               0],[               0]])
    Wo, o2pls.Wo = direct(Wo, o2pls.Wo, compare = True)
    assert_array_almost_equal(Wo, o2pls.Wo, decimal=5, err_msg="O2PLS does not" \
            " give the correct unique weights in X")

    To = np.asarray([[ 0.1166363558019],[ 0.3982081656371],[-0.4607187897875],
                     [ 0.7141661638895],[ 1.3523258395355],[ 0.5103916850288],
                     [ 0.0373342153684],[-0.5658484610118],[ 0.8644955981846],
                     [ 0.7835700349552]])
    To, o2pls.To = direct(To, o2pls.To, compare = True)
    assert_array_almost_equal(To, o2pls.To, decimal=5, err_msg="O2PLS does not" \
            " give the correct unique scores in X")

    Po = np.asarray([[-0.4629100479271],[-0.4629100479271],[-0.4629100479271],
                     [-0.4629100479271],[-0.4629100494572],[-0.4629100494572],
                     [ 0.0000000000150],[               0],[               0]])
    Po, o2pls.Po = direct(Po, o2pls.Po, compare = True)
    assert_array_almost_equal(Po, o2pls.Po, decimal=5, err_msg="O2PLS does not" \
            " give the correct unique loadings in X")

    C  = np.asarray([[               0],[               0],[ 0.5773502653499],
                     [ 0.5773502711095],[ 0.5773502711095],[-0.0000000006458],
                     [-0.0000000006458],[-0.0000000006458],[-0.0000000006458]])
    C, o2pls.C = direct(C, o2pls.C, compare = True)
    assert_array_almost_equal(C, o2pls.C, decimal=5, err_msg="O2PLS does not" \
            " give the correct weights in Y")

    U  = np.asarray([[-0.3320724688962],[ 0.0924349962939],[-0.4330046405197],
                     [-1.0182039005573],[ 0.1565405177895],[ 0.1565295466182],
                     [-1.0557643651281],[-0.5130595684168],[ 0.3138616457148],
                     [-0.3627222001620]])
    U, o2pls.U = direct(U, o2pls.U, compare = True)
    assert_array_almost_equal(U, o2pls.U, decimal=5, err_msg="O2PLS does not" \
            " give the correct scores in Y")

    Q  = np.asarray([[               0],[               0],[ 0.5773502653499],
                     [ 0.5773502711095],[ 0.5773502711095],[-0.0000000006458],
                     [-0.0000000006458],[-0.0000000006458],[-0.0000000006458]])
    Q, o2pls.Q = direct(Q, o2pls.Q, compare = True)
    assert_array_almost_equal(Q, o2pls.Q, decimal=5, err_msg="O2PLS does not" \
            " give the correct loadings in Y")

    Co = np.asarray([[-0.0000000000000],[-0.0000000000000],[ 0.3086067007710],
                     [-0.1543033498817],[-0.1543033498817],[-0.4629100497585],
                     [-0.4629100497585],[-0.4629100497585],[-0.4629100497585]])
    Co, o2pls.Co = direct(Co, o2pls.Co, compare = True)
    assert_array_almost_equal(Co, o2pls.Co, decimal=5, err_msg="O2PLS does not" \
            " give the correct unique weights in Y")

    Uo = np.asarray([[-1.8954219076301],[ 0.0598668575527],[-0.0696963020914],
                     [ 0.4526578568357],[-0.1430318048907],[-0.3371378335299],
                     [ 0.5494330695544],[-0.3942048787367],[ 0.3248866786239],
                     [-0.4046679696962]])
    Uo, o2pls.Uo = direct(Uo, o2pls.Uo, compare = True)
    assert_array_almost_equal(Uo, o2pls.Uo, decimal=5, err_msg="O2PLS does not" \
            " give the correct unique scores in Y")

    Qo = np.asarray([[               0],[               0],[ 0.0000000008807],
                     [-0.4629100498755],[-0.4629100498755],[-0.4629100499092],
                     [-0.4629100499092],[-0.4629100499092],[-0.4629100499092]])
    Qo, o2pls.Qo = direct(Qo, o2pls.Qo, compare = True)
    assert_array_almost_equal(Qo, o2pls.Qo, decimal=5, err_msg="O2PLS does not" \
            " give the correct unique loadings in Y")



    np.random.seed(42)

    # Test primal version
    X = np.random.rand(10,5)
    Y = np.random.rand(10,5)

    X = center(X)
    Y = center(Y)

#    print X
#    print Y

    o2pls = O2PLS(num_comp = [3, 2, 2], center = False, scale = False,
                  tolerance = 5e-12)
    o2pls.fit(X, Y)

    Xhat = dot(o2pls.T, o2pls.P.T) + dot(o2pls.To, o2pls.Po.T)
    assert_array_almost_equal(X, Xhat, decimal=5, err_msg="O2PLS does not" \
            " give a correct reconstruction of X")
    Yhat = dot(o2pls.U, o2pls.Q.T) + dot(o2pls.Uo, o2pls.Qo.T)
    assert_array_almost_equal(Y, Yhat, decimal=5, err_msg="O2PLS does not" \
            " give a correct reconstruction of Y")

    W  = np.asarray([[ 0.28574946790251, 0.80549463591904, 0.46033811220009],
                     [-0.82069177928760, 0.03104339272344, 0.41068049575333],
                     [ 0.23145983182932,-0.45751789277098, 0.78402240214039],
                     [-0.04763457200240, 0.32258037076527,-0.04720569565472],
                     [ 0.43470626726892,-0.19192181081164, 0.05010836360956]])
    W, o2pls.W = direct(W, o2pls.W, compare = True)
    assert_array_almost_equal(W, o2pls.W, decimal=5, err_msg="O2PLS does not" \
            " give the correct weights in X")

    C  = np.asarray([[ 0.33983623057749,-0.08643744675915, 0.61598479965533],
                     [-0.21522532354066, 0.76365073948393,-0.04894422260217],
                     [-0.80644847592354,-0.40113340935286, 0.35039634003963],
                     [-0.39193158427033, 0.44086245939825, 0.16315472674870],
                     [-0.18498617630960,-0.23259061820651,-0.68466789737349]])
    C, o2pls.C = direct(C, o2pls.C, compare = True)
    assert_array_almost_equal(C, o2pls.C, decimal=5, err_msg="O2PLS does not" \
            " give the correct weights in Y")

    Wo = np.asarray([[ 0.09275972900527, 0.22138222202790],
                     [-0.32377665878465, 0.22806033590264],
                     [ 0.13719457299956,-0.32185438652509],
                     [-0.48063246869535,-0.81267269197421],
                     [-0.79795638168787, 0.36735710766490]])
    Wo, o2pls.Wo = direct(Wo, o2pls.Wo, compare = True)
    assert_array_almost_equal(Wo, o2pls.Wo, decimal=5, err_msg="O2PLS does not" \
            " give the correct unique weights in X")

    Co = np.asarray([[-0.28602514350110,-0.64481954689928],
                     [ 0.41206605980471,-0.44533317148328],
                     [ 0.21923094429525,-0.13376487405816],
                     [-0.75775220257515, 0.22632291041065],
                     [-0.35516274044178,-0.56282414394259]])
    Co, o2pls.Co = direct(Co, o2pls.Co, compare = True)
    assert_array_almost_equal(Co, o2pls.Co, decimal=5, err_msg="O2PLS does not" \
            " give the correct unique weights in Y")



    np.random.seed(43)

    # Test dual version
    X = np.random.rand(5,10)
    Y = np.random.rand(5,10)

    X = center(X)
    Y = center(Y)

#    print X
#    print Y

    o2pls = O2PLS(num_comp = [3, 2, 2], center = False, scale = False,
                  tolerance = 5e-12)
    o2pls.fit(X, Y)

    Xhat = dot(o2pls.T, o2pls.P.T) + dot(o2pls.To, o2pls.Po.T)
    assert_array_almost_equal(X, Xhat, decimal=5, err_msg="O2PLS does not" \
            " give a correct reconstruction of X")
    Yhat = dot(o2pls.U, o2pls.Q.T) + dot(o2pls.Uo, o2pls.Qo.T)
    assert_array_almost_equal(Y, Yhat, decimal=5, err_msg="O2PLS does not" \
            " give a correct reconstruction of Y")

    W = np.asarray([[ 0.660803597278207, 0.283434091506369,-0.101117338537766],
                    [ 0.199245987632961,-0.371782870650567,-0.105580628700634],
                    [ 0.049056826578120, 0.510007873722491,-0.052587642609540],
                    [ 0.227075734276342, 0.321380074146550, 0.386973037543944],
                    [-0.366361672688720, 0.047162463451118,-0.513205123937845],
                    [-0.231028371367863,-0.053688420486769, 0.613919597967758],
                    [-0.441503068093162, 0.531948476266181, 0.122002207828563],
                    [ 0.155295133662634,-0.243203951406354, 0.327486426731878],
                    [ 0.177637796765300, 0.210455027475178, 0.022448447240951],
                    [ 0.177420328030713, 0.162892673626251,-0.251399720667828]])
    W, o2pls.W = direct(W, o2pls.W, compare = True)
    assert_array_almost_equal(W, o2pls.W, decimal=5, err_msg="O2PLS does not" \
            " give the correct weights in X")

    C = np.asarray([[ 0.029463187629622, 0.343448566391335, 0.439488969936543],
                    [-0.308514636956793, 0.567738611008308, 0.288894827826079],
                    [-0.041368032721661, 0.240281903256231,-0.165061925028986],
                    [ 0.507488061231666, 0.220214013827959,-0.197749809945398],
                    [-0.032831353872698,-0.456872251088426, 0.330712836574687],
                    [-0.088465736946256, 0.137459131512888,-0.199716885766439],
                    [ 0.575982215268074, 0.265756602507649, 0.284152520405287],
                    [-0.174758650490361,-0.236798361028871, 0.559859131854712],
                    [ 0.474892087711923,-0.156220599204360, 0.286373509842812],
                    [-0.219026289120523, 0.273412086540475, 0.177725330447690]])
    C, o2pls.C = direct(C, o2pls.C, compare = True)
    assert_array_almost_equal(C, o2pls.C, decimal=5, err_msg="O2PLS does not" \
            " give the correct weights in Y")

    Wo = np.asarray([[-0.072619636061551, 0],
                     [-0.186139609163295, 0],
                     [-0.193345377028291, 0],
                     [-0.416556339157776, 0],
                     [-0.556155354111327, 0],
                     [ 0.042355493010538, 0],
                     [ 0.170141007666872, 0],
                     [-0.158346774720306, 0],
                     [-0.079654387080921, 0],
                     [ 0.614579177338253, 0]])
    Wo, o2pls.Wo = direct(Wo, o2pls.Wo, compare = True)
    assert_array_almost_equal(Wo, o2pls.Wo, decimal=5, err_msg="O2PLS does not" \
            " give the correct unique weights in X")

    Co = np.asarray([[-0.143370887251568, 0],
                     [ 0.268379847579299, 0],
                     [ 0.538254868418912, 0],
                     [-0.076765726998542, 0],
                     [ 0.011841641465690, 0],
                     [-0.682168209668947, 0],
                     [-0.047632336295968, 0],
                     [-0.032431712285857, 0],
                     [ 0.059518204830295, 0],
                     [-0.373428712071223, 0]])
    Co, o2pls.Co = direct(Co, o2pls.Co, compare = True)
    assert_array_almost_equal(Co, o2pls.Co, decimal=5, err_msg="O2PLS does not" \
            " give the correct unique weights in Y")



def test_eigsym():

    d = load_linnerud()
    X = d.data

    n = 3
    X = dot(X.T, X)

    eig = EIGSym(num_comp = n, tolerance = 5e-12)
    eig.fit(X)
    Xhat = dot(eig.V, dot(eig.D, eig.V.T))

    assert_array_almost_equal(X, Xhat, decimal=4, err_msg="EIGSym does not" \
            " give the correct reconstruction of the matrix")

    [D,V] = np.linalg.eig(X)
    # linalg.eig does not return the eigenvalues in order, so need to sort
    idx = np.argsort(D, axis=None).tolist()[::-1]
    D = D[idx]
    V = V[:,idx]

    Xhat = dot(V, dot(np.diag(D), V.T))

    V, eig.V = direct(V, eig.V, compare = True)
    assert_array_almost_equal(V, eig.V, decimal=5, err_msg="EIGSym does not" \
            " give the correct eigenvectors")



def test_SVD_PCA():

    # Assure same answer every time
    np.random.seed(42)

    # Compare SVD with and without sparsity constraint to numpy.linalg.svd
    Xtr = np.random.rand(6,6)
    #Xte = np.random.rand(2,6)
    num_comp = 3
    tol = 5e-12

    Xtr, m = center(Xtr, return_means = True)
    Xtr, s = scale(Xtr, return_stds = True)
    #Xte = (Xte - m) / s

    for st in [0.1, 0.01, 0.001, 0.0001, 0]:
        # multiblock.SVD
        svd = SVD(num_comp = num_comp, tolerance = tol, max_iter = 1000,
                  prox_op = prox_op.L1(st))
        svd.fit(Xtr)

        # numpy.lialg.svd
        U, S, V = np.linalg.svd(Xtr)
        V = V.T
        S = np.diag(S)
        U = U[:,0:num_comp]
        S = S[:,0:num_comp]
        V = V[:,0:num_comp]
        #SVDte = dot(Xte, V)

        if st < tol:
            num_decimals = 5
        else:
            num_decimals = int(log(1./st, 10) + 0.5)
        svd.V, V = direct(svd.V, V, compare = True)
        assert_array_almost_equal(svd.V, V, decimal=num_decimals-2,
                err_msg="sklearn.NIPALS.SVD and numpy.linalg.svd implementations " \
                "lead to different loadings")



    # Compare PCA with sparsity constraint to numpy.linalg.svd
    Xtr = np.random.rand(5,5)
    num_comp = 5
    tol = 5e-9

    Xtr, m = center(Xtr, return_means = True)
    Xtr, s = scale(Xtr, return_stds = True)

    for st in [0.1, 0.01, 0.001, 0.0001, 0]:
        pca = PCA(center = False, scale = False, num_comp = num_comp,
                  tolerance = tol, max_iter = 500,
                  prox_op = prox_op.L1(st))
        pca.fit(Xtr)
        Tte = pca.transform(Xtr)
        U, S, V = np.linalg.svd(Xtr)
        V = V.T
        US = dot(U,np.diag(S))
        US = US[:,0:num_comp]
        V  = V[:,0:num_comp]

        if st < tol:
            num_decimals = 5
        else:
            num_decimals = int(log(1./st, 10) + 0.5)
        assert_array_almost_equal(Xtr, dot(pca.T,pca.P.T), decimal = num_decimals-1,
                err_msg="Model does not equal the matrices")



    # Compare PCA without the sparsity constraint to numpy.linalg.svd
    Xtr = np.random.rand(50,50)
    Xte = np.random.rand(20,50)
    num_comp = 3

    Xtr, m = center(Xtr, return_means = True)
    Xtr, s = scale(Xtr, return_stds = True)
    Xte = (Xte - m) / s

    pca = PCA(center = False, scale = False, num_comp = num_comp,
                  tolerance = 5e-12, max_iter = 1000)
    pca.fit(Xtr)
    pca.P, pca.T = direct(pca.P, pca.T)
    Tte = pca.transform(Xte)

    U, S, V = np.linalg.svd(Xtr)
    V = V.T
    US = dot(U,np.diag(S))
    US = US[:,0:num_comp]
    V  = V[:,0:num_comp]
    V, US = direct(V, US)
    SVDte = dot(Xte, V)

    assert_array_almost_equal(pca.P, V, decimal = 2, err_msg = "NIPALS PCA and "
            "numpy.linalg.svd implementations lead to different loadings")

    assert_array_almost_equal(pca.T, US, decimal = 2, err_msg = "NIPALS PCA and "
            "numpy.linalg.svd implementations lead to different scores")

    assert_array_almost_equal(Tte, SVDte, decimal = 2, err_msg = "NIPALS PCA and "
            "numpy.linalg.svd implementations lead to different scores")



    # Compare PCA without the sparsity constraint to numpy.linalg.svd
    X = np.random.rand(50,100)
    num_comp = 50

    X = center(X)
    X = scale(X)

    pca = PCA(center = False, scale = False, num_comp = num_comp,
                  tolerance = 5e-12, max_iter = 1000)
    pca.fit(X)
    Xhat_1 = dot(pca.T, pca.P.T)

    U, S, V = np.linalg.svd(X, full_matrices = False)
    Xhat_2 = dot(U, dot(np.diag(S), V))

    assert_array_almost_equal(X, Xhat_1, decimal = 2, err_msg = "PCA performs "
            " a faulty reconstruction of X")

    assert_array_almost_equal(Xhat_1, Xhat_2, decimal = 2, err_msg = "PCA and "
            "numpy.linalg.svd implementations lead to different reconstructions")



    # Compare PCA without the sparsity constraint to numpy.linalg.svd
    X = np.random.rand(100,50)
    num_comp = 50

    X = center(X)
    X = scale(X)

    pca = PCA(center = False, scale = False, num_comp = num_comp,
                  tolerance = 5e-12, max_iter = 1000)
    pca.fit(X)
    Xhat_1 = dot(pca.T, pca.P.T)

    U, S, V = np.linalg.svd(X, full_matrices = False)
    Xhat_2 = dot(U, dot(np.diag(S), V))

    assert_array_almost_equal(X, Xhat_1, decimal = 2, err_msg = "PCA performs "
            " a faulty reconstruction of X")

    assert_array_almost_equal(Xhat_1, Xhat_2, decimal = 2, err_msg = "PCA and "
            "numpy.linalg.svd implementations lead to different reconstructions")



def test_predictions():

    d = load_linnerud()
    X = d.data
    Y = d.target
    tol = 5e-12
    miter = 1000
    num_comp = 2
    Xorig = X.copy()
    Yorig = Y.copy()
    SSY = np.sum(Yorig**2)
    SSX = np.sum(Xorig**2)
#    center = True
    scale  = False


    pls1 = PLSRegression(n_components = num_comp, scale = scale,
                 tol = tol, max_iter = miter, copy = True)
    pls1.fit(Xorig, Yorig)
    Yhat1 = pls1.predict(Xorig)

    SSYdiff1 = np.sum((Yorig-Yhat1)**2)
#    print "PLSRegression: R2Yhat = %.4f" % (1 - (SSYdiff1 / SSY))

    # Compare PLSR and sklearn.PLSRegression
    pls3 = PLSR(num_comp = num_comp, center = True, scale = scale,
                tolerance = tol, max_iter = miter)
    pls3.fit(X, Y)
    Yhat3 = pls3.predict(X)

    assert_array_almost_equal(Yhat1, Yhat3, decimal = 5,
            err_msg = "PLSR gives wrong prediction")

    SSYdiff3 = np.sum((Yorig-Yhat3)**2)
#    print "PLSR         : R2Yhat = %.4f" % (1 - (SSYdiff3 / SSY))

    assert abs(SSYdiff1 - SSYdiff3) < 0.00005


    pls2 = PLSCanonical(n_components = num_comp, scale = scale,
                        tol = tol, max_iter = miter, copy = True)
    pls2.fit(Xorig, Yorig)
    Yhat2 = pls2.predict(Xorig)

    SSYdiff2 = np.sum((Yorig-Yhat2)**2)
#    print "PLSCanonical : R2Yhat = %.4f" % (1 - (SSYdiff2 / SSY))

    # Compare PLSC and sklearn.PLSCanonical
    pls4 = PLSC(num_comp = num_comp, center = True, scale = scale,
                tolerance = tol, max_iter = miter)
    pls4.fit(X, Y)
    Yhat4 = pls4.predict(X)

    SSYdiff4 = np.sum((Yorig-Yhat4)**2)
#    print "PLSC         : R2Yhat = %.4f" % (1 - (SSYdiff4 / SSY))

    # Compare O2PLS and sklearn.PLSCanonical
    pls5 = O2PLS(num_comp = [num_comp, 1, 0], center = True, scale = scale,
                 tolerance = tol, max_iter = miter)
    pls5.fit(X, Y)
    Yhat5 = pls5.predict(X)

    SSYdiff5 = np.sum((Yorig-Yhat5)**2)
#    print "O2PLS X-Y    : R2Yhat = %.4f" % (1 - (SSYdiff5 / SSY))

    assert SSYdiff2 > SSYdiff4
    assert SSYdiff2 > SSYdiff5


    # Make sure O2PLS is symmetric!
    pls6 = O2PLS(num_comp = [num_comp, 0, 1], center = True, scale = scale,
                 tolerance = tol, max_iter = miter)
    pls6.fit(Y, X)
    Yhat6 = pls6.predict(Y = X)

    pls5.W, pls6.C = direct(pls5.W, pls6.C, compare = True)
    assert_array_almost_equal(pls5.W, pls6.C, decimal = 5,
            err_msg = "O2PLS is not symmetic")
    pls5.T, pls6.U = direct(pls5.T, pls6.U, compare = True)
    assert_array_almost_equal(pls5.T, pls6.U, decimal = 5,
            err_msg = "O2PLS is not symmetic")
    pls5.P, pls6.Q = direct(pls5.P, pls6.Q, compare = True)
    assert_array_almost_equal(pls5.P, pls6.Q, decimal = 5,
            err_msg = "O2PLS is not symmetic")

    pls5.C, pls6.W = direct(pls5.C, pls6.W, compare = True)
    assert_array_almost_equal(pls5.C, pls6.W, decimal = 5,
            err_msg = "O2PLS is not symmetic")
    pls5.U, pls6.T = direct(pls5.U, pls6.T, compare = True)
    assert_array_almost_equal(pls5.U, pls6.T, decimal = 5,
            err_msg = "O2PLS is not symmetic")
    pls5.Q, pls6.P = direct(pls5.Q, pls6.P, compare = True)
    assert_array_almost_equal(pls5.Q, pls6.P, decimal = 5,
            err_msg = "O2PLS is not symmetic")

    assert_array_almost_equal(pls5.Bx, pls6.By, decimal = 5,
            err_msg = "O2PLS is not symmetic")
    assert_array_almost_equal(pls5.By, pls6.Bx, decimal = 5,
            err_msg = "O2PLS is not symmetic")

    SSYdiff6 = np.sum((Yorig-Yhat6)**2)
#    print "O2PLS Y-X    : R2Yhat = %.4f" % (1 - (SSYdiff6 / SSY))

    assert abs(SSYdiff6 - SSYdiff5) < TOLERANCE



def test_L1_regularisation():

    d = load_linnerud()
    X = d.data
    Y = d.target
#    print X.shape
#    print Y.shape
    tol = 5e-12
    miter = 1000
    num_comp = 2
    Xorig = X.copy()
    Yorig = Y.copy()
    center = True
    scale  = True
    inf = 2**30
    SSY = np.sum(Yorig**2)


    # Test Sparse PLSR
    pls = PLSR(num_comp = num_comp, center = center, scale = scale,
               tolerance = tol, max_iter = miter)
    pls.fit(X, Y)
    Yhat = pls.predict(X)
    SSYdiff = np.sum((Yorig-Yhat)**2)
    R2Yhat = (1 - (SSYdiff / SSY))
    print "PLS : R2Yhat = %.6f" % R2Yhat


    spls1 = PLSR(num_comp = num_comp, center = center, scale = scale,
                 tolerance = tol, max_iter = miter,
                 prox_op = prox_op.L1(0., 0., normaliser = [norm, normI]))
    spls1.fit(X, Y)
    Yhat1 = spls1.predict(X)
    SSYdiff1 = np.sum((Yorig-Yhat1)**2)
    print "sPLS: R2Yhat = %.6f" % (1 - (SSYdiff1 / SSY))
    assert abs(R2Yhat - (1 - (SSYdiff1 / SSY))) < TOLERANCE
    assert_array_almost_equal(Yhat, Yhat1, decimal = 5,
            err_msg = "Sparse PLS with no thresholding does not give correct result")


    spls2 = PLSR(num_comp = num_comp, center = center, scale = scale,
                 tolerance = tol, max_iter = miter,
                 prox_op = prox_op.L1_binsearch(float('Inf'), float('Inf'), normaliser=[norm, normI]))
    spls2.fit(X, Y)
    Yhat2 = spls2.predict(X)
    SSYdiff2 = np.sum((Yorig-Yhat2)**2)
    print "sPLS: R2Yhat = %.6f" % (1 - (SSYdiff2 / SSY))
    assert abs(R2Yhat - (1 - (SSYdiff1 / SSY))) < TOLERANCE
    assert_array_almost_equal(Yhat, Yhat2, decimal = 5,
            err_msg = "Sparse PLS with no thresholding does not give correct result")


    spls3 = PLSR(num_comp = num_comp, center = center, scale = scale,
                 tolerance = tol, max_iter = miter,
                 prox_op = prox_op.L0_binsearch(inf, inf, normaliser=[norm, normI]))
    spls3.fit(X, Y)
    Yhat3 = spls3.predict(X)
    SSYdiff3 = np.sum((Yorig-Yhat3)**2)
    print "sPLS: R2Yhat = %.6f" % (1 - (SSYdiff3 / SSY))
    assert_array_almost_equal(Yhat, Yhat3, decimal = 5,
            err_msg = "Sparse PLS with no thresholding does not give correct result")


    spls4 = PLSR(num_comp = num_comp, center = center, scale = scale,
                 tolerance = tol, max_iter = miter,
                 prox_op = prox_op.L0_by_count(inf, inf, normaliser=[norm, normI]))
    spls4.fit(X, Y)
    Yhat4 = spls4.predict(X)
    SSYdiff4 = np.sum((Yorig-Yhat4)**2)
    print "sPLS: R2Yhat = %.6f" % (1 - (SSYdiff4 / SSY))
    assert_array_almost_equal(Yhat, Yhat4, decimal = 5,
            err_msg = "Sparse PLS with no thresholding does not give correct result")

    num_comp = 1

    n = 11
    Y = rand(10,1)
    X = zeros(10,n)
    for j in xrange(n-1):
        x = rand(10,1)
        while abs(abs(corr(x, Y)) - j/(n-1.0)) > 0.01:
            x = rand(10,1)
        if corr(x, Y) < 0:
            x *= -1
        X[:,j] = x.ravel()
    X[:,n-1] = Y.ravel()
    SSX = np.sum(X**2)
    SSY = np.sum(Y**2)

#    for j in xrange(11):
#        print corr(X[:,[j]], y)

    # Test Sparse PLSR
    pls = PLSR(num_comp = num_comp, center = center, scale = scale,
               tolerance = tol, max_iter = miter)
    pls.fit(X, Y)
    Yhat = pls.predict(X)
    SSYdiff = np.sum((Y-Yhat)**2)
    R2Yhat = (1 - (SSYdiff / SSY))
    print
    print "PLS :           R2Yhat = %.6f" % R2Yhat

    nonzero = []
    for l in np.linspace(0, 0.6, 13).tolist():
        spls1 = PLSR(num_comp = num_comp, center = center, scale = scale,
                     tolerance = tol, max_iter = miter,
                     prox_op = prox_op.L1(l, 0, normaliser = [norm, normI]))
        spls1.fit(X, Y)
        Yhat1 = spls1.predict(X)
        SSYdiff1 = np.sum((Y-Yhat1)**2)
        R2Yhat1 = 1 - (SSYdiff1 / SSY)
        nonzero.append(np.count_nonzero(spls1.W))
        print "sPLS: l = %.2f, R2Yhat = %.6f, nonzero: %d" % (l, R2Yhat1, nonzero[-1])

        assert all(x <= y for x, y in zip(np.abs(spls1.W)[:,0], (np.abs(spls1.W)[:,0])[1:]))
        assert all(x >= y for x, y in zip(nonzero, nonzero[1:]))
    assert abs(R2Yhat1 - 1) < TOLERANCE


    print
    print "PLS :          R2Yhat = %.6f" % R2Yhat
    nonzero = []
    for s in [float('Inf'), 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0]:
        spls2 = PLSR(num_comp = num_comp, center = center, scale = scale,
                     tolerance = tol, max_iter = miter,
                     prox_op = prox_op.L1_binsearch(s, float('Inf'), normaliser = [norm, normI]))
        spls2.fit(X, Y)
        Yhat2 = spls2.predict(X)
        SSYdiff2 = np.sum((Y-Yhat2)**2)
        R2Yhat2 = 1 - (SSYdiff2 / SSY)
        nonzero.append(np.count_nonzero(spls2.W))
        print "sPLS: s = %.1f, R2Yhat = %.6f, nonzero: %d" % (s, R2Yhat2, nonzero[-1])

        assert all(x <= y for x, y in zip(np.abs(spls2.W)[:,0], (np.abs(spls2.W)[:,0])[1:]))
        assert all(x >= y for x, y in zip(nonzero, nonzero[1:]))
    assert abs(R2Yhat2 - 1) < TOLERANCE


    print
    print "PLS :          R2Yhat = %.6f" % R2Yhat
    nonzero = []
    for n in [100, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]:
        spls3 = PLSR(num_comp = num_comp, center = center, scale = scale,
                     tolerance = tol, max_iter = miter,
                     prox_op = prox_op.L0_binsearch(n, 100, normaliser = [norm, normI]))
        spls3.fit(X, Y)
        Yhat3 = spls3.predict(X)
        SSYdiff3 = np.sum((Y-Yhat3)**2)
        R2Yhat3 = 1 - (SSYdiff3 / SSY)
        nonzero.append(np.count_nonzero(spls3.W))
        print "sPLS: n = %3d, R2Yhat = %.6f, nonzero: %d" % (n, R2Yhat3, nonzero[-1])

        assert all(x <= y for x, y in zip(np.abs(spls3.W)[:,0], (np.abs(spls3.W)[:,0])[1:]))
        assert all(x >= y for x, y in zip(nonzero, nonzero[1:]))
    assert abs(R2Yhat3 - 1) < TOLERANCE


    print
    print "PLS :          R2Yhat = %.6f" % R2Yhat
    nonzero = []
    for n in [100]+range(11, -1, -1):
        spls4 = PLSR(num_comp = num_comp, center = center, scale = scale,
                     tolerance = tol, max_iter = miter,
                     prox_op = prox_op.L0_by_count(n, 100, normaliser = [norm, normI]))
        spls4.fit(X, Y)
        Yhat4 = spls4.predict(X)
        SSYdiff4 = np.sum((Y-Yhat4)**2)
        R2Yhat4 = 1 - (SSYdiff4 / SSY)
        nonzero.append(np.count_nonzero(spls4.W))
        print "sPLS: n = %3d, R2Yhat = %.6f, nonzero: %d" % (n, R2Yhat4, nonzero[-1])

        assert all(x <= y for x, y in zip(np.abs(spls4.W)[:,0], (np.abs(spls4.W)[:,0])[1:]))
        assert all(x >= y for x, y in zip(nonzero, nonzero[1:]))
    assert abs(R2Yhat4 - 1) < TOLERANCE

    return

    spls = spls3
#    print
#    print pls.T
    print
    print pls.W
    print
    print spls.W
    print
    print pls.C
    print
    print spls.C
    print
    print sstot(pls.T)
    print sstot(spls.T)
    print sstot(pls.U)
    print sstot(spls.U)
    print sstot(pls.C)
    print sstot(spls.C)
#    print np.sum(pls.C**2, axis = 0)
#    print np.sum(spls.C**2, axis = 0)
#    print pls.T - spls.T





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

    d = load_linnerud()
    X = d.data
    Y = d.target

    # causes X[:, -1].std() to be zero
    X[:, -1] = 1.0

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

#    from time import clock
##    timesPCA  = []
#    timesFast = []
#    timesSVD  = []
#    Xrange = [8000] # range(50, 1000, 50) # + range(100, 1000, 100)
#    Yrange = range(1,6, 2)#np.linspace(1, n, 2).tolist()
#    for n in Xrange:
#        print "Size:", n
##        tmPCA  = []
#        tmFast = []
#        tmSVD  = []
#        for num_comp in Yrange:
#            print "Comps:", num_comp
#            ncomp = int(num_comp + 0.5)
#            Xtr = np.random.rand(n,n)
##            Xte = np.random.rand(20,50)
#            tol   = 5e-14
#            miter = 1000
#            Xtr, m = pls.center(Xtr, return_means = True)
#            Xtr, s = pls.scale(Xtr, return_stds = True)
##            Xte = (Xte - m) / s
##            start_time = clock()
##            pca = pls.PCA(center = False, scale = False, num_comp = ncomp,
##                          tolerance = tol, max_iter = miter)
##            pca.fit(Xtr)
##            tmPCA.append(clock() - start_time)
#
#            start_time = clock()
#            fast = pls.PCAfast(center = False, scale = False, num_comp = ncomp,
#                               tolerance = tol, max_iter = miter)
#            fast.fit(Xtr)
#            tmFast.append(clock() - start_time)
#
#            start_time = clock()
#            U, S, V = np.linalg.svd(Xtr)
#            T = dot(U, np.diag(S))
#            P = V.T
#            T = T[:,0:ncomp]
#            P = P[:,0:ncomp]
#            tmSVD.append(clock() - start_time)
#
##            P, pca.P, fast.P = pls.direct(P, pca.P, fast.P, compare = True)
##            T, pca.T, fast.T = pls.direct(T, pca.T, fast.T, compare = True)
#            P, fast.P = pls.direct(P, fast.P, compare = True)
#            T, fast.T = pls.direct(T, fast.T, compare = True)
##        timesPCA.append(tmPCA)
#        timesFast.append(tmFast)
#        timesSVD.append(tmSVD)
##            assert_array_almost_equal(T, pca.T, decimal=2,
##                    err_msg="numpy.linalg.svd and PCA differ!")
##            assert_array_almost_equal(T, fast.T, decimal=2,
##                    err_msg="numpy.linalg.svd and PCAfast differ!")
#
##    import matplotlib.pyplot as plt
##    from matplotlib.pyplot import pcolor
##
##    times = np.array(times)
##    plt.plot()
##    pcolor(times)
##    plt.show()
#
##    timesPCA = np.array(timesPCA)
#    timesFast = np.array(timesFast)
#    timesSVD = np.array(timesSVD)
#
##    np.save("timesPCA", timesPCA)
##    np.save("timesFast", timesPCA)
##    np.save("timesSVD", timesPCA)
#
##    print timesPCA
#    print timesFast
#    print timesSVD
#
##    from pylab import *
##    from mpl_toolkits.mplot3d import Axes3D
##
##    fig = figure()
##    ax = Axes3D(fig)
##    X = np.array(Xrange)#np.arange(-4, 4, 0.25)
##    Y = np.array(Yrange)#np.arange(-4, 4, 0.25)
##    X, Y = np.meshgrid(X, Y)
##
###    Z = timesPCA
###    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='Reds')
##    Z = timesFast
##    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='Greens')
##    Z = timesSVD
##    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='Blues')
##
##    show()
#
#
##        print pca.P
##        print fast.P
##        print P
#
##    Tpca  = pca.transform(Xte)
##    Tfast = fast.transform(Xte)
#
    test_multiblock()
#    test_scale()

#    import math
#    import numpy as np
#    from time import clock
#    A = np.random.rand(100000000,1)
#    num = int(1)
#
#    t = clock()
#    for i in xrange(num):
#        n = math.sqrt(np.dot(A.T,A))
#    print clock() - t
#
##    t = clock()
##    for i in xrange(num):
##        n = math.sqrt(sum(abs(A)**2))
##    print clock() - t
#
##    t = clock()
##    for i in xrange(num):
##        n = np.sqrt((np.abs(A)**2).sum())
##    print clock() - t
#
#    t = clock()
#    for i in xrange(num):
#        n = np.linalg.norm(A)
#    print clock() - t

#    from sandlada.multiblock import SVD
#    A = np.random.rand(6,5)
#    print A
#    svd = SVD(num_comp = 2)
#    svd.fit(A)
#
#    print svd.U
#    print svd.S
#    print svd.V
#    print
#
#    U, S, V = np.linalg.svd(A)
#
#    print U
#    print np.diag(S)
#    print (V.T)
