# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.functions.objectives` module contains multiblock objective
functions.

Created on Tue Feb  4 08:51:43 2014

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import numbers

import numpy as np

import parsimony.functions.interfaces as interfaces
import interfaces as mb_interfaces

__all__ = ["LatentVariableCovariance"]


class LatentVariableCovariance(mb_interfaces.MultiblockFunction,
                               mb_interfaces.MultiblockGradient,
                               mb_interfaces.MultiblockLipschitzContinuousGradient,
                               interfaces.Eigenvalues):

    def __init__(self, X, unbiased=True):

        self.X = X
        if unbiased:
            self.n = X[0].shape[0] - 1.0
        else:
            self.n = X[0].shape[0]

        self.reset()

    def reset(self):

        self._lambda_max = None

    def f(self, w):
        """Function value.

        From the interface "Function".
        """
        wX = np.dot(self.X[0], w[0]).T
        Yc = np.dot(self.X[1], w[1])
        wXYc = np.dot(wX, Yc)
        return -wXYc[0, 0] / float(self.n)

    def grad(self, w, index):
        """Gradient of the function.

        From the interface "MultiblockGradient".
        """
        index = int(index)
        grad = -np.dot(self.X[index].T,
                       np.dot(self.X[1 - index], w[1 - index])) / float(self.n)

#        def fun(x):
#            w_ = [0, 0]
#            w_[index] = x
#            w_[1 - index] = w[1 - index]
#            return self.f(w_)
#        approx_grad = utils.approx_grad(fun, w[index], eps=1e-6)
#        print "LatentVariableCovariance:", maths.norm(grad - approx_grad)

        return grad

    def L(self, w, index):
        """Lipschitz constant of the gradient with given index.

        From the interface "MultiblockLipschitzContinuousGradient".
        """
#        return maths.norm(self.grad(w, index))

#        if self._lambda_max is None:
#            self._lambda_max = self.lambda_max()

        return 0  # self._lambda_max

    def lambda_max(self):
        """ Largest eigenvalue of the corresponding covariance matrix.

        From the interface "Eigenvalues".
        """
        # Note that we can save the state here since lmax(A) does not

        from algorithms import FastSVDProduct
        svd = FastSVDProduct()
        v = svd(self.X[0].T, self.X[1], max_iter=100)
        s = np.dot(self.X[0].T, np.dot(self.X[1], v))

        return np.sum(s ** 2.0) / (self.n ** 2.0)


class GeneralisedMultiblock(mb_interfaces.MultiblockFunction,
                            mb_interfaces.MultiblockGradient,
#                            mb_interfaces.MultiblockProximalOperator,
                            mb_interfaces.MultiblockProjectionOperator,
                            interfaces.StepSize,
#                            LipschitzContinuousGradient,
#                            NesterovFunction, Continuation, DualFunction
                            ):

    def __init__(self, X, functions):

        self.X = X
        self.functions = functions

        self.reset()

    def reset(self):

        for i in xrange(len(self.functions)):
            for j in xrange(len(self.functions[i])):
                if i == j:
                    for k in xrange(len(self.functions[i][j])):
                        self.functions[i][j][k].reset()
                else:
                    if not self.functions[i][j] is None:
                        self.functions[i][j].reset()

    def f(self, w):
        """Function value.
        """
        val = 0.0
        for i in xrange(len(self.functions)):
            fi = self.functions[i]
            for j in xrange(len(fi)):
                fij = fi[j]
                if i == j and isinstance(fij, (list, tuple)):
                    for k in xrange(len(fij)):
#                        print "Diag: ", i
                        val += fij[k].f(w[i])
                else:
#                    print "f(w[%d], w[%d])" % (i, j)
                    if not fij is None:
                        val += fij.f([w[i], w[j]])

        # TODO: Check instead if it is a numpy array.
        if not isinstance(val, numbers.Number):
            return val[0, 0]
        else:
            return val

    def grad(self, w, index):
        """Gradient of the differentiable part of the function.

        From the interface "MultiblockGradient".
        """
        grad = 0.0
        fi = self.functions[index]
        for j in xrange(len(fi)):
            fij = fi[j]
            if index != j:
                if isinstance(fij, interfaces.Gradient):
                    grad += fij.grad(w[index])
                elif isinstance(fij, mb_interfaces.MultiblockGradient):
                    grad += fij.grad([w[index], w[j]], 0)

        for i in xrange(len(self.functions)):
            fij = self.functions[i][index]
            if i != index:
                if isinstance(fij, interfaces.Gradient):
                    # We shouldn't do anything here, right? This means e.g.
                    # that this (block i) is the y of a logistic regression.
                    pass
#                    grad += fij.grad(w)
                elif isinstance(fij, mb_interfaces.MultiblockGradient):
                    grad += fij.grad([w[i], w[index]], 1)

        fii = self.functions[index][index]
        for k in xrange(len(fii)):
            if isinstance(fii[k], interfaces.Gradient):
                grad += fii[k].grad(w[index])

        return grad

#    def prox(self, w, index, factor=1.0):
#        """The proximal operator corresponding to the function with the index.
#
#        From the interface "MultiblockProximalOperator".
#        """
##        # Find a proximal operator.
##        fii = self.functions[index][index]
##        for k in xrange(len(fii)):
##            if isinstance(fii[k], ProximalOperator):
##                w[index] = fii[k].prox(w[index], factor)
##                break
##        # If no proximal operator was found, we will just return the same
##        # vectors again. The proximal operator of the zero function returns the
##        # vector itself.
#
#        return w

    def proj(self, w, index):
        """The projection operator corresponding to the function with the
        index.

        From the interface "MultiblockProjectionOperator".
        """
        # Find a projection operators.
#        fii = self.functions[index][index]
        f = self.get_constraints(index)
        for k in xrange(len(f)):
            if isinstance(f[k], interfaces.ProjectionOperator):
                w[index] = f[k].proj(w[index])
                break

        # If no projection operator was found, we will just return the same
        # vectors again.
        return w

    def step(self, w, index):

#        return 0.0001

        all_lipschitz = True

        # Add the Lipschitz constants.
        L = 0.0
        fi = self.functions[index]
        for j in xrange(len(fi)):
            if j != index and fi[j] is not None:
                fij = fi[j]
                if isinstance(fij, interfaces.LipschitzContinuousGradient):
                    L += fij.L()
                elif isinstance(fij,
                        mb_interfaces.MultiblockLipschitzContinuousGradient):
                    L += fij.L(w, index)
                else:
                    all_lipschitz = False
                    break

        if all_lipschitz:
            fii = self.functions[index][index]
            for k in xrange(len(fii)):
                if fi[j] is None:
                    continue
                if isinstance(fii[k], interfaces.LipschitzContinuousGradient):
                    L += fii[k].L()
                elif isinstance(fii[k],
                        mb_interfaces.MultiblockLipschitzContinuousGradient):
                    L += fii[k].L(w, index)
                else:
                    all_lipschitz = False
                    break

        if all_lipschitz and L > 0.0:
            t = 1.0 / L
        else:
            # If all functions did not have Lipschitz continuous gradients,
            # try to find the step size through backtracking line search.
            class F(interfaces.Function,
                    interfaces.Gradient):
                def __init__(self, func, w, index):
                    self.func = func
                    self.w = w
                    self.index = index

                def f(self, x):

                    # Temporarily replace the index:th variable with x.
                    w_old = self.w[self.index]
                    self.w[self.index] = x
                    f = self.func.f(w)
                    self.w[self.index] = w_old

                    return f

                def grad(self, x):

                    # Temporarily replace the index:th variable with x.
                    w_old = self.w[self.index]
                    self.w[self.index] = x
                    g = self.func.grad(w, index)
                    self.w[self.index] = w_old

                    return g

            func = F(self, w, index)
            p = -self.grad(w, index)

            from algorithms import BacktrackingLineSearch
            import parsimony.functions.penalties as penalties
            line_search = BacktrackingLineSearch(
                condition=penalties.SufficientDescentCondition, max_iter=30)
            a = np.sqrt(1.0 / self.X[index].shape[1])  # Arbitrarily "small".
            t = line_search(func, w[index], p, rho=0.5, a=a, c=1e-4)

        return t