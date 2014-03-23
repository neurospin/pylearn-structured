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
import parsimony.utils.consts as consts
import interfaces as mb_interfaces

__all__ = ["CombinedMultiblockFunction", "LatentVariableCovariance"]


class CombinedMultiblockFunction(mb_interfaces.MultiblockFunction,
                                 mb_interfaces.MultiblockGradient,
                                 mb_interfaces.MultiblockProximalOperator,
                                 mb_interfaces.MultiblockProjectionOperator,
                                 mb_interfaces.MultiblockStepSize):
    """Combines one or more loss functions, any number of penalties and zero
    or one proximal operator.

    This function thus represents

        f(x) = f_1(x) [ + f_2(x) ... ] [ + p_1(x) ... ] [ + P_1(x) ...],

    subject to [ C_1(x) <= c_1,
                 C_2(x) <= c_2,
                 ... ],

    where f_i are differentiable Functions that may be multiblock, p_j are
    differentiable penalties and P_k are a ProximalOperators. All functions
    and penalties must thus be Gradient, unless they are ProximalOperators.

    If no ProximalOperator is given, then prox is the identity.
    """
    def __init__(self, X, functions=[], penalties=[], prox=[], constraints=[]):
        """
        Parameters
        ----------
        X : List of numpy arrays. The blocks of data in the multiblock model.

        functions : List of lists of lists. A function matrix, with elements
                i,j connecting block i to block j.

        penalties : A list of lists. Element i of the outer list contains the
                penalties for block i.

        prox : A list of lists. Element i of the outer list contains the
                penalties that can be expressed as proximal operators for
                block i.

        constraints : A list of lists. Element i of the outer list contains
                the constraints for block i.
        """
        self.K = len(X)
        self.X = X

        if len(functions) != self.K:
            self._f = [0] * self.K
            for i in xrange(self.K):
                self._f[i] = [0] * self.K
                for j in xrange(self.K):
                    self._f[i][j] = list()

        if len(penalties) != self.K:
            self._p = [0] * self.K
            for i in xrange(self.K):
                self._p[i] = list()

        if len(prox) != self.K:
            self._prox = [0] * self.K
            for i in xrange(self.K):
                self._prox[i] = list()

        if len(constraints) != self.K:
            self._c = [0] * self.K
            for i in xrange(self.K):
                self._c[i] = list()

    def reset(self):

        for fi in self._f:
            for fij in fi:
                for fijk in fij:
                    fijk.reset()

        for pi in self._p:
            for pik in pi:
                pik.reset()

        for proxi in self._prox:
            for proxik in proxi:
                proxik.reset()

        for ci in self._c:
            for cik in ci:
                cik.reset()

    def add_function(self, function, i, j):
        """Add a function between blocks i and j.

        Parameters
        ----------
        function : Function or MultiblockFunction. A function that connects
                block i and block j.

        i : Non-negative integer. Index of the first block. Zero based, so 0
                is the first block.

        j : Non-negative integer. Index of the second block. Zero based, so 0
                is the first block.
        """
        if not isinstance(function, interfaces.Gradient):
            if not isinstance(function, mb_interfaces.MultiblockGradient):
                raise ValueError("Functions must have gradients.")

        self._f[i][j].append(function)

    def add_penalty(self, penalty, i):

        if not isinstance(penalty, interfaces.Penalty):
            raise ValueError("Not a penalty.")
        if not isinstance(penalty, interfaces.Gradient):
            raise ValueError("Penalties must have gradients.")

        self._p[i].append(penalty)

    def add_prox(self, penalty, i):

        if not isinstance(penalty, interfaces.ProximalOperator):
            raise ValueError("Not a proximal operator.")

        self._prox[i].append(penalty)

    def add_constraint(self, constraint, i):

        if not isinstance(constraint, interfaces.Constraint):
            raise ValueError("Not a constraint.")
        if not isinstance(constraint, interfaces.ProjectionOperator):
            raise ValueError("Constraints must have projection operators.")

        self._c[i].append(constraint)

    def f(self, w):
        """Function value.

        Parameters
        ----------
        w : List of numpy arrays. The weight vectors.
        """
        val = 0.0

        for i in xrange(len(self._f)):
            fi = self._f[i]
            for j in xrange(len(fi)):
                fij = self._f[i][j]
                for k in xrange(len(fij)):
                    val += fij[k].f([w[i], w[j]])

        for i in xrange(len(self._p)):
            pi = self._p[i]
            for k in xrange(len(pi)):
                val += pi[k].f(w[i])

        for i in xrange(len(self._prox)):
            proxi = self._prox[i]
            for k in xrange(len(proxi)):
                val += proxi[k].f(w[i])

        return val

    def grad(self, w, index):
        """Gradient of the differentiable part of the function.

        From the interface "MultiblockGradient".

        Parameters
        ----------
        w : List of numpy arrays. The weight vectors, w[index] is the point at
                which to evaluate the gradient.

        index : Non-negative integer. Which variable the step is for.
        """
        grad = 0.0

        # Add gradients from the loss functions.
        fi = self._f[index]
        for j in xrange(len(fi)):
            fij = fi[j]
            for k in xrange(len(fij)):
                fijk = fij[k]
                if isinstance(fijk, interfaces.Gradient):
                    grad += fijk.grad(w[index])
                elif isinstance(fijk, mb_interfaces.MultiblockGradient):
                    grad += fijk.grad([w[index], w[j]], 0)

        for i in xrange(len(self._f)):
            fij = self._f[i][index]
            if i != index:  # Do not count these twice.
                if isinstance(fij, interfaces.Gradient):
                    # We shouldn't do anything here, right? This means e.g.
                    # that this (block i) is the y of a logistic regression.
                    pass
#                    grad += fij.grad(w[i])
                elif isinstance(fij, mb_interfaces.MultiblockGradient):
                    grad += fij.grad([w[i], w[index]], 1)

        # Add gradients from the penalties.
        pi = self._p[index]
        for k in xrange(len(pi)):
            grad += pi[k].grad(w[index])

        return grad

    def prox(self, w, index, factor=1.0):
        """The proximal operator of the non-differentiable part of the
        function with the given index.

        From the interface "MultiblockProximalOperator".

        Parameters
        ----------
        w : List of numpy arrays. The weight vectors.

        index : Non-negative integer. Which variable the step is for.

        factor : Positive float. A factor by which the Lagrange multiplier is
                scaled. This is usually the step size.
        """
        prox = self._prox[index]
        proj = self._c[index]

        if len(prox) == 1 and len(proj) == 0:
            prox_w = prox[0].prox(w[index])

        elif len(prox) == 0 and (len(proj) == 1 or len(proj) == 2):
            prox_w = self.proj(w, index)

        else:
            from parsimony.algorithms.explicit \
                    import ParallelDykstrasProximalAlgorithm
            combo = ParallelDykstrasProximalAlgorithm(output=False,
                                                      eps=consts.TOLERANCE,
                                                      max_iter=consts.MAX_ITER,
                                                      min_iter=1)
            prox_w = combo.run(w[index], prox=prox, proj=proj, factor=factor)

        return prox_w

    def proj(self, w, index):
        """The projection operator of a constraint that corresponds to the
        function with the given index.

        From the interface "MultiblockProjectionOperator".

        Parameters
        ----------
        w : List of numpy arrays. The weight vectors.

        index : Non-negative integer. Which variable the step is for.
        """
        prox = self._prox[index]
        proj = self._c[index]
        if len(proj) == 1 and len(prox) == 0:
            proj_w = proj[0].proj(w[index])

        elif len(proj) == 2 and len(prox) == 0:
            constraint = interfaces.CombinedProjectionOperator(proj)
            proj_w = constraint.proj(w[index])

        else:
            proj_w = self.prox(w, index)

        return proj_w

    def step(self, w, index):
        """The step size to use in descent methods.

        From the interface "StepSize".

        Parameters
        ----------
        w : Numpy array. The point at which to determine the step size.

        index : Non-negative integer. The variable which the step is for.
        """
        all_lipschitz = True
        L = 0.0

        # Add gradients from the loss functions.
        fi = self._f[index]
        for j in xrange(len(fi)):
            fij = fi[j]
            for k in xrange(len(fij)):
                fijk = fij[k]
                if isinstance(fijk, interfaces.Gradient):
                    if not isinstance(fijk,
                                      interfaces.LipschitzContinuousGradient):
                        all_lipschitz = False
                        break
                    else:
                        L += fijk.L(w[index])
                elif isinstance(fijk, mb_interfaces.MultiblockGradient):
                    if not isinstance(fijk,
                                      mb_interfaces.MultiblockLipschitzContinuousGradient):
                        all_lipschitz = False
                        break
                    else:
                        L += fijk.L(w, index)

            if not all_lipschitz:
                break

        for i in xrange(len(self._f)):
            fij = self._f[i][index]
            if i != index:  # Do not visit these twice.
                for k in xrange(len(fij)):
                    fijk = fij[k]
                    if isinstance(fijk, interfaces.Gradient):
                        # We shouldn't do anything here, right? This means e.g.
                        # that this (block i) is the y of a logistic
                        # regression.
                        pass
                    elif isinstance(fijk, mb_interfaces.MultiblockGradient):
                        if not isinstance(fijk,
                                          mb_interfaces.MultiblockLipschitzContinuousGradient):
                            all_lipschitz = False
                            break
                        else:
                            L += fijk.L(w, index)

        # Add gradients from the penalties.
        pi = self._p[index]
        for k in xrange(len(pi)):
            if not isinstance(pi[k], interfaces.LipschitzContinuousGradient):
                all_lipschitz = False
                break
            else:
                L += pi[k].L()  # w[index])

        step = 0.0
        if all_lipschitz and L > 0.0:
            step = 1.0 / L
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

            from parsimony.algorithms.explicit import BacktrackingLineSearch
            import parsimony.functions.penalties as penalties
            line_search = BacktrackingLineSearch(
                condition=penalties.SufficientDescentCondition, max_iter=30)
            a = np.sqrt(1.0 / self.X[index].shape[1])  # Arbitrarily "small".
            step = line_search.run(func, w[index], p, rho=0.5, a=a, c=1e-4)

        return step


class MultiblockFunctionWrapper(interfaces.CompositeFunction,
                                interfaces.Gradient,
                                interfaces.StepSize,
                                interfaces.ProximalOperator):

    def __init__(self, function, w, index):
        self.function = function
        self.w = w
        self.index = index

    def f(self, w):
        """Function value.

        From the interface "Function".

        Parameters
        ----------
        w : Numpy array (p-by-1). The parameter at which to evaluate the
                function.
        """
        return self.function.f(self.w[:self.index] + \
                               [w] + \
                               self.w[self.index + 1:])

    def grad(self, w):
        """Gradient of the function.

        Parameters
        ----------
        w : Numpy array (p-by-1). The point at which to evaluate the gradient.
        """
        return self.function.grad(self.w[:self.index] + \
                                  [w] + \
                                  self.w[self.index + 1:],
                                  self.index)

    def prox(self, w, factor=1.0):
        """The proximal operator corresponding to the function.

        Parameters
        ----------
        w : Numpy array (p-by-1). The point at which to apply the proximal
                operator.

        factor : Positive float. A factor by which the Lagrange multiplier is
                scaled. This is usually the step size.
        """
        return self.function.prox(self.w[:self.index] + \
                                  [w] + \
                                  self.w[self.index + 1:],
                                  self.index, factor)

    def step(self, w, index=0):
        """The step size to use in descent methods.

        Parameters
        ----------
        w : Numpy array. The point at which to determine the step size.
        """
        return self.function.step(self.w[:self.index] + \
                                  [w] + \
                                  self.w[self.index + 1:],
                                  self.index)


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