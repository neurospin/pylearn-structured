# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.algorithms` module includes several algorithms used
throughout the package.

Algorithms may not store states. I.e., if they are classes, do not keep
references to objects with state in the algorithm objects. It should be
possible to copy and share algorithms between estimators, and thus they should
not depend on any state.

There are currently two types of algorithms: implicit and explicit. The
difference is whether they run directly on the data (implicit) or if they have
an actual loss function than is minimised (explicit). Implicit algorithms take
the data as input, and then run on the data. Explicit algorithms take a loss
function and a start vector as input, and then minimise the function value
starting from the point of the start vector.

Created on Thu Feb 20 22:12:00 2014

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import copy

from . import bases
import parsimony.utils.maths as maths
import parsimony.utils.consts as consts
import parsimony.functions.interfaces as interfaces
import parsimony.functions.multiblock.interfaces as multiblock_interfaces

__all__ = ["MultiblockProjectedGradientMethod"]


#class GeneralisedMultiblockISTA(ExplicitAlgorithm):
#    """ The iterative shrinkage threshold algorithm in a multiblock setting.
#    """
#    INTERFACES = [functions.MultiblockFunction,
#                  functions.MultiblockGradient,
#                  functions.MultiblockProximalOperator,
#                  functions.StepSize,
#                 ]
#
#    def __init__(self, step=None, output=False,
#                 eps=consts.TOLERANCE,
#                 max_iter=consts.MAX_ITER, min_iter=1):
#
#        self.step = step
#        self.output = output
#        self.eps = eps
#        self.max_iter = max_iter
#        self.min_iter = min_iter
#
#    def __call__(self, function, w):
#
#        self.check_compatability(function, self.INTERFACES)
#
#        for it in xrange(10):  # TODO: Get number of iterations!
#            print "it:", it
#
#            for i in xrange(len(w)):
#                print "  i:", i
#
#                for k in xrange(10000):
#                    print "    k:", k
#
#                    t = function.step(w, i)
#                    w[i] = w[i] - t * function.grad(w, i)
#                    w = function.prox(w, i, t)
##                    = w[:i] + [wi] + w[i+1:]
#
#                    print "    f:", function.f(w)
#
##                w[i] = wi
#
#        return w


class MultiblockProjectedGradientMethod(bases.ExplicitAlgorithm):
    """ The projected gradient algorithm with alternating minimisations in a
    multiblock setting.
    """
    INTERFACES = [multiblock_interfaces.MultiblockFunction,
                  multiblock_interfaces.MultiblockGradient,
                  multiblock_interfaces.MultiblockProjectionOperator,
                  interfaces.StepSize]

    def __init__(self, step=None, output=False,
                 eps=consts.TOLERANCE,
                 outer_iter=25, max_iter=consts.MAX_ITER, min_iter=1):

        self.step = step
        self.output = output
        self.eps = eps
        self.outer_iter = outer_iter
        self.max_iter = max_iter
        self.min_iter = min_iter

    def run(self, function, w):

        self.check_compatibility(function, self.INTERFACES)

        print "outer_iter:", self.outer_iter
        print "len(w):", len(w)
        print "max_iter:", self.max_iter

#        z = w_old = w

        if self.output:
            f = [function.f(w)]

        t = [1.0] * len(w)

        for it in xrange(self.outer_iter):  # TODO: Get number of iterations!
            all_converged = True
            for i in xrange(len(w)):
                converged = False
                print "it: %d, i: %d" % (it, i)
                for k in xrange(self.max_iter):
#                    print "it: %d, i: %d, k: %d" % (it, i, k)

#                    z = w[i] + ((k - 2.0) / (k + 1.0)) * (w[i] - w_old[i])

                    w_old = copy.deepcopy(w)

#                    _t = time()
                    t[i] = function.step(w_old, i)
#                    print "t:", t[i]
#                    print "step:", time() - _t

#                    _t = time()
                    grad = function.grad(w_old, i)
                    w[i] = w_old[i] - t[i] * grad

#                    def fun(x):
#                        w_ = [0, 0]
#                        w_[i] = x
#                        w_[1 - i] = w[1 - i]
#                        return function.f(w_)
#                    approx_grad = utils.approx_grad(fun, w[i], eps=1e-6)
#                    diff = float(maths.norm(grad - approx_grad))
#                    print "grad err: %e, lim: %e" % (diff, 5e-5)
#                    if diff > 5e-4:
#                        pass

#                    w[i] = z[i] - t[i] * function.grad(w_old[:i] +
#                                                       [z] +
#                                                       w_old[i + 1:], i)
#                    print "grad:", time() - _t

#                    _t = time()
                    w = function.proj(w, i)
#                    print "proj:", time() - _t

#                    print "l0 :", maths.norm0(w[i]), \
#                        ", l1 :", maths.norm1(w[i]), \
#                        ", l2²:", maths.norm(w[i]) ** 2.0

                    if self.output:
                        f_ = function.f(w)
#                        print "f:", f_
                        improvement = f_ - f[-1]
                        if improvement > 0.0:
                            # If this happens there are two possible reasons:
                            if abs(improvement) <= consts.TOLERANCE:
                                # 1. The function is actually converged, and
                                #         the "increase" is because of
                                #         precision errors. This happens
                                #         sometimes.
                                pass
                            else:
                                # 2. There is an error and the function
                                #         actually increased. Does this
                                #         happen? If so, we need to
                                #         investigate! Possible errors are:
                                #          * The gradient is wrong.
                                #          * The step size is too large.
                                #          * Other reasons?
                                print "ERROR! Function increased!"

                            # Either way, we stop and regroup if it happens.
                            break

                        f.append(f_)

                    err = maths.norm(w_old[i] - w[i])
#                    print "err: %.10f < %.10f * %.10f = %.10f" \
#                        % (err, t[i], self.eps, t[i] * self.eps)
                    if err <= t[i] * self.eps and k + 1 >= self.min_iter:
                        converged = True
                        break

                print "l0 :", maths.norm0(w[i]), \
                    ", l1 :", maths.norm1(w[i]), \
                    ", l2²:", maths.norm(w[i]) ** 2.
                print "f:", f[-1]

                if not converged:
                    all_converged = False

            if all_converged:
                print "All converged!"
                break

        if self.output:
#            output = {"t": t, "f": f}
            output = {"f": f}
            return (w, output)
        else:
            return w