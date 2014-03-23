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
import parsimony.utils as utils
import parsimony.utils.maths as maths
import parsimony.utils.consts as consts
import parsimony.functions.interfaces as interfaces
import parsimony.functions.multiblock.interfaces as multiblock_interfaces
import parsimony.functions.multiblock.losses as mb_losses

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
                  multiblock_interfaces.MultiblockStepSize]

    def __init__(self, step=None, output=False,
                 eps=consts.TOLERANCE,
                 outer_iter=25, max_iter=consts.MAX_ITER, min_iter=1):

        self.step = step
        self.output = output
        self.eps = eps
        self.outer_iter = outer_iter
        self.max_iter = max_iter
        self.min_iter = min_iter

        import parsimony.algorithms.explicit as algorithms
        self.fista = algorithms.FISTA(step=self.step,
                                      output=self.output,
                                      eps=self.eps,
                                      max_iter=self.max_iter,
                                      min_iter=self.min_iter)

    def run(self, function, w):

        self.check_compatibility(function, self.INTERFACES)

        print "outer_iter:", self.outer_iter
        print "len(w):", len(w)
        print "max_iter:", self.max_iter

#        z = w_old = w

        if self.output:
            f = [function.f(w)]
            print "f:", f[-1]

        t = [1.0] * len(w)

        w_old = [0] * len(w)

        for it in xrange(self.outer_iter):  # TODO: Get number of iterations!
            all_converged = True
            for i in xrange(len(w)):
                converged = False
                print "it: %d, i: %d" % (it, i)

#                func = mb_losses.MultiblockFunctionWrapper(function, w, i)

#                if self.output:
#                    w[i], output = self.fista.run(func, w[i])
#
#                    f = f + output["f"]
#                else:
#                    w[i] = self.fista.run(func, w[i])

                for j in xrange(len(w)):
                    w_old[j] = w[j]

                for k in xrange(self.max_iter):
#                    print "it: %d, i: %d, k: %d" % (it, i, k)

#                    if i == 1:
#                        print "w00t!?"

                    z = w[i] + ((k - 2.0) / (k + 1.0)) * (w[i] - w_old[i])

#                    _t = utils.time_cpu()
                    t[i] = function.step(w[:i] + [z] + w[i + 1:], i)
#                    t[i] = func.step(z)
#                    print "step:", utils.time_cpu() - _t

                    w_old[i] = w[i]

#                    _t = utils.time_cpu()
                    grad = function.grad(w_old[:i] + [z] + w_old[i + 1:], i)
#                    grad = func.grad(z)
                    w[i] = z - t[i] * grad
#                    print "grad:", utils.time_cpu() - _t

#                    _t = utils.time_cpu()
                    w[i] = function.prox(w, i, t[i])
#                    w[i] = func.prox(w[i], t[i])
#                    print "proj:", utils.time_cpu() - _t

                    if self.output:
#                        _t = utils.time_cpu()
                        f_ = function.f(w)
#                        print "   f:", utils.time_cpu() - _t
#                        print "f:", f_
#                        improvement = f_ - f[-1]
#                        if improvement > 0.0:
#                            print "INCREASE!!:", improvement
#                            # If this happens there are two possible reasons:
#                            if abs(improvement) <= consts.TOLERANCE:
#                                # 1. The function is actually converged, and
#                                #         the "increase" is because of
#                                #         precision errors. This happens
#                                #         sometimes.
#                                pass
#                            else:
#                                # 2. There is an error and the function
#                                #         actually increased. Does this
#                                #         happen? If so, we need to
#                                #         investigate! Possible errors are:
#                                #          * The gradient is wrong.
#                                #          * The step size is too large.
#                                #          * Other reasons?
#                                print "ERROR! Function increased!"
#
#                            # Either way, we stop and regroup if it happens.
##                            break

                        f.append(f_)

                    err = maths.norm(z - w[i])
                    if err < t[i] * self.eps and k + 1 >= self.min_iter:
                        converged = True
                        break

                print "l0 :", maths.norm0(w[i]), \
                    ", l1 :", maths.norm1(w[i]), \
                    ", l2²:", maths.norm(w[i]) ** 2.0
                if self.output:
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