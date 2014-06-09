# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.algorithms.gradient` module includes several algorithms
that minimises an explicit loss function while utilising the gradient of the
function.

Algorithms may not store states. I.e., if they are classes, do not keep
references to objects with state in the algorithm objects. It should be
possible to copy and share algorithms between e.g. estimators, and thus they
should not depend on any state.

Created on Wed Jun  4 15:22:50 2014

Copyright (c) 2013-2014, CEA/DSV/I2BM/Neurospin. All rights reserved.

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
try:
    from . import bases  # Only works when imported as a package.
except ValueError:
    import parsimony.algorithms.bases as bases  # When run as a program.
from parsimony.utils import Info
import parsimony.utils as utils
import parsimony.utils.maths as maths
import parsimony.utils.consts as consts
import parsimony.functions.properties as properties

__all__ = ["GradientDescent"]


class GradientDescent(bases.ExplicitAlgorithm,
                      bases.IterativeAlgorithm,
                      bases.InformationAlgorithm):
    """The gradient descent algorithm.

    Parameters
    ----------
    eps : Positive float. Tolerance for the stopping criterion.

    info : Information. If, and if so what, extra run information should be
            returned. Default is None, which is replaced by Information(),
            which means that no run information is computed nor returned.

    max_iter : Positive integer. Maximum allowed number of iterations.

    min_iter : Positive integer. Minimum number of iterations.

    Examples
    --------
    >>> from parsimony.algorithms.gradient import GradientDescent
    >>> from parsimony.functions.losses import RidgeRegression
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> X = np.random.rand(100, 50)
    >>> y = np.random.rand(100, 1)
    >>> gd = GradientDescent(max_iter=10000)
    >>> function = RidgeRegression(X, y, k=0.0, mean=False)
    >>> beta1 = gd.run(function, np.random.rand(50, 1))
    >>> beta2 = np.dot(np.linalg.pinv(X), y)
    >>> np.linalg.norm(beta1 - beta2)
    0.0003121557632556645
    """
    INTERFACES = [properties.Function,
                  properties.Gradient,
                  properties.StepSize]

    PROVIDED_INFO = [Info.ok,
                     Info.num_iter,
                     Info.t,
                     Info.f,
                     Info.converged]

    def __init__(self, eps=consts.TOLERANCE,
                 info=None, max_iter=20000, min_iter=1):
        super(GradientDescent, self).__init__(info=info,
                                              max_iter=max_iter,
                                              min_iter=min_iter)

        self.eps = eps

    @bases.check_compatibility
    def run(self, function, beta):
        """Find the minimiser of the given function, starting at beta.

        Parameters
        ----------
        function : Function. The function to minimise.

        beta : Numpy array. The start vector.
        """
        if self.info.allows(Info.ok):
            self.info[Info.ok] = False

        step = function.step(beta)

        betanew = betaold = beta

        if self.info.allows(Info.t):
            t = []
        if self.info.allows(Info.f):
            f = []
        if self.info.allows(Info.converged):
            self.info[Info.converged] = False

        for i in xrange(1, self.max_iter + 1):
            if self.info.allows(Info.t):
                tm = utils.time_cpu()

            step = function.step(betanew)

            betaold = betanew
            betanew = betaold - step * function.grad(betaold)

            if self.info.allows(Info.t):
                t.append(utils.time_cpu() - tm)
            if self.info.allows(Info.f):
                f.append(function.f(betanew))

            if maths.norm(betanew - betaold) < self.eps \
                    and i >= self.min_iter:

                if self.info.allows(Info.converged):
                    self.info[Info.converged] = True

                break

        if self.info.allows(Info.num_iter):
            self.info[Info.num_iter] = i
        if self.info.allows(Info.t):
            self.info[Info.t] = t
        if self.info.allows(Info.f):
            self.info[Info.f] = f
        if self.info.allows(Info.ok):
            self.info[Info.ok] = True

        return betanew