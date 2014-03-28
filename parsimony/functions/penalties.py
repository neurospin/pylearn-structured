# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.functions.penalties` module contains the penalties used to
constrain the loss functions. These represent mathematical functions and
should thus have properties used by the corresponding algorithms. These
properties are defined in :mod:`parsimony.functions.interfaces`.

Penalties should be stateless. Penalties may be shared and copied and should
therefore not hold anything that cannot be recomputed the next time it is
called.

Created on Mon Apr 22 10:54:29 2013

@author:  Tommy Löfstedt, Vincent Guillemot, Edouard Duchesnay and
          Fouad Hadj-Selem
@email:   lofstedt.tommy@gmail.com, edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
import numpy as np

try:
    from . import interfaces  # Only works when imported as a package.
except ValueError:
    import parsimony.functions.interfaces as interfaces  # Run as a script.
import parsimony.utils.maths as maths
import parsimony.utils.consts as consts

__all__ = ["ZeroFunction", "L0", "L1", "L2", "LInf",
           "QuadraticConstraint", "RGCCAConstraint",
           "SufficientDescentCondition"]


class ZeroFunction(interfaces.AtomicFunction,
                   interfaces.Gradient,
                   interfaces.Penalty,
                   interfaces.Constraint,
                   interfaces.ProximalOperator,
                   interfaces.ProjectionOperator):

    def __init__(self, l=1.0, c=0.0, penalty_start=0):
        """
        Parameters
        ----------
        l : Non-negative float. The Lagrange multiplier, or regularisation
                constant, of the function.

        c : Float. The limit of the constraint. The function is feasible if
                ||\beta||_1 <= c. The default value is c=0, i.e. the default is
                a regularisation formulation.

        penalty_start : Non-negative integer. The number of columns, variables
                etc., to be exempt from penalisation. Equivalently, the first
                index to be penalised. Default is 0, all columns are included.
        """
        self.l = float(l)
        self.c = float(c)
        if self.c < 0.0:
            raise ValueError("A negative constraint parameter does not make " \
                             "sense, since the function is always zero.")
        self.penalty_start = int(penalty_start)

        self.reset()

    def reset(self):

        self._zero = None

    def f(self, x):
        """Function value.
        """
        return 0.0

    def grad(self, x):
        """Gradient of the function.

        From the interface "Gradient".
        """
        if self._zero is None:
            self._zero = np.zeros(x.shape)

        return self._zero

    def prox(self, x, factor=1.0):
        """The corresponding proximal operator.

        From the interface "ProximalOperator".
        """
        return x

    def proj(self, x):
        """The corresponding projection operator.

        From the interface "ProjectionOperator".
        """
        return x

    def feasible(self, x):
        """Feasibility of the constraint.

        From the interface "Constraint".
        """
        return self.c >= 0.0


class L1(interfaces.AtomicFunction,
         interfaces.Penalty,
         interfaces.Constraint,
         interfaces.ProximalOperator,
         interfaces.ProjectionOperator):
    """The proximal operator of the L1 function with a penalty formulation

        f(\beta) = l * (||\beta||_1 - c),

    where ||\beta||_1 is the L1 loss function. The constrained version has the
    form

        ||\beta||_1 <= c.

    Parameters
    ----------
    l : Non-negative float. The Lagrange multiplier, or regularisation
            constant, of the function.

    c : Float. The limit of the constraint. The function is feasible if
            ||\beta||_1 <= c. The default value is c=0, i.e. the default is a
            regularisation formulation.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first index
            to be penalised. Default is 0, all columns are included.
    """
    def __init__(self, l=1.0, c=0.0, penalty_start=0):

        self.l = float(l)
        self.c = float(c)
        self.penalty_start = int(penalty_start)

    def f(self, beta):
        """Function value.
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta
        return self.l * (maths.norm1(beta_) - self.c)

    def prox(self, beta, factor=1.0):
        """The corresponding proximal operator.

        From the interface "ProximalOperator".
        """
        l = self.l * factor
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        prox = (np.abs(beta_) > l) * (beta_ - l * np.sign(beta_ - l))

        if self.penalty_start > 0:
            prox = np.vstack((beta[:self.penalty_start, :], prox))

        return prox

    def proj(self, beta):
        """The corresponding projection operator.

        From the interface "ProjectionOperator".
        """
        if self.feasible(beta):
            return beta

        from parsimony.algorithms.explicit import Bisection
        bisection = Bisection(force_negative=True,
                              parameter_positive=True,
                              parameter_negative=False,
                              parameter_zero=False,
                              eps=1e-8)

        class F(interfaces.Function):
            def __init__(self, beta, c):
                self.beta = beta
                self.c = c

            def f(self, l):
                beta = (np.abs(self.beta) > l) \
                    * (self.beta - l * np.sign(self.beta - l))

                return maths.norm1(beta) - self.c

        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        func = F(beta_, self.c)
        l = bisection.run(func, [0.0, np.max(np.abs(beta_))])

        return (np.abs(beta_) > l) * (beta_ - l * np.sign(beta_ - l))

    def feasible(self, beta):
        """Feasibility of the constraint.

        From the interface "Constraint".

        Parameters
        ----------
        beta : Numpy array. The variable to check for feasibility.
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        return maths.norm1(beta_) <= self.c


class L0(interfaces.AtomicFunction,
         interfaces.Penalty,
         interfaces.Constraint,
         interfaces.ProximalOperator,
         interfaces.ProjectionOperator):
    """The proximal operator of the "pseudo" L0 function

        f(x) = l * (||x||_0 - c),

    where ||x||_0 is the L0 loss function. The constrainted version has the
    form

        ||x||_0 <= c.

    Warning: Note that this function is not convex, and the regular assumptions
    when using it in e.g. ISTA or FISTA will not apply. Nevertheless, it will
    still converge to a local minimum if we can guarantee that we obtain a
    reduction of the smooth part in each step. See e.g.:

        http://eprints.soton.ac.uk/142499/1/BD_NIHT09.pdf
        http://people.ee.duke.edu/~lcarin/blumensath.pdf

    Parameters
    ----------
    l : Non-negative float. The Lagrange multiplier, or regularisation
            constant, of the function.

    c : Float. The limit of the constraint. The function is feasible if
            ||x||_0 <= c. The default value is c=0, i.e. the default is a
            regularisation formulation.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first index
            to be penalised. Default is 0, all columns are included.
    """
    def __init__(self, l=1.0, c=0.0, penalty_start=0):

        self.l = float(l)
        self.c = float(c)
        self.penalty_start = int(penalty_start)

    def f(self, x):
        """Function value.

        From the interface "Function".

        Example
        -------
        >>> import numpy as np
        >>> from parsimony.functions.penalties import L0
        >>> import parsimony.utils.maths as maths
        >>>
        >>> np.random.seed(42)
        >>> x = np.random.rand(10, 1)
        >>> l0 = L0(l=0.5)
        >>> maths.norm0(x)
        10
        >>> l0.f(x) - 0.5 * maths.norm0(x)
        0.0
        >>> x[0, 0] = 0.0
        >>> maths.norm0(x)
        9
        >>> l0.f(x) - 0.5 * maths.norm0(x)
        0.0
        """
        if self.penalty_start > 0:
            x_ = x[self.penalty_start:, :]
        else:
            x_ = x

        return self.l * (maths.norm0(x_) - self.c)

    def prox(self, x, factor=1.0):
        """The corresponding proximal operator.

        From the interface "ProximalOperator".

        Example
        -------
        >>> import numpy as np
        >>> from parsimony.functions.penalties import L0
        >>> import parsimony.utils.maths as maths
        >>>
        >>> np.random.seed(42)
        >>> x = np.random.rand(10, 1)
        >>> l0 = L0(l=0.5)
        >>> maths.norm0(x)
        10
        >>> l0.prox(x)
        array([[ 0.        ],
               [ 0.95071431],
               [ 0.73199394],
               [ 0.59865848],
               [ 0.        ],
               [ 0.        ],
               [ 0.        ],
               [ 0.86617615],
               [ 0.60111501],
               [ 0.70807258]])
        >>> l0.f(l0.prox(x))
        3.0
        >>> 0.5 * maths.norm0(l0.prox(x))
        3.0
        """
        if self.penalty_start > 0:
            x_ = x[self.penalty_start:, :]
        else:
            x_ = x

        l = self.l * factor
        prox = x_ * (np.abs(x_) > l)  # Hard thresholding.
        prox = np.vstack((x[:self.penalty_start, :],  # Unregularised variables
                          prox))

        return prox

    def proj(self, x):
        """The corresponding projection operator.

        From the interface "ProjectionOperator".

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.functions.penalties import L0
        >>>
        >>> np.random.seed(42)
        >>> x = np.random.rand(10, 1) * 2.0 - 1.0
        >>> l0 = L0(c=5.0)
        >>> l0.proj(x)
        array([[ 0.        ],
               [ 0.90142861],
               [ 0.        ],
               [ 0.        ],
               [-0.68796272],
               [-0.68801096],
               [-0.88383278],
               [ 0.73235229],
               [ 0.        ],
               [ 0.        ]])
        """
        if self.penalty_start > 0:
            x_ = x[self.penalty_start:, :]
        else:
            x_ = x

        if maths.norm0(x_) <= self.c:
            return x

        K = int(np.floor(self.c) + 0.5)
        ind = np.abs(x_.ravel()).argsort()[:K]
        y = np.copy(x_)
        y[ind] = 0.0

        y = np.vstack((x[:self.penalty_start, :],  # Unregularised variables.
                       y))

        return y

    def feasible(self, beta):
        """Feasibility of the constraint.

        From the interface "Constraint".

        Parameters
        ----------
        beta : Numpy array. The variable to check for feasibility.

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.functions.penalties import L0
        >>>
        >>> np.random.seed(42)
        >>> x = np.random.rand(10, 1) * 2.0 - 1.0
        >>> l0 = L0(c=5.0)
        >>> l0.feasible(x)
        False
        >>> l0.feasible(l0.proj(x))
        True
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        return maths.norm0(beta_) <= self.c


class LInf(interfaces.AtomicFunction,
         interfaces.Penalty,
         interfaces.Constraint,
         interfaces.ProximalOperator,
         interfaces.ProjectionOperator):
    """The proximal operator of the L-infinity function

        f(x) = l * (||x||_inf - c),

    where ||x||_inf is the L-infinity loss function. The constrainted version
    has the form

        ||x||_inf <= c.

    Parameters
    ----------
    l : Non-negative float. The Lagrange multiplier, or regularisation
            constant, of the function.

    c : Float. The limit of the constraint. The function is feasible if
            ||x||_inf <= c. The default value is c=0, i.e. the default is a
            regularisation formulation.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first index
            to be penalised. Default is 0, all columns are included.
    """
    def __init__(self, l=1.0, c=0.0, penalty_start=0):

        self.l = float(l)
        self.c = float(c)
        self.penalty_start = int(penalty_start)

    def f(self, x):
        """Function value.

        From the interface "Function".

        Parameters
        ----------
        x : Numpy array. The point at which to evaluate the function.

        Example
        -------
        >>> import numpy as np
        >>> from parsimony.functions.penalties import LInf
        >>> import parsimony.utils.maths as maths
        >>>
        >>> np.random.seed(42)
        >>> x = np.random.rand(10, 1)
        >>> linf = LInf(l=1.1)
        >>> linf.f(x) - 1.1 * maths.normInf(x)
        0.0
        """
        if self.penalty_start > 0:
            x_ = x[self.penalty_start:, :]
        else:
            x_ = x

        return self.l * (maths.normInf(x_) - self.c)

    def prox(self, x, factor=1.0):
        """The corresponding proximal operator.

        From the interface "ProximalOperator".

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.functions.penalties import LInf
        >>> import parsimony.utils.maths as maths
        >>>
        >>> np.random.seed(42)
        >>> x = np.random.rand(10, 1)
        >>> linf = LInf(l=1.45673045, c=0.5)
        >>> linf_prox = linf.prox(x)
        >>> linf_prox
        array([[ 0.37454012],
               [ 0.5       ],
               [ 0.5       ],
               [ 0.5       ],
               [ 0.15601864],
               [ 0.15599452],
               [ 0.05808361],
               [ 0.5       ],
               [ 0.5       ],
               [ 0.5       ]])
        >>> linf_proj = linf.proj(x)
        >>> linf_proj
        array([[ 0.37454012],
               [ 0.5       ],
               [ 0.5       ],
               [ 0.5       ],
               [ 0.15601864],
               [ 0.15599452],
               [ 0.05808361],
               [ 0.5       ],
               [ 0.5       ],
               [ 0.5       ]])
        >>> np.linalg.norm(linf_prox - linf_proj)
        7.5691221815410567e-09
        """
        if self.penalty_start > 0:
            x_ = x[self.penalty_start:, :]
        else:
            x_ = x

        l = self.l * factor
        l1 = L1(c=l)  # Project onto an L1 ball with radius c=l.
        y = x_ - l1.proj(x_)
        # TODO: Check if this is correct!

        # Put the unregularised variables back.
        if self.penalty_start > 0:
            y = np.vstack((x[:self.penalty_start, :],
                           y))

        return y

    def proj(self, x):
        """The corresponding projection operator.

        From the interface "ProjectionOperator".

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.functions.penalties import LInf
        >>>
        >>> np.random.seed(42)
        >>> x = np.random.rand(10, 1) * 2.0 - 1.0
        >>> linf = LInf(c=0.618)
        >>> linf.proj(x)
        array([[-0.25091976],
               [ 0.618     ],
               [ 0.46398788],
               [ 0.19731697],
               [-0.618     ],
               [-0.618     ],
               [-0.618     ],
               [ 0.618     ],
               [ 0.20223002],
               [ 0.41614516]])
        """
        if self.penalty_start > 0:
            x_ = x[self.penalty_start:, :]
        else:
            x_ = x

        if maths.normInf(x_) <= self.c:
            return x

        y = np.copy(x_)
        y[y > self.c] = self.c
        y[y < -self.c] = -self.c

        # Put the unregularised variables back.
        if self.penalty_start > 0:
            y = np.vstack((x[:self.penalty_start, :],
                           y))

        return y

    def feasible(self, x):
        """Feasibility of the constraint.

        From the interface "Constraint".

        Parameters
        ----------
        x : Numpy array. The variable to check for feasibility.

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.functions.penalties import LInf
        >>>
        >>> np.random.seed(42)
        >>> x = np.random.rand(10, 1) * 2.0 - 1.0
        >>> linf = LInf(c=0.618)
        >>> linf.feasible(x)
        False
        >>> linf.feasible(linf.proj(x))
        True
        """
        if self.penalty_start > 0:
            x_ = x[self.penalty_start:, :]
        else:
            x_ = x

        return maths.normInf(x_) <= self.c


class L2(interfaces.AtomicFunction,
         interfaces.Gradient,
         interfaces.LipschitzContinuousGradient,
         interfaces.Penalty,
         interfaces.Constraint,
         interfaces.ProximalOperator,
         interfaces.ProjectionOperator):
    """The proximal operator of the L2 function with a penalty formulation

        f(\beta) = l * (0.5 * ||\beta||²_2 - c),

    where ||\beta||²_2 is the squared L2 loss function. The constrained
    version has the form

        ||\beta||²_2 <= c.

    Parameters
    ----------
    l : Non-negative float. The Lagrange multiplier, or regularisation
            constant, of the function.

    c : Float. The limit of the constraint. The function is feasible if
            0.5 * ||\beta||²_2 <= c. The default value is c=0, i.e. the
            default is a regularised formulation.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first index
            to be penalised. Default is 0, all columns are included.
    """
    def __init__(self, l=1.0, c=0.0, penalty_start=0):

        self.l = float(l)
        self.c = float(c)
        self.penalty_start = int(penalty_start)

    def f(self, beta):
        """Function value.

        From the interface "Function".
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        return self.l * (0.5 * np.dot(beta_.T, beta_)[0, 0] - self.c)

    def grad(self, beta):
        """Gradient of the function.

        From the interface "Gradient".

        Example
        -------
        >>> import numpy as np
        >>> from parsimony.functions.penalties import L2
        >>>
        >>> np.random.seed(42)
        >>> beta = np.random.rand(100, 1)
        >>> l2 = L2(l=3.14159, c=2.71828)
        >>> np.linalg.norm(l2.grad(beta) - l2.approx_grad(beta, eps=1e-4))
        1.3549757024941964e-10
        >>>
        >>> l2 = L2(l=3.14159, c=2.71828, penalty_start=5)
        >>> np.linalg.norm(l2.grad(beta) - l2.approx_grad(beta, eps=1e-4))
        2.1291553983770027e-10
        """
#        if self.unbiased:
#            n = self.X.shape[0] - 1.0
#        else:
#            n = self.X.shape[0]

        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        grad = np.vstack((np.zeros((self.penalty_start, 1)),
                          self.l * beta_))

#        approx_grad = utils.approx_grad(self.f, beta, eps=1e-4)
#        print maths.norm(grad - approx_grad)

        return grad

    def L(self):
        """Lipschitz constant of the gradient.
        """
        return self.l

    def prox(self, beta, factor=1.0):
        """The corresponding proximal operator.

        From the interface "ProximalOperator".
        """
        l = self.l * factor
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        prox = np.vstack((beta[:self.penalty_start, :],
                          beta_ / (1.0 + l)))

        return prox

    def proj(self, beta):
        """The corresponding projection operator.

        From the interface "ProjectionOperator".

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.functions.penalties import L2
        >>> np.random.seed(42)
        >>> l2 = L2(c=0.3183098861837907)
        >>> y1 = l2.proj(np.random.rand(100, 1) * 2.0 - 1.0)
        >>> 0.5 * np.linalg.norm(y1) ** 2.0
        0.31830988618379052
        >>> y2 = np.random.rand(100, 1) * 2.0 - 1.0
        >>> l2.feasible(y2)
        False
        >>> l2.feasible(l2.proj(y2))
        True
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        sqnorm = np.dot(beta_.T, beta_)[0, 0]

        # Feasible?
        if 0.5 * sqnorm <= self.c:
            return beta

        # The correction by eps is to nudge the squared norm just below self.c.
        eps = consts.FLOAT_EPSILON
        proj = np.vstack((beta[:self.penalty_start, :],
                          beta_ * np.sqrt((2.0 * self.c - eps) / sqnorm)))

        return proj

    def feasible(self, beta):
        """Feasibility of the constraint.

        From the interface "Constraint".

        Parameters
        ----------
        beta : Numpy array. The variable to check for feasibility.

        Examples
        --------
        >>> import numpy as np
        >>> from parsimony.functions.penalties import L2
        >>> np.random.seed(42)
        >>> l2 = L2(c=0.3183098861837907)
        >>> y1 = 0.1 * (np.random.rand(50, 1) * 2.0 - 1.0)
        >>> l2.feasible(y1)
        True
        >>> y2 = 10.0 * (np.random.rand(50, 1) * 2.0 - 1.0)
        >>> l2.feasible(y2)
        False
        >>> y3 = l2.proj(50.0 * np.random.rand(100, 1) * 2.0 - 1.0)
        >>> l2.feasible(y3)
        True
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        sqnorm = np.dot(beta_.T, beta_)[0, 0]

        return 0.5 * sqnorm <= self.c


class QuadraticConstraint(interfaces.AtomicFunction,
                          interfaces.Gradient,
                          interfaces.Penalty,
                          interfaces.Constraint):
    """The proximal operator of the quadratic function

        f(x) = l * (x'Mx - c),

    or

        f(x) = l * (x'M'Nx - c),

    where M or M'N is a given symmatric positive-definite matrix. The
    constrained version has the form

        x'Mx <= c,

    or

        x'M'Nx <= c

    if two matrices are given.

    Parameters
    ----------
    l : Non-negative float. The Lagrange multiplier, or regularisation
            constant, of the function.

    c : Float. The limit of the constraint. The function is feasible if
            x'Mx <= c. The default value is c=0, i.e. the default is a
            regularisation formulation.

    M : Numpy array. The given positive definite matrix. It is assumed that
            the first penalty_start columns must be excluded.

    N : Numpy array. The second matrix if the factors of the positive-definite
            matrix are given. It is assumed that the first penalty_start
            columns must be excluded.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first index
            to be penalised. Default is 0, all columns are included.
    """
    def __init__(self, l=1.0, c=0.0, M=None, N=None, penalty_start=0):

        self.l = float(l)
        self.c = float(c)
        if self.penalty_start > 0:
            self.M = M[:, self.penalty_start:]  # NOTE! We slice M here!
            self.N = N[:, self.penalty_start:]  # NOTE! We slice N here!
        else:
            self.M = M
            self.N = N
        self.penalty_start = penalty_start

    def f(self, beta):
        """Function value.
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        if self.N is None:
            val = self.l * (np.dot(beta_.T, np.dot(self.M, beta_)) - self.c)
        else:
            val = self.l * (np.dot(beta_.T, np.dot(self.M.T,
                                                   np.dot(self.N, beta_))) \
                    - self.c)

        return val

    def grad(self, beta):
        """Gradient of the function.

        From the interface "Gradient".
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        if self.N is None:
            grad = (2.0 * self.l) * np.dot(self.M, beta_)
        else:
            grad = (2.0 * self.l) * np.dot(self.M.T, np.dot(self.N, beta_))

        grad = np.vstack(np.zeros((self.penalty_start, 1)), grad)

        return grad

    def feasible(self, beta):
        """Feasibility of the constraint.

        From the interface "Constraint".
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        if self.N is None:
            bMb = np.dot(beta_.T, np.dot(self.M, beta_))
        else:
            bMb = np.dot(beta_.T, np.dot(self.M.T, np.dot(self.N, beta_)))

        return bMb <= self.c


class RGCCAConstraint(QuadraticConstraint,
                      interfaces.ProjectionOperator):
    """The proximal operator of the quadratic function

        f(x) = l * (x'(\tau * I + ((1 - \tau) / n) * X'X)x - c),

    where \tau is a given regularisation constant. The constrained version has
    the form

        x'(\tau * I + ((1 - \tau) / n) * X'X)x <= c.

    Parameters
    ----------
    l : Non-negative float. The Lagrange multiplier, or regularisation
            constant, of the function.

    c : Float. The limit of the constraint. The function is feasible if
            x'(\tau * I + ((1 - \tau) / n) * X'X)x <= c. The default value is
            c=0, i.e. the default is a regularisation formulation.

    tau : Non-negative float. The regularisation constant.

    unbiased : Boolean. Whether the sample variance should be unbiased or not.
            Default is unbiased.

    penalty_start : Non-negative integer. The number of columns, variables
            etc., to be exempt from penalisation. Equivalently, the first index
            to be penalised. Default is 0, all columns are included.
    """
    def __init__(self, l=1.0, c=0.0, tau=1.0, X=None, unbiased=True,
                 penalty_start=0):

        self.l = float(l)
        self.c = float(c)
        self.tau = max(0.0, min(float(tau), 1.0))
        if penalty_start > 0:
            self.X = X[:, penalty_start:]  # NOTE! We slice X here!
        else:
            self.X = X
        self.unbiased = unbiased
        self.penalty_start = penalty_start

        self.reset()

    def reset(self):

        self._UP = None
        self._x_UPPtinvIVUPtVx = None

        self._D = None
        self._P = None

        self._M = None
        self._sqrtD = None
        self._Ptx = None

    def f(self, beta):
        """Function value.
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        xtMx = self._compute_value(beta_)

        return self.l * (xtMx - self.c)

    def grad(self, beta):
        """Gradient of the function.

        From the interface "Gradient".
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        if self.unbiased:
            n = float(self.X.shape[0] - 1.0)
        else:
            n = float(self.X.shape[0])

        if self.tau < 1.0:
            XtXbeta = np.dot(self.X.T, np.dot(self.X, beta_))
            grad = (self.tau * 2.0) * beta_ \
                 + ((1.0 - self.tau) * 2.0 / n) * XtXbeta
        else:
            grad = (self.tau * 2.0) * beta_

        if self.penalty_start > 0:
            grad = np.vstack(np.zeros((self.penalty_start, 1)), grad)

#        approx_grad = utils.approx_grad(self.f, beta, eps=1e-4)
#        print maths.norm(grad - approx_grad)

        return grad

    def feasible(self, beta):
        """Feasibility of the constraint.

        From the interface "Constraint".
        """
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        xtMx = self._compute_value(beta_)

        return xtMx <= self.c

    def proj(self, beta):
        """The projection operator corresponding to the function.

        From the interface "ProjectionOperator".
        """
        from parsimony.algorithms.explicit import Bisection
        bisection = Bisection(force_negative=True,
                              parameter_positive=True,
                              parameter_negative=False,
                              parameter_zero=False,
                              eps=1e-5)

        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
            X_ = self.X[:, self.penalty_start:]
        else:
            beta_ = beta
            X_ = self.X

        xtMx = self._compute_value(beta_)
        if xtMx <= self.c:
            return beta

        n, p = X_.shape

        if self.unbiased:
            n_ = float(n - 1.0)
        else:
            n_ = float(n)

        class F(interfaces.Function):
            def __init__(self, x, c, val, prox):
                self.x = x
                self.c = c
                self.val = val
                self.prox = prox
                self.y = None

            def f(self, l):
                # We avoid one evaluation of prox by saving it here.
                self.y = self.prox(self.x, l)

                return self.val(self.y) - self.c

        if self.tau == 1.0:

#            sqnorm = maths.norm(beta_) ** 2.0
            sqnorm = np.dot(beta_.T, beta_)
            eps = consts.FLOAT_EPSILON
            y = beta_ * np.sqrt((self.c - eps) / sqnorm)

#        elif p > n and self.tau == 0.0:
#
#            U = X_.T                       # p-by-n
#            V = X_                         # n-by-p
#            Vx = np.dot(V, beta_)          # n-by-1
#
#            if self._D is None:
#                VU = np.dot(V, U)    # n-by-n, VU = XX'
#                self._D, P = np.linalg.eig(VU)
#
#                self._UP = np.dot(U, P)
#                self._PtVx = np.dot(P.T, Vx)  # Pt = np.linalg.pinv(P)
#
#            def prox_2(x, l):
#                k = n_ / (2.0 * l)
#
#                invIMx = x - np.dot(self._UP,
#                                (np.reciprocal(k + self._D) * self._PtVx.T).T)
#
#                return invIMx
#
#            func = F(beta_, self.c, self._compute_value, prox_2)
#
##            l, low_start, high_start = bisection.run(func, [low, high])
#            l = bisection.run(func, [consts.TOLERANCE, p])
#
#            y = func.y

#        elif p > n:
#            U = X_.T                           # p-by-n
#            V = X_                             # n-by-p
#            Vx = np.dot(V, beta_)              # n-by-1
#
#            k = self.tau + 1.0
#            m = ((1.0 - self.tau) / n_)
#
#            if self._x_UPPtinvIVUPtVx is None:
#                VU = np.dot(V, U)              # n-by-n, VU = XX'
#                D, P = np.linalg.eig(VU)
#                PtVx = Vx  # np.dot(P.T, Vx)  # Pt = np.linalg.pinv(P)
#                UPPtinvIVUPtVx = np.dot(U,  # np.dot(U, P),
#                                     (np.reciprocal(D + (k / m)) * PtVx.T).T)
#                self._x_UPPtinvIVUPtVx = (beta_ - UPPtinvIVUPtVx) / k
#
#            def prox_3(x, l):
#
#                invIMx = self._x_UPPtinvIVUPtVx / l
#
#                return invIMx
#
#            func = F(beta_, self.c, self._compute_value, prox_3)
#
##            l, low_start, high_start = bisection.run(func, [low, high])
##            l = bisection.run(func, [consts.TOLERANCE, 1.0])
#            if p >= 10000 and n >= 500:
#                l = bisection.run(func, [p / 5, p])
#            elif p >= 1000 and n >= 100:
#                l = bisection.run(func, [p / 100, p / 10])
#            else:
#                l = bisection.run(func, [consts.TOLERANCE, 1.0])
#            print "l:", l
#
#            y = func.y
#
#        else:  # The case when: p <= n
#
#            if self._M is None:
#                Ip = np.eye(p)
#                XtX = np.dot(X_.T, X_)
#
#                self._M = self.tau * Ip + \
#                          ((1.0 - self.tau) / n_) * XtX
#
#            if self._D is None or self._P is None:
#                self._D, self._P = np.linalg.eig(self._M)
#
#                self._sqrtD = np.sqrt(self._D)
#                self._Ptx = np.dot(self._P.T, beta_)
#
#            def prox_4(x, l):
#
#                invIM = np.dot(self._P * \
#                               np.reciprocal(1.0 + (2.0 * l) * self._D),
#                               self._P.T)
#                y = np.dot(invIM, x)
#
#                return y
#
##            func = F(beta_, self.c, self._compute_value, prox_4)
##            l = bisection.run(func, [consts.TOLERANCE, 1.0])
##
##            y = func.y
#
#            class F(interfaces.Function):
#                def __init__(self, c, D, sqrtD, Ptx):
#                    self.c = c
#                    self.D = D
#                    self.sqrtD = sqrtD
#                    self.Ptx = Ptx
#
#                def f(self, l):
#
#                    K = np.reciprocal(1.0 + (2.0 * l) * self.D)
#                    sqrtDK = self.sqrtD * K  # Slightly faster than np.multiply
#                    c = maths.norm((sqrtDK * self.Ptx.T).T) ** 2.0
#
#                    return c - self.c
#
#            func = F(self.c, self._D, self._sqrtD, self._Ptx)
#            low_start = consts.TOLERANCE
#            high_start = 1.0
##            low_start = np.max((consts.TOLERANCE, np.log10(n)))
##            high_start = np.max((1.0, 1.0 + np.log10(n) ** 2.0))
#            l = bisection.run(func, [low_start, high_start])
##            print "l:", l
#            y = prox_4(beta_, l)

        else:
#            _, l, P = np.linalg.svd(np.sqrt((1.0 - self.tau) / n_) * X_, full_matrices=0)
#            l **= 2.0
#            l += self.tau
            _, l, P = np.linalg.svd(X_, full_matrices=0)
            l =  ((1.0 - self.tau) / n_) * (l ** 2.0) + self.tau
            l = l.reshape((min(n, p), 1))

            def ssq(vec):
                return np.sum(vec ** 2.0)

            atilde = np.dot(P, beta_)
            ssdiff = np.sum(beta_ ** 2.0) - np.sum(atilde ** 2.0)
            atilde2 = atilde ** 2.0
            atilde2lambdas = atilde2 * l
            atilde2lambdas2 = atilde2 * l ** 2.0
            tau2 = self.tau ** 2.0

            def f(mu):
                term1 = (self.tau / ((1.0 + 2.0 * mu * self.tau) ** 2.0)) * ssdiff
                term2 = np.sum(atilde2lambdas / ((1.0 + 2.0 * mu * l) ** 2.0))
                return term1 + term2 - self.c

            def df(mu):
                term1 = -4.0 * tau2 / ((1.0 + 2.0 * mu * self.tau) ** 3.0) \
                            * ssdiff
                term2 = -4.0 * np.sum(atilde2lambdas2 \
                            / ((1.0 + 2.0 * mu * l) ** 3.0))
                return term1 + term2

            print "f :", f(0.0)
            print "df:", df(0.0)

            from parsimony.algorithms.explicit import NewtonRaphson
            newton = NewtonRaphson(force_negative=True,
                                   parameter_positive=True,
                                   parameter_negative=False,
                                   parameter_zero=False,
                                   eps=consts.TOLERANCE)
            class F(interfaces.Function):

                def f(self, x):    
                    return f(x)

                def grad(self, x):
                    return df(x)

            mu = newton.run(F(), 1.0)

            print "mu       :", mu

            if p > n:
                D, P = np.linalg.eig(np.dot(X_, X_.T))

                a = 1 + 2.0 * mu * self.tau
                b = 2.0 * mu * (1.0 - self.tau) / n_
                PtXbeta = np.dot(P.T, np.dot(X_, beta_))
                y = (beta_ - np.dot(X_.T,
                                    np.dot(np.reciprocal(D + (a / b)) * P,
                                           PtXbeta))) / a

                Ip = np.eye(p)
                M = self.tau * Ip + ((1.0 - self.tau) / n_) * np.dot(X_.T, X_)
                invIM = np.linalg.inv(Ip + (2.0 * mu) * M)
                y_ = np.dot(invIM, beta_)
                print "err:", np.linalg.norm(y - y_)
#            print "yMy      :", np.dot(y.T, np.dot(M, y))[0, 0]
            print "f(beta_) :", self.f(y)

#            class F(interfaces.Function):
#                def __init__(self, c):
#                    self.c = c
#
#                def f(self, mu):
#
#                    y = np.dot(np.linalg.inv(Ip + (2.0 * mu) * M), beta_)
#
#                    return np.dot(y.T, np.dot(M, y)) - self.c
#
#            func = F(self.c)
#            low_start = consts.TOLERANCE
#            high_start = 1.0
#            mu = bisection.run(func, [low_start, high_start])
#            print "mu       :", mu
#            y = np.dot(np.linalg.inv(Ip + (2.0 * mu) * M), beta_)
#            print "yMy      :", np.dot(y.T, np.dot(M, y))

        if self.penalty_start > 0:
            y = np.vstack((beta[:self.penalty_start, :], y))

        return y
#        return y, low_start, high_start

    def _compute_value(self, beta):
        """Helper function to compute the function value.

        Note that beta must already be sliced!
        """

        if self.unbiased:
            n = float(self.X.shape[0] - 1.0)
        else:
            n = float(self.X.shape[0])

        Xbeta = np.dot(self.X, beta)
        val = self.tau * np.dot(beta.T, beta) \
            + ((1.0 - self.tau) / n) * np.dot(Xbeta.T, Xbeta)

        return val[0, 0]


class SufficientDescentCondition(interfaces.Function,
                                 interfaces.Constraint):

    def __init__(self, function, p, c):
        """The sufficient condition

            f(x + a * p) <= f(x) + c * a * grad(f(x))'p

        for descent. This condition is sometimes called the Armijo condition.

        Parameters
        ----------
        c : Float, 0 < c < 1. A constant for the condition. Should be small.
        """
        self.function = function
        self.p = p
        self.c = c

    def f(self, x, a):

        return self.function.f(x + a * self.p)

    """Feasibility of the constraint at point x with step a.

    From the interface "Constraint".
    """
    def feasible(self, xa):

        x = xa[0]
        a = xa[1]

        f_x_ap = self.function.f(x + a * self.p)
        f_x = self.function.f(x)
        grad_p = np.dot(self.function.grad(x).T, self.p)[0, 0]
#        print "f_x_ap = %.10f, f_x = %.10f, grad_p = %.10f, feas = %.10f" % (f_x_ap, f_x, grad_p, f_x + self.c * a * grad_p)
#        if grad_p >= 0.0:
#            pass
        feasible = f_x_ap <= f_x + self.c * a * grad_p

        return feasible


#class WolfeCondition(Function, Constraint):
#
#    def __init__(self, function, p, c1=1e-4, c2=0.9):
#        """
#        Parameters:
#        ----------
#        c1 : Float. 0 < c1 < c2 < 1. A constant for the condition. Should be
#                small.
#        c2 : Float. 0 < c1 < c2 < 1. A constant for the condition. Depends on
#                the minimisation algorithms. For Newton or quasi-Newton
#                descent directions, 0.9 is a good choice. 0.1 is appropriate
#                for nonlinear conjugate gradient.
#        """
#        self.function = function
#        self.p = p
#        self.c1 = c1
#        self.c2 = c2
#
#    def f(self, x, a):
#
#        return self.function.f(x + a * self.p)
#
#    """Feasibility of the constraint at point x.
#
#    From the interface "Constraint".
#    """
#    def feasible(self, x, a):
#
#        grad_p = np.dot(self.function.grad(x).T, self.p)[0, 0]
#        cond1 = self.function.f(x + a * self.p) \
#            <= self.function.f(x) + self.c1 * a * grad_p
#        cond2 = np.dot(self.function.grad(x + a * self.p).T, self.p)[0, 0] \
#            >= self.c2 * grad_p
#
#        return cond1 and cond2
#
#
#class StrongWolfeCondition(Function, Constraint):
#
#    def __init__(self, function, p, c1=1e-4, c2=0.9):
#        """
#        Parameters:
#        ----------
#        c1 : Float. 0 < c1 < c2 < 1. A constant for the condition. Should be
#                small.
#        c2 : Float. 0 < c1 < c2 < 1. A constant for the condition. Depends on
#                the minimisation algorithms. For Newton or quasi-Newton
#                descent directions, 0.9 is a good choice. 0.1 is appropriate
#                for nonlinear conjugate gradient.
#        """
#        self.function = function
#        self.p = p
#        self.c1 = c1
#        self.c2 = c2
#
#    def f(self, x, a):
#
#        return self.function.f(x + a * self.p)
#
#    """Feasibility of the constraint at point x.
#
#    From the interface "Constraint".
#    """
#    def feasible(self, x, a):
#
#        grad_p = np.dot(self.function.grad(x).T, self.p)[0, 0]
#        cond1 = self.function.f(x + a * self.p) \
#            <= self.function.f(x) + self.c1 * a * grad_p
#        grad_x_ap = self.function.grad(x + a * self.p)
#        cond2 = abs(np.dot(grad_x_ap.T, self.p)[0, 0]) \
#            <= self.c2 * abs(grad_p)
#
#        return cond1 and cond2