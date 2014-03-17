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
        bisection = Bisection(force_negative=True, eps=1e-8)

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

        self._VU = None
        self._Pt = None
        self._UP = None

#        self._Ip = None
        self._M = None

        self._D = None
        self._P = None

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
            n = self.X.shape[0] - 1.0
        else:
            n = self.X.shape[0]

        if self.tau < 1.0:
            XtXbeta = np.dot(self.X.T, np.dot(self.X, beta_))
            grad = (self.tau * 2.0) * beta_ \
                 + ((1.0 - self.tau) * 2.0 / float(n)) * XtXbeta
        else:
            grad = (self.tau * 2.0) * beta_

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
        if self.penalty_start > 0:
            beta_ = beta[self.penalty_start:, :]
        else:
            beta_ = beta

        xtMx = self._compute_value(beta_)
        if xtMx <= self.c:
            return beta, beta

        n, p = self.X.shape
        if p > n:
#            In = np.eye(n)                    # n-by-n
            U = self.X.T                      # p-by-n
#            V = self.X                        # n-by-p
            Vx = np.dot(self.X, beta_)        # n-by-1
            if self._VU is None:
                self._VU = np.dot(self.X, U)  # XX', n-by-n

#                self._P, self._D, self._Pt = np.linalg.svd(self._VU)
                self._D, self._P = np.linalg.eig(self._VU)
#                self._Pt = np.linalg.pinv(self._P)
                self._Pt = self._P.T
                self._UP = np.dot(U, self._P)

            if self.unbiased:
                n_ = float(n - 1.0)
            else:
                n_ = float(n)

            def prox(x, l):
                k = 0.5 * l * self.tau + 1.0
                m = 0.5 * l * ((1.0 - self.tau) / n_)

#                invIVU = np.linalg.inv((k / m) * In + self._VU)
#                invIVU = np.dot(self._P * np.reciprocal(self._D + (k / m)), self._Pt)

                PtinvIVU = (np.reciprocal(self._D + (k / m)) * self._Pt.T).T
#                invIMx = (x - np.dot(U, np.dot(invIVU, Vx))) / k
                invIMx = (x - np.dot(self._UP, np.dot(PtinvIVU, Vx))) / k

                return invIMx

            from parsimony.algorithms import Bisection
            bisection = Bisection(force_negative=True,
                                  parameter_positive=True,
                                  parameter_negative=False,
                                  parameter_zero=False,
                                  eps=1e-6)

            class F(interfaces.Function):
                def __init__(self, x, c, val):
                    self.x = x
                    self.c = c
                    self.val = val
                    self.y = None

                def f(self, l):

                    # We avoid one evaluation of prox by saving it here.
                    self.y = prox(self.x, l)

                    return self.val(self.y) - self.c

            func = F(beta_, self.c, self._compute_value)

            # TODO: Tweak these magic numbers on real data. Or even better,
            # find theoretical bounds. Convergence is faster if these bounds
            # are close to accurate when we start the bisection algorithm.
            if p >= 600000:
                low = (p / 100.0) - np.log10(n) * 5.0
                high = (p / 80.0) - np.log10(n) * 5.0
            elif p >= 500000:
                low = p / 85.71
                high = (p / 76.9) - np.log10(n) * 5.0
            elif p >= 400000:
                low = p / 78.125
                high = (p / 68.97) - np.log10(n) * 5.0
            elif p >= 300000:
                low = p / 70.17
                high = (p / 59.4) - np.log10(n) * 5.0
            elif p >= 200000:
                low = p / 61.22
                high = (p / 48.78) - np.log10(n) * 5.0
            elif p >= 150000:
                low = p / 50.0
                high = (p / 41.66) - np.log10(n) * 5.0  # ^^
            elif p >= 100000:
                low = p / 42.86
                high = (p / 34.5) - np.log10(n) * 6.0  # !
            elif p >= 75000:
                low = p / 35.71
                high = (p / 29.9) - np.log10(n) * 6.0  # !
            elif p >= 50000:
                low = p / 31.25
                high = (p / 23.81) - np.log10(n) * 6.0  # !
            elif p >= 25000:
                low = p / 25.0
                high = (p / 16.67) - np.log10(n) * 7.0  # !
            elif p >= 10000:
                low = p / 17.86
                high = (p / 10.87) - np.log10(n) * 7.0  # !
            elif p >= 5000:
                low = p / 11.63
                high = (p / 7.69) - np.log10(n) * 7.0  # !
            elif p >= 1000:
                low = p / 8.62
                high = (p / 3.45) - np.log10(n) * 8.0  # !
            else:
                low = p / 4.75
                high = p / 2.25

            l = bisection(func, [low, high])

            y = func.y

        else:  # The case when: p <= n

#            if self._Ip is None:
#                self._Ip = np.eye(p)  # p-by-p

            if self._M is None:
                XtX = np.dot(self.X.T, self.X)

                if self.unbiased:
                    n_ = float(n - 1.0)
                else:
                    n_ = float(n)

                Ip = np.eye(p)
                self._M = self.tau * Ip + \
                          ((1.0 - self.tau) / n_) * XtX

                self._D, self._P = np.linalg.eig(self._M)

            def prox2(x, l):

#                y = np.dot(np.linalg.inv(self._Ip + (0.5 * l) * self._M), x)

#                invIM = np.linalg.inv(self._Ip + (0.5 * l) * self._M)
#                print maths.norm(np.linalg.inv(self._Ip + (0.5 * l) * self._M) - \
#                                  np.dot(self._P * np.reciprocal(0.5 * l * self._D + 1.0),
#                                         self._P.T))
#                print maths.norm(self._M - np.dot(self._P, np.dot(np.diag(self._D), self._P.T)))
#                print maths.norm((self._Ip + (0.5 * l) * self._M) - \
#                                  np.dot(self._P,
#                                         np.dot(np.diag(self._D + 1.0 + 0.5 * l),
#                                                self._P.T)))

#                invIM = np.linalg.inv(self._Ip + (0.5 * l) * self._M)
                invIM = np.dot(self._P * \
                                   np.reciprocal(0.5 * l * self._D + 1.0),
                               self._P.T)
                y = np.dot(invIM, x)

#                print "err:", maths.norm(y - yd)

                return y

            from parsimony.algorithms import Bisection
            bisection = Bisection(force_negative=True,
                                  parameter_positive=True,
                                  parameter_negative=False,
                                  parameter_zero=False,
                                  eps=1e-6,
                                  max_iter=100)

            class F(interfaces.Function):
                def __init__(self, x, c, val):
                    self.x = x
                    self.c = c
                    self.val = val
                    self.y = None

                def f(self, l):

                    # We avoid one evaluation of prox by saving it here.
                    self.y = prox2(self.x, l)

                    return self.val(self.y) - self.c

            func = F(beta_, self.c, self._compute_value)

            # TODO: Tweak these magic numbers on real data. Or even better,
            # find theoretical bounds. Convergence is faster if these bounds
            # are close to accurate when we start the bisection algorithm.
            if p >= 950:
                low = p / 5.25
                high = (p / 4.50) - np.log10(n)  # !
            elif p >= 850:
                low = p / 4.65
                high = (p / 4.25) - np.log10(n)  # !
            elif p >= 750:
                low = p / 4.45
                high = (p / 4.00) - np.log10(n)  # !
            elif p >= 650:
                low = p / 4.28
                high = (p / 3.70) - np.log10(n)  # !
            elif p >= 550:
                low = p / 4.10
                high = (p / 3.40) - np.log10(n)  # !
            elif p >= 450:
                low = p / 3.85
                high = (p / 3.05) - np.log10(n)  # !
            elif p >= 350:
                low = p / 3.59
                high = (p / 2.82) - np.log10(n)  # !
            elif p >= 250:
                low = p / 3.16
                high = (p / 2.42) - np.log10(n)  # !
            elif p >= 150:
                low = p / 2.7
                high = (p / 1.85) - np.log10(n)  # !
            elif p >= 50:
                low = p / 2.23
                high = (p / 1.23) - np.log10(n)  # !
            else:
                low = p / 1.1
                high = p / 0.8  # !

            l = bisection(func, [low, high])

            y = func.y

#        print low, ", ", high
#        print l

        _Ip = np.eye(p)  # p-by-p

        XtX = np.dot(self.X.T, self.X)
        _M = self.tau * _Ip + ((1.0 - self.tau) / float(n - 1)) * XtX
        _D, _P = np.linalg.eig(_M)

#        l = 2.0 * max(0.0, np.sqrt(xtMx) - self.c)
        y_ = np.dot(np.linalg.inv(_Ip + (0.5 * l) * _M), beta_)

        sqrtD = np.sqrt(_D)
        K = np.reciprocal(1.0 + (0.5 * l) * _D)
        sqrtDK = sqrtD * K  # Slightly faster than np.multiply.
        Ptx = np.dot(_P.T, beta)
        c = maths.norm((sqrtDK * Ptx.T).T) ** 2.0
        print "c:", c
        c = maths.norm(np.dot(np.diag(sqrtDK), Ptx)) ** 2.0
        print "c:", c
        print "y'My:", np.dot(y.T, np.dot(_M, y))
        print "y_'My_:", np.dot(y_.T, np.dot(_M, y_))

#        if maths.norm(beta_ - (beta_ / np.sqrt(xtMx))) < maths.norm(beta_ - y):
#            print maths.norm(beta_ - (beta_ / np.sqrt(xtMx)))
#            print maths.norm(beta_ - y)

        return y#, y_

    def _compute_value(self, beta):
        """Helper function to compute the function value.

        Note that beta must already be sliced!
        """

        if self.unbiased:
            n = self.X.shape[0] - 1.0
        else:
            n = self.X.shape[0]

        Xbeta = np.dot(self.X, beta)
        val = self.tau * np.dot(beta.T, beta) \
            + ((1.0 - self.tau) / float(n)) * np.dot(Xbeta.T, Xbeta)

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