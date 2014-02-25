# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.functions.interfaces` module contains interfaces that
describes the functionality of the functions.

Try to keep the inheritance tree loop-free unless absolutely impossible.

Created on Mon Apr 22 10:54:29 2013

@author:  Tommy Löfstedt, Vincent Guillemot, Edouard Duchesnay and
          Fouad Hadj-Selem
@email:   lofstedt.tommy@gmail.com, edouard.duchesnay@cea.fr
@license: BSD 3-clause.
"""
import abc

import numpy as np

__all__ = ["Function", "AtomicFunction", "CompositeFunction",
           "Penalty", "Constraint",
           "ProximalOperator", "ProjectionOperator",
           "CombinedProjectionOperator",
           "Continuation",
           "Gradient", "Hessian", "LipschitzContinuousGradient", "StepSize",
           "GradientMap", "DualFunction", "Eigenvalues", "StronglyConvex"]


class Function(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def f(self, *args, **kwargs):
        """Function value.
        """
        raise NotImplementedError('Abstract method "f" must be '
                                  'specialised!')

    def reset(self):
        """Free any cached computations from previous use of this Function.
        """
        pass

    def set_params(self, **kwargs):

        for k in kwargs:
            self.__setattr__(k, kwargs[k])


class AtomicFunction(Function):
    """ This is a function that is not in general supposed to be minimised by
    itself. Instead it should be combined with other atomic functions and
    composite functions into composite functions.
    """
    __metaclass__ = abc.ABCMeta


class CompositeFunction(Function):
    """ This is a function that is the combination (e.g. sum) of other
    composite or atomic functions. It may also be a constrained function.
    """
    __metaclass__ = abc.ABCMeta

#    constraints = list()
#
#    def add_constraint(self, function):
#        """Add a constraint to this function.
#        """
#        self.constraints.append(function)
#
#    def get_constraints(self):
#        """Returns the constraint functions for this function. Returns an
#        empty list if no constraint functions exist.
#        """
#        return self.constraints


class Penalty(object):
    """Represents the penalisation of a function.

    Penalties must take a parameter penalty_start, with default value 0.
    Columns, variables or corresponding entities with indices smaller than
    penalty_start must not be penalised.

    Parameters
    ----------
    penalty_start : Non-negative integer. The number of columns, variables
            etc., to except from penalisation. Equivalently, the first index
            to be penalised. Default is 0, all columns are included.
    """
    __metaclass__ = abc.ABCMeta


# TODO: Should all constraints have the projection operator?
class Constraint(object):
    """Represents a constraint of a function.

    Constraints must take a parameter penalty_start, with default value 0.
    Columns, variables or corresponding entities with indices smaller than
    penalty_start must not be penalised.

    Parameters
    ----------
    penalty_start : The number of columns, variables etc., to except from
            penalisation. Equivalently, the first index to be penalised.
            Default is 0, all columns are included.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def feasible(self, x):
        """Feasibility of the constraint at point x.
        """
        raise NotImplementedError('Abstract method "feasible" must be '
                                  'specialised!')


class ProximalOperator(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def prox(self, beta, factor=1.0):
        """The proximal operator corresponding to the function.
        """
        raise NotImplementedError('Abstract method "prox" must be '
                                  'specialised!')


class ProjectionOperator(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def proj(self, beta):
        """The projection operator corresponding to the function.
        """
        raise NotImplementedError('Abstract method "proj" must be '
                                  'specialised!')


class CombinedProjectionOperator(Function, ProjectionOperator):

    def __init__(self, functions):
        """Functions must currently be a tuple or list with two projection
        operators.
        """
        self.functions = functions

#        from algorithms import ProjectionADMM
#        self.proj_op = ProjectionADMM()
        from algorithms import DykstrasProjectionAlgorithm
        self.proj_op = DykstrasProjectionAlgorithm()

    def f(self, x):

        val = 0
        for func in self.functions:
            val += func.f(x)

        return val

    def proj(self, x):
        """The projection operator corresponding to the function.

        From the interface "ProjectionOperator".
        """
#        proj1 = self.proj_op(self.functions, x)
        proj = self.proj_op(self.functions, x)

#        print "diff:", np.linalg.norm(proj1 - proj2)

        return proj


# TODO: This should be in the NesterovFunction interface, right?
class Continuation(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def mu_opt(self, eps):
        """The optimal value of \mu given \epsilon.
        """
        raise NotImplementedError('Abstract method "mu_opt" must be '
                                  'specialised!')

    @abc.abstractmethod
    def eps_opt(self, mu):
        """The optimal value of \epsilon given \mu.
        """
        raise NotImplementedError('Abstract method "eps_opt" must be '
                                  'specialised!')

    @abc.abstractmethod
    def eps_max(self, mu):
        """The maximum value of \epsilon.
        """
        raise NotImplementedError('Abstract method "eps_max" must be '
                                  'specialised!')


class Gradient(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def grad(self, beta):
        """Gradient of the function.

        Parameters
        ----------
        beta : The point at which to evaluate the gradient.
        """
        raise NotImplementedError('Abstract method "grad" must be '
                                  'specialised!')

    def approx_grad(self, x, eps=1e-4):
        """Numerical approximation of the gradient.

        Parameters
        ----------
        beta : The point at which to evaluate the gradient.

        eps : The precision of the numerical solution. Smaller is better, but
                too small may result in floating point precision errors.
        """
        p = x.shape[0]
        grad = np.zeros(x.shape)
        if isinstance(self, (Penalty, Constraint)):
            start = self.penalty_start
        else:
            start = 0
        for i in xrange(start, p):
            x[i, 0] -= eps
            loss1 = self.f(x)
            x[i, 0] += 2.0 * eps
            loss2 = self.f(x)
            x[i, 0] -= eps
            grad[i, 0] = (loss2 - loss1) / (2.0 * eps)

        return grad


class Hessian(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def hessian(self, beta, vector=None):
        """The Hessian of the function.

        Parameters
        ----------
        beta : The point at which to evaluate the Hessian.

        vector : If not None, it is multiplied with the Hessian from the right.
        """
        raise NotImplementedError('Abstract method "hessian" must be '
                                  'specialised!')

    @abc.abstractmethod
    def hessian_inverse(self, beta, vector=None):
        """Inverse of the Hessian (second derivative) of the function.

        Sometimes this can be done efficiently if we know the structure of the
        Hessian. Also, if we multiply the Hessian by a vector, it is often
        possible to do efficiently.

        Parameters
        ----------
        beta : The point at which to evaluate the Hessian.

        vector : If not None, it is multiplied with the inverse of the Hessian
                from the right.
        """
        raise NotImplementedError('Abstract method "hessian_inverse" must be '
                                  'specialised!')


class LipschitzContinuousGradient(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def L(self):
        """Lipschitz constant of the gradient.
        """
        raise NotImplementedError('Abstract method "L" must be '
                                  'specialised!')


class StepSize(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def step(self, beta, index=0):
        """The step size to use in descent methods.

        Parameters
        ----------
        beta : Numpy array. The point at which to determine the step size.

        index : Non-negative integer. For multiblock functions, to know which
                variable the step is for.
        """
        raise NotImplementedError('Abstract method "step" must be '
                                  'specialised!')


class GradientMap(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def V(self, alpha, beta, L):
        """The gradient map associated to the function.
        """
        raise NotImplementedError('Abstract method "V" must be '
                                  'specialised!')


class DualFunction(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def gap(self, beta, beta_hat):
        """Compute the duality gap.
        """
        raise NotImplementedError('Abstract method "gap" must be '
                                  'specialised!')

    @abc.abstractmethod
    def betahat(self, alpha, beta=None):
        """Return the beta that minimises the dual function.
        """
        raise NotImplementedError('Abstract method "betahat" must be '
                                  'specialised!')


class Eigenvalues(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def lambda_max(self):
        """Largest eigenvalue of the corresponding covariance matrix.
        """
        raise NotImplementedError('Abstract method "lambda_max" must be '
                                  'specialised!')

    def lambda_min(self):
        """Smallest eigenvalue of the corresponding covariance matrix.
        """
        raise NotImplementedError('Abstract method "lambda_min" is not '
                                  'implemented!')


class StronglyConvex(object):
    """Represents strongly convex functions.

    A function is strongly convex with parameter m if

        (grad(f(x) - grad(f(y))'(x - y) >= m.||x - y||²_2,

    or equivalently if

        H(f(x)) >= mI,

    where H is the Hessian, I is the identity matrix. The second ">=" means
    that H(f(x)) - mI is positive semi-definite.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def parameter(self):
        """Returns the strongly convex parameter for the function.
        """
        raise NotImplementedError('Abstract method "parameter" is not '
                                  'implemented!')