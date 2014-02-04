# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.functions.interfaces` module contains interfaces that
describes the functionality of the multiblock functions.

Try to keep the inheritance tree loop-free unless absolutely impossible.

Created on Mon Feb  3 09:55:51 2014

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import abc

import parsimony.functions.interfaces as interfaces

__all__ = ["MultiblockFunction", "MultiblockGradient",
           "MultiblockLipschitzContinuousGradient",
           "MultiblockProximalOperator", "MultiblockProjectionOperator"]


class MultiblockFunction(interfaces.CompositeFunction):
    """ This is a function that is the combination (i.e. sum) of other
    multiblock, composite or atomic functions. The difference from
    CompositeFunction is that this function assumes that relevant functions
    accept an index, i, that is the block we are working with.
    """
    __metaclass__ = abc.ABCMeta

    constraints = dict()

    def add_constraint(self, function, index):
        """Add a constraint to this function.
        """
        if index in self.constraints:
            self.constraints[index].append(function)
        else:
            self.constraints[index] = [function]

    def get_constraints(self, index):
        """Returns the constraint functions for the function with the given
        index. Returns an empty list if no constraint functions exist for the
        given index.
        """
        if index in self.constraints:
            return self.constraints[index]
        else:
            return []


class MultiblockGradient(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def grad(self, beta, index):
        """Gradient of the function.

        Parameters
        ----------
        beta : The point at which to evaluate the gradient.
        """
        raise NotImplementedError('Abstract method "grad" must be '
                                  'specialised!')


class MultiblockLipschitzContinuousGradient(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def L(self, w, index):
        """Lipschitz constant of the gradient with given index.
        """
        raise NotImplementedError('Abstract method "L" must be '
                                  'specialised!')


class MultiblockProximalOperator(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def prox(self, beta, index, factor=1.0):
        """The proximal operator corresponding to the function with the given
        index.
        """
        raise NotImplementedError('Abstract method "prox" must be '
                                  'specialised!')


class MultiblockProjectionOperator(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def proj(self, beta, index):
        """The projection operator corresponding to the function with the
        given index.
        """
        raise NotImplementedError('Abstract method "proj" must be '
                                  'specialised!')