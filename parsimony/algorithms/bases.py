# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.algorithms.bases` module includes several base classes
for using and creating algorithms.

Algorithms may not store states. I.e., if they are classes, do not keep
references to objects with state in the algorithm objects. It should be
possible to copy and share algorithms between e.g. estimators, and thus they
should not depend on any state.

There are currently two main types of algorithms: implicit and explicit. The
difference is whether they run directly on the data (implicit) or if they have
an actual loss function than is minimised (explicit). Implicit algorithms take
the data as input, and then run on the data. Explicit algorithms take a loss
function and possibly a start vector as input, and then minimise the function
value starting from the point of the start vector.

Algorithms that don't fit well in either category should go in utils instead.

Created on Thu Feb 20 17:42:16 2014

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import abc

import parsimony.functions.interfaces as interfaces

__all__ = ["BaseAlgorithm", "ImplicitAlgorithm", "ExplicitAlgorithm"]


class BaseAlgorithm(object):

    def check_compatibility(self, function, required_interfaces):
        """Check if the function considered implements the given interfaces.
        """
        for interface in required_interfaces:
            if isinstance(interface, interfaces.OR):
                if not interface.evaluate(function):
                    raise ValueError("%s does not implement interfaces %s" %
                                    (str(function), str(interface)))
            elif not isinstance(function, interface):
                raise ValueError("%s does not implement interface %s" %
                                (str(function), str(interface)))

    def set_params(self, **kwargs):

        for k in kwargs:
            self.__setattr__(k, kwargs[k])


class ImplicitAlgorithm(BaseAlgorithm):
    """Implicit algorithms are algorithms that do not use a loss function, but
    instead minimise or maximise some underlying function implicitly, from the
    data.

    Parameters
    ----------
    X : Regressor
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def run(X, **kwargs):
        raise NotImplementedError('Abstract method "run" must be ' \
                                  'specialised!')


class ExplicitAlgorithm(BaseAlgorithm):
    """Explicit algorithms are algorithms that minimises a given function
    explicitly from properties of the function.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def run(function, beta, **kwargs):
        """Run this algorithm to obtain the variable that gives the minimum of
        the give function(s).

        Parameters
        ----------
        function : The function to minimise.

        beta : A start vector.
        """
        raise NotImplementedError('Abstract method "run" must be ' \
                                  'specialised!')