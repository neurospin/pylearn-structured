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
function and possibly a starting point as input, and then minimise the function
value starting from the point of the start vector.

Algorithms that don't fit well in either category should go in utils instead.

Created on Thu Feb 20 17:42:16 2014

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import abc
import functools

import parsimony.utils.consts as consts
import parsimony.functions.interfaces as interfaces
from parsimony.utils import LimitedDict

__all__ = ["BaseAlgorithm", "check_compatibility",
           "ImplicitAlgorithm", "ExplicitAlgorithm",
           "IterativeAlgorithm", "InformationAlgorithm"]


class BaseAlgorithm(object):

    @staticmethod
    def check_compatibility(function, required_interfaces):
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


def check_compatibility(f):
    """Automatically checks if a function implements a given set of interfaces.
    """
    @functools.wraps(f)
    def wrapper(self, function, beta=None, *args, **kwargs):

        BaseAlgorithm.check_compatibility(function, self.INTERFACES)

        return f(self, function, beta, *args, **kwargs)

    return wrapper


class ImplicitAlgorithm(BaseAlgorithm):
    """Implicit algorithms are algorithms that do not utilise a loss function.

    Implicit algorithms instead minimise or maximise some underlying function
    implicitly, usually from the data.

    Parameters
    ----------
    X : One or more data matrices.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def run(X, **kwargs):
        raise NotImplementedError('Abstract method "run" must be ' \
                                  'specialised!')


class ExplicitAlgorithm(BaseAlgorithm):
    """Explicit algorithms are algorithms that minimises a given function.

    The function is explicitly minimised from properties of said function.

    Implementing classes should update the INTERFACES class variable with
    the interfaces that function must implement. Defauls to a list with one
    element, the Function.
    """
    __metaclass__ = abc.ABCMeta

    INTERFACES = [interfaces.Function]

    @abc.abstractmethod
    def run(function, x, **kwargs):
        """This function obtains a minimiser of a give function.

        Parameters
        ----------
        function : The function to minimise.

        x : A starting point.
        """
        raise NotImplementedError('Abstract method "run" must be ' \
                                  'specialised!')


class IterativeAlgorithm(object):
    """Algorithms that require iterative steps to achieve the goal.

    Parameters
    ----------
    max_iter : Non-negative integer. The maximum number of allowed iterations.

    min_iter : Non-negative integer. The minimum number of required iterations.
    """
    def __init__(self, max_iter=consts.MAX_ITER, min_iter=1, **kwargs):
        super(IterativeAlgorithm, self).__init__(**kwargs)

        self.max_iter = max_iter
        self.min_iter = min_iter


class InformationAlgorithm(object):
    """Algorithms that produce information about their run.

    Implementing classes should update the PROVIDED_INFO class variable with
    the information provided by the algorithm. Defauls to an empty list.

    Examples
    --------
    >>> import parsimony.algorithms as algorithms
    >>> from parsimony.utils import LimitedDict, Info
    >>> from parsimony.functions.losses import LinearRegression
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> gd = algorithms.explicit.GradientDescent(info=LimitedDict(Info.f))
    >>> gd.info  # doctest:+ELLIPSIS
    LimitedDict(set([EnumItem('Info', 'f', ...)])).update({})
    >>> lr = LinearRegression(X=np.random.rand(10,15), y=np.random.rand(10,1))
    >>> beta = gd.run(lr, np.random.rand(15, 1))
    >>> gd.info[Info.f]  # doctest:+ELLIPSIS
    [0.068510926021035312, ... 1.8856122733915382e-12]
    """
    PROVIDED_INFO = []

    def __init__(self, info=None, **kwargs):
        """
        Parameters
        ----------
        info : LimitedDict. The data structure to store the run information in.
        """
        super(InformationAlgorithm, self).__init__(**kwargs)

        if info == None:
            self.info = LimitedDict()
        else:
            self.info = info

        self.check_info_compatibility(self.PROVIDED_INFO)

    def check_info_compatibility(self, req_info):
        for i in self.info:
            if i not in req_info:
                raise ValueError("Requested information unknown.")