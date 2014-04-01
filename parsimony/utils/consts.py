# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 11:26:45 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD 3-clause.
"""
import numpy as np

__all__ = ["TOLERANCE", "MAX_ITER", "FLOAT_EPSILON",
           "Undefined"]

# Settings
TOLERANCE = 5e-8
# TODO: MAX_ITER is heavily algorithm-dependent, so we have to think about if
# we should include a package-wide maximum at all.
MAX_ITER = 10000
#mu_zero = 5e-8

FLOAT_EPSILON = np.finfo(float).eps


class UndefinedType(object):

    def __eq__(self, other):
        if isinstance(other, UndefinedType):
            return True
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __le__(self, other):
        return self.__eq__(other)

    def __ge__(self, other):
        return self.__eq__(other)

    def __lt__(self, other):
        return False  # Same behaviour as None.

    def __gt__(self, other):
        return False  # Same behaviour as None.

    def __cmp__(self, other):
        if self.__eq__(other):
            return 0
        else:
            return -1  # Same behaviour as None.

    def __str__(self):
        return "Undefined"

    def __repr__(self):
        return "Undefined"

    def __setattr__(self, name, value):
        if hasattr(self, name):
            raise AttributeError("'UndefinedType' object attribute '%s' is " \
                                 "read-only." % (name,))
        else:
            raise AttributeError("'UndefinedType' object has no attribute " \
                                 "'%s'." % (name,))

    def __getattr__(self, name):
        if hasattr(self, name):
            return super(UndefinedType, self).__getattr__(name)
        else:
            raise AttributeError("'UndefinedType' object has no attribute " \
                                 "'%s'." % (name,))

    def __delattr__(self, name):
        if hasattr(self, name):
            raise AttributeError("'UndefinedType' object attribute '%s' is " \
                                 "read-only." % (name,))
        else:
            raise AttributeError("'UndefinedType' object has no attribute " \
                                 "'%s'." % (name,))

    # Make it unhashable. We can't e.g. allow Undefined to be the key in a
    # dictionary.
    __hash__ = None

Undefined = UndefinedType()