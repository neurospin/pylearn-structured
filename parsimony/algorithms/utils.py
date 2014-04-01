# -*- coding: utf-8 -*-
"""
The :mod:`parsimony.algorithms.utils` module includes algorithm specific code.

Created on Thu Mar 31 17:25:01 2014

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import collections

from parsimony.utils import Enum

__all__ = ["AlgorithmInfo",
           "Info"]

Info = Enum("Info", "ok", "t", "f", "gap", "mu", "converged")


class AlgorithmInfo(collections.MutableMapping):
    """Used as input and output of algorithm information.

    This class is essentially a dict, but it only allows a set of keys defined
    at initialisation.

    Parameters
    ----------
    keys : A sequence of allowed keys. The set of keys that are allowed.
    """
    def __init__(self, *keys):
        if (len(keys) == 1 and isinstance(keys[0], collections.Sequence) \
                and len(keys[0]) == 0) or len(keys) == 0:
            self.__keys = list()

        elif (len(keys) == 1 and isinstance(keys[0], collections.Sequence) \
                and len(keys[0]) == 1):
            self.__keys = list(keys[0])

        else:
            self.__keys = list(keys)

        self.__dict = dict()

    def add_key(self, key):
        if key not in self.__keys:
            self.__keys.append(key)

    def remove_key(self, key):
        if key in self.__keys:
            self.__keys.remove(key)

            # Key no longer valid. Remove from dictionary if present.
            if key in self.__dict:
                del self.__dict[key]

    def allows(self, key):
        return key in self.__keys

    def allowed_keys(self):
        return self.__keys[:]

    def __len__(self):
        return len(self.__dict)

    def __getitem__(self, key):
        if key not in self.__keys:
            raise KeyError("'%s' is not an allowed key." % (key,))

        return self.__dict[key]

    def __setitem__(self, key, value):
        if key not in self.__keys:
            raise KeyError("'%s' is not an allowed key." % (key,))

        self.__dict[key] = value

    def __delitem__(self, key):
        if key not in self.__keys:
            raise KeyError("'%s' is not an allowed key." % (key,))

        del self.__dict[key]

    def __contains__(self, key):
        if key not in self.__keys:
            raise KeyError("'%s' is not an allowed key." % (key,))

        return key in self.__dict

    def __iter__(self):
        return iter(self.__dict)

    def clear(self):
        self.__dict.clear()

    def copy(self):
        info = AlgorithmInfo(self.__keys[:])
        info.__dict = self.__dict.copy()

        return info

    @classmethod
    def fromkeys(cls, keys, value=None):
        info = cls(keys)
        info.__dict = dict.fromkeys(keys, value)

        return info

    def get(self, key, default=None):
        if key not in self.__keys:
            raise KeyError("'%s' is not an allowed key." % (key,))

        if key in self.__dict:
            return self.__dict[key]
        else:
            return default

    def items(self):
        return self.__dict.items()

    def iteritems(self):
        return self.__dict.iteritems()

    def iterkeys(self):
        return self.__dict.iterkeys()

    def itervalues(self):
        return self.__dict.itervalues()

    def keys(self):
        return self.__dict.keys()

    def pop(self, *args):
        if len(args) == 0:
            raise TypeError("pop expected at least 1 arguments, got 0")
        if len(args) > 2:
            raise TypeError("pop expected at most 2 arguments, got %d" \
                    % (len(args),))

        if len(args) >= 1:
            key = args[0]
            default_given = False
        if len(args) >= 2:
            default = args[1]
            default_given = True

        if key not in self.__keys:
            raise KeyError("'%s' is not an allowed key." % (key,))

        if key not in self.__dict:
            if default_given:
                return default
            else:
                raise KeyError(str(key))
        else:
            return self.__dict[key]

    def popitem(self):
        return self.__dict.popitem()

    def setdefault(self, key, default=None):
        if key not in self.__keys:
            raise KeyError("'%s' is not an allowed key." % (key,))

        if key in self.__dict:
            return self.__dict[key]
        else:
            self.__dict[key] = default
            return default

    def update(self, *args, **kwargs):
        info = dict()
        info.update(*args, **kwargs)
        for key in info.keys():
            if key not in self.__keys:
                raise KeyError("'%s' is not an allowed key." % (key,))

        self.__dict.update(info)

    def values(self):
        return self.__dict.values()

    def viewitems(self):
        return self.__dict.viewitems()

    def viewkeys(self):
        return self.__dict.viewkeys()

    def viewvalues(self):
        return self.__dict.viewvalues()

    def __format__(self, *args, **kwargs):
        return self.__dict.__format__(*args, **kwargs)

    def __eq__(self, other):
        if not isinstance(other, AlgorithmInfo):
            return False
        return self.__keys == other.__keys and self.__dict == other.__dict

    def __ge__(self, other):
        return self.__keys == other.__keys and self.__dict >= other.__dict

    def __gt__(self, other):
        return self.__keys == other.__keys and self.__dict > other.__dict

    def __hash__(self):
        return hash(self.__keys) | hash(self.__dict)

    def __le__(self, other):
        return self.__keys == other.__keys and self.__dict <= other.__dict

    def __lt__(self, other):
        return self.__keys == other.__keys and self.__dict < other.__dict

    def __ne__(self, other):
        keys_eq = self.__keys == other.__keys
        if not keys_eq:
            return False
        else:
            return self.__dict != other.__dict

    def __cmp__(self, other):
        keys_cmp = cmp(self.__keys, other.__keys)
        if keys_cmp != 0:
            return keys_cmp
        else:
            return cmp(self.__dict, other.__dict)

    def __repr__(self):
        return "AlgorithmInfo(%s).update(%s)" \
                % (self.__keys.__repr__(), self.__dict.__repr__())

    def __str__(self):
        return "Keys: %s. Dict: %s." % (str(self.__keys), str(self.__dict))