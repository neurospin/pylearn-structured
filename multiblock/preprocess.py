# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:43:21 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD Style
"""

__all__ = ['Preprocess', 'Center', 'Scale']

import abc
from multiblock.utils import TOLERANCE


class Preprocess(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, which, *kwargs):
        if not isinstance(which, (tuple, list)):
            raise ValueError('Argument "which" must be a list or tuple of ' \
                             'bools')
        for i in xrange(len(which)):
            if not isinstance(which[i], bool):
                raise ValueError('Argument "which" must be a list or tuple ' \
                                 'of bools')
        self.which = which
        self.params = kwargs

    @abc.abstractmethod
    def process(self, *X):
        raise NotImplementedError('Abstract method "process" must be '\
                                  'specialised!')

    @abc.abstractmethod
    def revert(self, *X):
        raise NotImplementedError('Abstract method "revert" must be '\
                                  'specialised!')


class Center(Preprocess):

    def __init__(self, which):
        Preprocess.__init__(self, which)
        self.means = None

    def process(self, *X):
        """ Centers the numpy array(s) in X

        Arguments
        ---------
        X : The matrices to center

        Returns
        -------
        Centered X
        """

        if self.means != None:
            for i in xrange(len(X)):
                if self.which[i]:
                    X[i] = X[i] - self.means[i]
        else:
            self.means = []
            for i in xrange(len(X)):
                if self.which[i]:
                    mean = X[i].mean(axis=0)
                    X[i] = X[i] - mean
                    self.means.append(mean)
                else:
                    self.means.append([0] * X[i].shape[1])

        return X

    def revert(self, *X):
        """ Un-centers the previously centered numpy array(s) in X

        Arguments
        ---------
        X : The matrices to center

        Returns
        -------
        Un-centered X
        """

        if self.means == None:
            raise ValueError('The method "process" must be applied before ' \
                             '"revert" can be applied.')

        for i in xrange(len(X)):
            if self.which[i]:
                X[i] = X[i] + self.means[i]

        return X


class Scale(Preprocess):

    def __init__(self, which, *kwargs):
        Preprocess.__init__(self, which)
        self.centered = kwargs.pop('centered', True)
        if not isinstance(self.centered, (tuple, list)):
            self.centered = [self.centered] * len(self.which)
        self.stds = None

    def process(self, *X):
        """ Scales the numpy arrays in arrays to standard deviation 1

        Arguments
        ---------
        X : The matrices to scale

        Returns
        -------
        Scaled X
        """

        if self.stds != None:
            for i in xrange(len(X)):
                if self.which[i]:
                    X[i] = X[i] / self.stds[i]
        else:
            self.stds = []
            for i in xrange(len(X)):
                if self.which[i]:
                    ddof = 1 if self.centered[i] else 0
                    std = X[i].std(axis=0, ddof=ddof)
                    std[std < TOLERANCE] = 1.0
                    X[i] = X[i] / std
                    self.stds.append(std)
                else:
                    self.stds.append([1] * X[i].shape[1])

        return X

    def revert(self, *X):
        """ Un-scales the previously scaled numpy arrays in arrays to standard
        deviation 1

        Arguments
        ---------
        X : The matrices to un-scale

        Returns
        -------
        Un-scaled X
        """

        if self.stds == None:
            raise ValueError('The method "process" must be applied before ' \
                             '"revert" can be applied.')

        for i in xrange(len(X)):
            if self.which[i]:
                X[i] = X[i] * self.stds[i]

        return X