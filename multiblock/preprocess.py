# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 14:43:21 2013

@author:  Tommy Löfstedt
@email:   tommy.loefstedt@cea.fr
@license: BSD Style
"""

__all__ = ['Preprocess', 'Center', 'Scale', 'Mask']

import abc
import utils
import numpy as np


class PreprocessQueue(object):
    def __init__(self, pq, X=None):
        """ Preprocess queue.

        Arguments
        ---------
        pq : Either a PreprocessQueue, a list of Preprocess instances or an
             instance of Preprocess.

        X  : An optional matrix to use for initialising the preprocessing.
        """

        super(PreprocessQueue, self).__init__()

        if pq == None:
            self.queue = []
        elif isinstance(pq, (tuple, list)):
            self.queue = []
            for p in pq:
                if not isinstance(p, Preprocess):
                    raise ValueError('If argument "preprocess" is a list, it '\
                                     'must be a list of Preprocess instances')
                self.queue.append(p.__class__(**p.params))
        elif isinstance(pq, PreprocessQueue):
            self.queue = []
            for p in pq.queue:
                self.queue.append(p.__class__(**p.params))
        elif isinstance(pq, Preprocess):
            self.queue = [pq.__class__(**pq.params)]
        else:
            raise ValueError('Argument "pq" must either be a PreprocessQueue,'\
                             ' a list of Preprocess instances or an instance '\
                             'of Preprocess')

        # Run once to initialise
        if X != None:
            for p in self.queue:
                X = p.process(X)

    def push(self, p):
        """ Adds (pushes) a Preprocess instance to the end of the queue.
        """

        if not isinstance(p, Preprocess):
            raise ValueError('Argument must be an instance of "Preprocess"')
        self.queue.append(p)

    def pop(self):
        """ Removes (pops) the first Preprocess instance from the queue.
        """

        self.queue.pop(0)

    def process(self, X):
        for p in self.queue:
            X = p.process(X)
        return X

    def revert(self, X):
        for p in reversed(self.queue):
            X = p.revert(X)
        return X


class Preprocess(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, **kwargs):
        super(Preprocess, self).__init__()

        self.params = kwargs

    @abc.abstractmethod
    def process(self, X):
        raise NotImplementedError('Abstract method "process" must be '\
                                  'specialised!')

    @abc.abstractmethod
    def revert(self, X):
        raise NotImplementedError('Abstract method "revert" must be '\
                                  'specialised!')


class Center(Preprocess):

    def __init__(self, **kwargs):
#        Preprocess.__init__(self, **kwargs)
        super(Center, self).__init__(**kwargs)

        self.means = None

    def process(self, X):
        """ Centers the numpy array in X

        Arguments
        ---------
        X : The matrix to center

        Returns
        -------
        Centered X
        """

        if self.means == None:
            self.means = X.mean(axis=0)
        X = X - self.means

        return X

    def revert(self, X):
        """ Un-centers the previously centered numpy array in X

        Arguments
        ---------
        X : The matrix to center

        Returns
        -------
        Un-centered X
        """

        if self.means == None:
            raise ValueError('The method "process" must be applied before ' \
                             '"revert" can be applied.')
        X = X + self.means

        return X


class Scale(Preprocess):

    def __init__(self, **kwargs):
#        Preprocess.__init__(self, **kwargs)
        super(Scale, self).__init__(**kwargs)

        self.centered = kwargs.pop('centered', True)
        self.stds = None

    def process(self, X):
        """ Scales the numpy array in X to standard deviation 1

        Arguments
        ---------
        X : The matrix to scale

        Returns
        -------
        Scaled X
        """

        if self.stds == None:
            ddof = 1 if self.centered else 0
            self.stds = X.std(axis=0, ddof=ddof)
            self.stds[self.stds < utils.TOLERANCE] = 1.0

        X = X / self.stds

        return X

    def revert(self, X):
        """ Un-scales the previously scaled numpy array in X to standard
        deviation 1

        Arguments
        ---------
        X : The matrix to un-scale

        Returns
        -------
        Un-scaled X
        """

        if self.stds == None:
            raise ValueError('The method "process" must be applied before ' \
                             '"revert" can be applied.')

        X = X * self.stds

        return X


class Mask(Preprocess):
    """Applies a mask to the columns of a matrix.

    The columns are removed, and on revert the columns removed are set to zero.

    Arguments
    ---------
    mask : The mask to apply. Must be an integer vector where 0 means excluded.
    """

    def __init__(self, mask, **kwargs):

        super(Mask, self).__init__(**kwargs)

        self.mask = mask
        self.included = None
        self.excluded = None
        self.shape = None

    def process(self, X):

        # Transposed? Weights vectors are likely transposed.
        self.transposed = False
        if X.shape[0] == len(self.mask) and X.shape[1] != len(self.mask):
            X = X.T
            self.transposed = True

        if X.shape[1] != len(self.mask):
            raise ValueError('The matrix X does not fit the mask!')

        mask = np.array(self.mask, dtype=int)
        self.included = mask != 0
        self.excluded = mask == 0
        self.shape = X.shape
        X = X[:, self.included]

        if self.transposed:
            X = X.T

        return X

    def revert(self, X):

        # Transposed? Weights vectors are likely transposed.
        transposed = False
        axis = 0
        if X.shape[0] == len(self.mask) and X.shape[1] != len(self.mask):
            X = X.T
            transposed = True
            axis = 1

        if self.included == None:
            raise ValueError('The method "process" must be applied before ' \
                             '"revert" can be applied.')

        X_ = np.zeros((X.shape[axis], len(self.mask)), dtype=X.dtype)
        X_[:, self.included] = X

        if transposed:
            X_ = X_.T

        return X_