# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 13:52:23 2014

@author:  edouard.duchesnay@cea.fr
@license: BSD-3-Clause
"""
import numpy as np

def class_weight_to_sample_weight(class_weight, y):
    """Estimate class weights for unbalanced datasets.

    Parameters
    ----------
    class_weight : dict, 'auto' or None
        If 'auto', class weights will be given inverse proportional
        to the frequency of the class in the data.
        If a dictionary is given, keys are classes and values
        are corresponding class weights.
        If None is given, the class weights will be uniform.

    y : array-like, shape (n_samples,)
        Array of original class labels per sample;

    Returns
    -------
    weight_vect : ndarray, shape (n_samples,)
        Array with weight_vect[i] the weight for i-th sample

    Example
    -------
    >>> y = np.r_[np.ones(1), np.ones(2)*2, np.zeros(2)]
    >>> w = class_weight_to_sample_weight("auto", y)
    >>> print ["%i:%.2f" % (l, np.sum(w[y==l])) for l in np.unique(y)]
    ['0:1.50', '1:1.50', '2:1.50']
    >>> print class_weight_to_sample_weight({1:10, 2:100, 0:1}, y)
    [  10.  100.  100.    1.    1.]
    """
    # Import error caused by circular imports.
    #from ..preprocessing import LabelEncoder
    classes = np.unique(y)
    if class_weight is None or len(class_weight) == 0:
        # uniform class weights
        weight = np.ones(y.shape[0], dtype=np.float64)
    elif class_weight == 'auto':
        # Find the weight of each class as present in y.
        # inversely proportional to the number of samples in the class
        count_inv = 1. / np.bincount(y.astype(int).ravel())
        weight = count_inv[np.searchsorted(classes, y)] / np.mean(count_inv)
    else:
        # user-defined dictionary
        weight = np.zeros(y.shape[0], dtype=np.float64)
        if not isinstance(class_weight, dict):
            raise ValueError("class_weight must be dict, 'auto', or None,"
                             " got: %r" % class_weight)
        for c in class_weight:
            mask = y.ravel() == c
            if mask.sum() == 0:
                raise ValueError("Class label %d not present." % c)
            else:
                weight[mask] = class_weight[c]
    return weight


def check_labels(y):
    """ensure binary classification with 0, 1 labels"""
    nlevels = 2
    classes = np.unique(y)
    if len(classes) > nlevels:
        raise ValueError("Multinomial classification with more " \
                        "than %i labels is not possible" % nlevels)
    classes_recoded = np.arange(len(classes))
    if np.all(classes_recoded == classes):
        return y
    # Ensure labels are 0, 1
    y_recoded = np.zeros(y.shape, dtype=np.float64)
    for i in xrange(len(classes)):
        y_recoded[y == classes[i]] = classes_recoded[i]
    return y_recoded