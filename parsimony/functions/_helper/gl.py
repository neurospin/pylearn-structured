# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 14:26:06 2014

@author:  Tommy Löfstedt
@email:   lofstedt.tommy@gmail.com
@license: BSD 3-clause.
"""
import scipy.sparse as sparse

__all__ = ["A_from_groups"]


def A_from_groups(num_variables, groups, weights=None):
    """Generates the linear operator for the group lasso Nesterov function
    from the groups of variables.

    Parameters:
    ----------
    num_variables : Integer. The total number of variables.

    groups : A list of lists. The outer list represents the groups and the
            inner lists represent the variables in the groups. E.g. [[1, 2],
            [2, 3]] contains two groups ([1, 2] and [2, 3]) with variable 1 and
            2 in the first group and variables 2 and 3 in the second group.

    weights : A list. Weights put on the groups. Default is weight 1 for each
            group.
    """
    if weights is None:
        weights = [1.0] * len(groups)

    A = list()
    for g in xrange(len(groups)):
        Gi = groups[g]
        lenGi = len(Gi)
        Ag = sparse.lil_matrix((lenGi, num_variables))
        for i in xrange(lenGi):
            w = weights[g]
            Ag[i, Gi[i]] = w

        # Matrix operations are a lot faster when the sparse matrix is csr
        A.append(Ag.tocsr())

    return A