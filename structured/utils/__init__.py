# -*- coding: utf-8 -*-

from .utils import norm
from .utils import norm1
from .utils import norm0
from .utils import normI
from .utils import make_list
from .utils import sign
from .utils import cov
from .utils import corr
from .utils import TOLERANCE
from .utils import MAX_ITER
from .utils import copy
from .utils import sstot
from .utils import ssvar
from .utils import sqrt
from .utils import rand
from .utils import zeros
from .utils import direct
from .utils import debug
from .utils import warning
from .utils import _DEBUG
from .utils import optimal_shrinkage
from .utils import delete_sparse_csr_row
from .utils import AnonymousClass
from .check_arrays import check_arrays

import testing

__all__ = ['norm', 'norm1', 'norm0', 'normI', 'make_list', 'sign',
           'cov', 'corr', 'TOLERANCE', 'MAX_ITER', 'copy',
           'sstot', 'ssvar', 'sqrt', 'rand', 'zeros',
           'testing', 'direct', 'debug', 'warning', '_DEBUG',
           'optimal_shrinkage', 'delete_sparse_csr_row', 'AnonymousClass',
           'check_arrays']