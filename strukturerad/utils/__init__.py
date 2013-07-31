# -*- coding: utf-8 -*-

from .utils import TOLERANCE
from .utils import MAX_ITER
from .utils import DEBUG

from .utils import make_list
from .utils import direct
from .utils import optimal_shrinkage
from .utils import delete_sparse_csr_row
from .utils import AnonymousClass
from .utils import debug
from .utils import warning

from .check_arrays import check_arrays

__all__ = ['TOLERANCE', 'MAX_ITER', 'DEBUG', 'make_list', 'direct',
           'optimal_shrinkage', 'delete_sparse_csr_row', 'AnonymousClass',

           'debug', 'warning',

           'check_arrays']