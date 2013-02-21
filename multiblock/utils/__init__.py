# -*- coding: utf-8 -*-

from .utils import dot
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
from .utils import rand
from .utils import zeros

import testing

__all__ = ['dot', 'norm', 'norm1', 'norm0', 'normI', 'make_list', 'sign',
           'cov', 'corr', 'TOLERANCE', 'MAX_ITER', 'copy',
           'sstot', 'ssvar', 'rand', 'zeros',
           'testing']
