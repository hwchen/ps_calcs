# -*- coding: utf-8 -*-
"""
    growth library
    ~~~~~~~~~~~~~~

    The Growth library is a helper library for running product space
    calculationsin python with PANDAS and NumPy.

    :copyright: (c) 2018 by Alexander Simoes.
    :license: MIT, see LICENSE for more details.
"""
__version__ = '2.0.0'
name = "ps_calcs"

from .complexity import complexity
from .rca import rca
from .density import density
from .distance import distance
from .proximity import proximity
from .opportunity_gain import opportunity_gain
from .mhat import mhat
