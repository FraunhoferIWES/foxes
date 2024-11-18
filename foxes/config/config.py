import numpy as np

from foxes.utils.dict import Dict
from foxes.constants import DTYPE, ITYPE

config = Dict(
    {
        DTYPE: np.float64,
        ITYPE: np.int64,
    },
    name="config",
)
"""Foxes configurational data
:group: foxes.config
"""
