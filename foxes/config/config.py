import numpy as np

from foxes.utils.dict import Dict
from foxes.constants import DTYPE, ITYPE

class Config(Dict):
    """
    Container for configurational data
    
    :group: foxes.config
    """
    def __init__(self):
        """Constructor"""
        super().__init__(
            {
                DTYPE: np.float64,
                ITYPE: np.int64,
            },
            name="config",
        )
    
    @property
    def dtype_double(self):
        """The default double data type"""
        return self.get_item(DTYPE)
    
    @property
    def dtype_int(self):
        """The default int data type"""
        return self.get_item(ITYPE)

config = Config()
"""Foxes configurational data object
:group: foxes.config
"""
