import numpy as np
from pathlib import Path

from foxes.utils.dict import Dict
import foxes.constants as FC


class Config(Dict):
    """
    Container for configurational data

    :group: foxes.config
    """

    def __init__(self):
        """Constructor"""
        super().__init__(
            {
                FC.DTYPE: np.float64,
                FC.ITYPE: np.int64,
                FC.CWD_DIR: Path("."),
            },
            name="config",
        )

    @property
    def dtype_double(self):
        """
        The default double data type
        
        Returns
        -------
        dtp: type
            The default double data type
            
        """
        return self.get_item(FC.DTYPE)

    @property
    def dtype_int(self):
        """
        The default int data type
        
        Returns
        -------
        dtp: type
            The default integer data type
            
        """
        return self.get_item(FC.ITYPE)

    @property
    def cwd(self):
        """
        The current working directory
        
        Returns
        -------
        pth: pathlib.Path
            Path to the current working directory
            
        """
        pth = self.get_item(FC.CWD_DIR)
        if not isinstance(pth, Path):
            self[FC.CWD_DIR] = Path(pth)
        return self[FC.CWD_DIR]

config = Config()
"""Foxes configurational data object
:group: foxes.config
"""
