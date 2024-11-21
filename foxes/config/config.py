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
                FC.WORK_DIR: Path("."),
                FC.OUT_DIR: Path("."),
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
    def work_dir(self):
        """
        The foxes working directory

        Returns
        -------
        pth: pathlib.Path
            Path to the foxes working directory

        """
        pth = self.get_item(FC.WORK_DIR)
        if not isinstance(pth, Path):
            self[FC.WORK_DIR] = Path(pth)
        return self[FC.WORK_DIR]

    @property
    def out_dir(self):
        """
        The default output directory

        Returns
        -------
        pth: pathlib.Path
            Path to the default output directory

        """
        return get_path(self.get_item(FC.OUT_DIR))


config = Config()
"""Foxes configurational data object
:group: foxes.config
"""


def get_path(pth):
    """
    Gets path object, respecting the configurations
    work directory

    Parameters
    ----------
    pth: str or pathlib.Path
        The path, optionally relative

    Returns
    -------
    out: pathlib.Path
        The path, absolute or relative to working directory
        from config

    :group: foxes.config

    """
    if not isinstance(pth, Path):
        pth = Path(pth)
    return pth if pth.is_absolute() else config.work_dir / pth


def get_output_path(pth):
    """
    Gets path object, respecting the configurations
    output directory

    Parameters
    ----------
    pth: str or pathlib.Path
        The path, optionally relative

    Returns
    -------
    out: pathlib.Path
        The path, absolute or relative to output directory
        from config

    :group: foxes.config

    """
    if not isinstance(pth, Path):
        pth = Path(pth)
    return pth if pth.is_absolute() else config.out_dir / pth
