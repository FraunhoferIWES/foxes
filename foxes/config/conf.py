import numpy as np
from pathlib import Path
from sys import version_info

from foxes.utils.dict import Dict
from foxes.utils.load import import_module
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
                FC.INPUT_DIR: None,
                FC.OUTPUT_DIR: None,
                FC.NC_ENGINE: "netcdf4",
            },
            name="config",
        )
        self.__utmn = None
        self.__utml = None

        # special treat for Python 3.8:
        if version_info[0] == 3 and version_info[1] == 8:
            self["nc_engine"] = None

    def __setitem__(self, key, value):
        if key == FC.UTM_ZONE:
            raise KeyError(
                "Direct setting of UTM zone is not allowed. "
                "Use config.set_utm_zone(...) instead."
            )
        super().__setitem__(key, value)

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
        if self[FC.WORK_DIR] is None:
            self[FC.WORK_DIR] = Path(".")
        elif not isinstance(pth, Path):
            self[FC.WORK_DIR] = Path(pth)
        return self[FC.WORK_DIR]

    @property
    def input_dir(self):
        """
        The input base directory

        Returns
        -------
        pth: pathlib.Path
            Path to the input base directory

        """
        if self[FC.INPUT_DIR] is None:
            return self.work_dir
        else:
            pth = self.get_item(FC.INPUT_DIR)
            if not isinstance(pth, Path):
                self[FC.INPUT_DIR] = Path(pth)
            return self[FC.INPUT_DIR]

    @property
    def output_dir(self):
        """
        The default output directory

        Returns
        -------
        pth: pathlib.Path
            Path to the default output directory

        """
        if self[FC.OUTPUT_DIR] is None:
            return self.work_dir
        else:
            pth = self.get_item(FC.OUTPUT_DIR)
            if not isinstance(pth, Path):
                self[FC.OUTPUT_DIR] = Path(pth)
            return self[FC.OUTPUT_DIR]

    @property
    def nc_engine(self):
        """
        The NetCDF engine

        Returns
        -------
        nce: str
            The NetCDF engine

        """
        nce = self[FC.NC_ENGINE]
        if nce == "netcdf4":
            import_module("netCDF4")
        elif nce is not None:
            import_module(nce)
        return nce

    @property
    def utm_zone_set(self):
        """
        Whether the UTM zone is set

        Returns
        -------
        uzs: bool
            True if both UTM zone number and letter are set

        """
        return self.__utmn is not None and self.__utml is not None

    @property
    def utm_zone(self):
        """
        The UTM zone (number, letter) tuple

        Returns
        -------
        zn: int
            The UTM zone number
        zl: str
            The UTM zone letter

        """
        assert self.utm_zone_set, "UTM zone has not been set"
        return self.__utmn, self.__utml

    def set_utm_zone(self, number, letter):
        """
        Set the UTM zone

        Parameters
        ----------
        number: int
            The UTM zone number
        letter: str
            The UTM zone letter
        verbosity: int
            The verbosity level, 0 = silent

        """
        assert not self.utm_zone_set, f"UTM zone already set to {self.utm_zone}"
        self.__utmn = number
        self.__utml = letter
        super().__setitem__(FC.UTM_ZONE, (number, letter))


config = Config()
"""Foxes configurational data object
:group: foxes.config
"""


def get_path(pth, base):
    """
    Gets path object, respecting the base directory

    Parameters
    ----------
    pth: str or pathlib.Path
        The path, optionally relative to base
    base: pathlib.Path
        The base directory

    Returns
    -------
    out: pathlib.Path
        The path, absolute or relative to base directory

    :group: foxes.config

    """
    if not isinstance(pth, Path):
        pth = Path(pth)
    if pth.is_absolute():
        return pth.expanduser()
    else:
        return (base / pth).expanduser()


def get_input_path(pth):
    """
    Gets path object, respecting the configurations
    input directory

    Parameters
    ----------
    pth: str or pathlib.Path
        The path, optionally relative

    Returns
    -------
    out: pathlib.Path
        The path, absolute or relative to input directory
        from config

    :group: foxes.config

    """
    return get_path(pth, base=config.input_dir)


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
    return get_path(pth, base=config.output_dir)
