from typing import cast
from typing import Any

import numpy as np
from pathlib import Path
from sys import version_info
from typing import Optional

from foxes.utils.dict import Dict
from foxes.utils.load import import_module
import foxes.constants as FC


class Config(Dict):
    """
    Container for configurational data

    :group: foxes.config
    """

    def __init__(self) -> None:
        """Constructor"""
        super().__init__(
            {
                FC.DTYPE: np.float64,
                FC.ITYPE: np.int64,
                FC.WORK_DIR: Path("."),
                FC.INPUT_DIR: None,
                FC.OUTPUT_DIR: None,
                FC.NC_ENGINE: "h5netcdf",
            },
            name="config",
        )
        self.__utmn: Optional[int] = None
        self.__utml: Optional[str] = None

        # special treat for Python 3.8:
        if version_info[0] == 3 and version_info[1] == 8:
            self["nc_engine"] = None

    def __setitem__(self, key: str, value: Any) -> None:
        if key == FC.UTM_ZONE:
            raise KeyError(
                "Direct setting of UTM zone is not allowed. "
                "Use config.set_utm_zone(...) instead."
            )
        super().__setitem__(key, value)

    @property
    def dtype_double(self) -> type:
        """
        The default double data type

        Returns
        -------
        dtp
            The default double data type

        """
        return cast(type, self.get_item(FC.DTYPE))

    @property
    def dtype_int(self) -> type:
        """
        The default int data type

        Returns
        -------
        dtp
            The default integer data type

        """
        return cast(type, self.get_item(FC.ITYPE))

    @property
    def work_dir(self) -> Path:
        """
        The foxes working directory

        Returns
        -------
        pth
            Path to the foxes working directory

        """
        pth = self.get_item(FC.WORK_DIR)
        if self[FC.WORK_DIR] is None:
            self[FC.WORK_DIR] = Path(".")
        elif not isinstance(pth, Path):
            self[FC.WORK_DIR] = Path(pth)
        return cast(Path, self[FC.WORK_DIR])

    @property
    def input_dir(self) -> Path:
        """
        The input base directory

        Returns
        -------
        pth
            Path to the input base directory

        """
        if self[FC.INPUT_DIR] is None:
            return self.work_dir
        else:
            pth = self.get_item(FC.INPUT_DIR)
            if not isinstance(pth, Path):
                self[FC.INPUT_DIR] = Path(pth)
            return cast(Path, self[FC.INPUT_DIR])

    @property
    def output_dir(self) -> Path:
        """
        The default output directory

        Returns
        -------
        pth
            Path to the default output directory

        """
        if self[FC.OUTPUT_DIR] is None:
            return self.work_dir
        else:
            pth = self.get_item(FC.OUTPUT_DIR)
            if not isinstance(pth, Path):
                self[FC.OUTPUT_DIR] = Path(pth)
            return cast(Path, self[FC.OUTPUT_DIR])

    @property
    def nc_engine(self) -> str | None:
        """
        The NetCDF engine

        Returns
        -------
        nce
            The NetCDF engine

        """
        nce = self[FC.NC_ENGINE]
        if nce == "netcdf4":
            import_module("netCDF4")
        elif nce is not None:
            import_module(nce)
        return cast(str | None, nce)

    @property
    def utm_zone_set(self) -> bool:
        """
        Whether the UTM zone is set

        Returns
        -------
        uzs
            True if both UTM zone number and letter are set

        """
        return self.__utmn is not None and self.__utml is not None

    @property
    def utm_zone(self) -> tuple[int, str]:
        """
        The UTM zone (number, letter) tuple

        Returns
        -------
        zn
            The UTM zone number
        zl
            The UTM zone letter

        """
        assert self.utm_zone_set, "UTM zone has not been set"
        assert self.__utmn is not None
        assert self.__utml is not None
        return self.__utmn, self.__utml

    def set_utm_zone(self, number: int, letter: str) -> None:
        """
        Set the UTM zone

        Parameters
        ----------
        number
            The UTM zone number
        letter
            The UTM zone letter

        """
        if self.utm_zone_set:
            if self.utm_zone != (number, letter):
                raise ValueError(
                    f"UTM zone already set to {self.utm_zone}, "
                    f"cannot set to {(number, letter)}"
                )
        else:
            self.__utmn = number
            self.__utml = letter
            super().__setitem__(FC.UTM_ZONE, (number, letter))


config = Config()
"""Foxes configurational data object
:group: foxes.config
"""


def get_path(pth: str | Path, base: Path) -> Path:
    """
    Gets path object, respecting the base directory

    Parameters
    ----------
    pth
        The path, optionally relative to base
    base
        The base directory

    Returns
    -------
    out
        The path, absolute or relative to base directory

    :group: foxes.config

    """
    if not isinstance(pth, Path):
        pth = Path(pth)
    if pth.is_absolute():
        return pth.expanduser()
    else:
        return (base / pth).expanduser()


def get_input_path(pth: str | Path) -> Path:
    """
    Gets path object, respecting the configurations
    input directory

    Parameters
    ----------
    pth
        The path, optionally relative

    Returns
    -------
    out
        The path, absolute or relative to input directory
        from config

    :group: foxes.config

    """
    return get_path(pth, base=config.input_dir)


def get_output_path(pth: str | Path) -> Path:
    """
    Gets path object, respecting the configurations
    output directory

    Parameters
    ----------
    pth
        The path, optionally relative

    Returns
    -------
    out
        The path, absolute or relative to output directory
        from config

    :group: foxes.config

    """
    return get_path(pth, base=config.output_dir)
