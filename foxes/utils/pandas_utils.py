import pandas as pd
from pathlib import Path
import xarray
from copy import deepcopy

import foxes.variables as FV


class PandasFileHelper:
    """
    This class helps reading and writing data
    to files via pandas.

    Attributes
    ----------
    DEFAULT_READING_PARAMETERS: dict
        Default parameters for file reading
        for the supported file formats
    DEFAULT_WRITING_PARAMETERS: dict
        Default parameters for file writing
        for the supported file formats
    DATA_FILE_FORMAT: list:str
        The supported file formats for data export
    DEFAULT_FORMAT_DICT: dict
        Default column formatting

    :group: utils

    """

    DEFAULT_READING_PARAMETERS = {
        "csv": {},
        "csv.gz": {},
        "csv.bz2": {},
        "csv.zip": {},
        "h5": {},
        "nc": {},
    }

    DEFAULT_WRITING_PARAMETERS = {
        "csv": {},
        "csv.gz": {},
        "csv.bz2": {},
        "csv.zip": {},
        "h5": {"key": "foxes", "mode": "w"},
        "nc": {},
    }

    DEFAULT_FORMAT_DICT = {
        FV.WD: "{:.3f}",
        FV.AMB_WD: "{:.3f}",
        FV.YAW: "{:.3f}",
        FV.AMB_YAW: "{:.3f}",
        FV.WS: "{:.4f}",
        FV.AMB_WS: "{:.4f}",
        FV.REWS: "{:.4f}",
        FV.AMB_REWS: "{:.4f}",
        FV.REWS2: "{:.4f}",
        FV.AMB_REWS2: "{:.4f}",
        FV.REWS3: "{:.4f}",
        FV.AMB_REWS3: "{:.4f}",
        FV.TI: "{:.6f}",
        FV.AMB_TI: "{:.6f}",
        FV.RHO: "{:.5f}",
        FV.AMB_RHO: "{:.5f}",
        FV.P: "{:.3f}",
        FV.AMB_P: "{:.3f}",
        FV.CT: "{:.6f}",
        FV.AMB_CT: "{:.6f}",
        FV.T: "{:.3f}",
        FV.AMB_T: "{:.3f}",
        FV.YLD: "{:.3f}",
        FV.AMB_YLD: "{:.3f}",
        FV.CAP: "{:.5f}",
        FV.AMB_CAP: "{:.5f}",
        FV.EFF: "{:.5f}",
    }

    DATA_FILE_FORMATS = list(DEFAULT_READING_PARAMETERS.keys())

    @classmethod
    def read_file(cls, file_path, **kwargs):
        """
        Helper for reading data according to file ending.

        Parameters
        ----------
        file_path: str
            The path to the file
        **kwargs: dict, optional
            Parameters forwarded to the pandas reading method.

        Returns
        -------
        pandas.DataFrame :
            The data

        """
        fpath = Path(file_path)
        fname = fpath.name
        sfx = ".".join(fname.split(".")[1:])
        f = None
        for fmt in cls.DATA_FILE_FORMATS:
            if sfx[:3] == "csv":
                f = pd.read_csv
            elif sfx == "h5":
                f = pd.read_hdf
            elif sfx == "nc":
                f = lambda fname, **pars: xarray.open_dataset(
                    fname, **pars
                ).to_dataframe()

            if f is not None:
                pars = deepcopy(cls.DEFAULT_READING_PARAMETERS[fmt])
                pars.update(kwargs)
                return f(file_path, **pars)

        raise KeyError(
            f"Unknown file format '{fname}'. Supported formats: {cls.DATA_FILE_FORMATS}"
        )

    @classmethod
    def write_file(cls, data, file_path, format_dict={}, **kwargs):
        """
        Helper for writing data according to file ending.

        Parameters
        ----------
        data: pandas.DataFrame
            The data
        file_path: str
            The path to the file
        format_dict: dict
            Dictionary with format entries for
            columns, e.g. '{:.4f}'
        **kwargs: dict, optional
            Parameters forwarded to the pandas writing method.

        """

        fdict = deepcopy(cls.DEFAULT_FORMAT_DICT)
        fdict.update(format_dict)

        out = pd.DataFrame(index=data.index, columns=data.columns)
        for c in data.columns:
            if c in fdict.keys():
                out[c] = data[c].map(
                    lambda x: fdict[c].format(x) if not pd.isna(x) else x
                )
            else:
                out[c] = data[c]

        fpath = Path(file_path)
        fname = fpath.name
        sfx = ".".join(fname.split(".")[1:])
        f = None
        for fmt in cls.DATA_FILE_FORMATS:
            if sfx[:3] == "csv":
                f = out.to_csv
            elif sfx == "h5":
                f = out.to_hdf
            elif sfx == "nc":
                f = out.to_netcdf

            if f is not None:
                pars = cls.DEFAULT_WRITING_PARAMETERS[fmt]
                pars.update(kwargs)

                f(file_path, **pars)

                return

        raise KeyError(
            f"Unknown file format '{file_path}'. Supported formats: {cls.DATA_FILE_FORMATS}"
        )
