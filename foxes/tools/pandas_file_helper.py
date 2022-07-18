import pandas as pd
import xarray
from copy import deepcopy

import foxes.variables as FV


class PandasFileHelper:
    """
    This class helps reading and writing data
    to files via pandas.

    Attributes
    ----------
    DEFAULT_READING_PARAMETERS : dict
        Default parameters for file reading
        for the supported file formats
    DEFAULT_WRITING_PARAMETERS : dict
        Default parameters for file writing
        for the supported file formats
    DATA_FILE_FORMAT : list:str
        The supported file formats for data export
    DEFAULT_FORMAT_DICT : dict
        Default column formatting

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
        "h5": {"key": "flappy", "mode": "w"},
        "nc": {},
    }

    DEFAULT_FORMAT_DICT = {
        str(FV.__dict__[v]): "{:.4f}" for v in FV.__dict__.keys() if v[0] != "_"
    }
    DEFAULT_FORMAT_DICT["weight"] = "{:.10e}"

    DATA_FILE_FORMATS = list(DEFAULT_READING_PARAMETERS.keys())

    @classmethod
    def read_file(cls, file_path, **kwargs):
        """
        Helper for reading data according to file ending.

        Parameters
        ----------
        file_path : str
            The path to the file
        **kwargs : dict, optional
            Parameters forwarded to the pandas reading method.

        Returns
        -------
        pandas.DataFrame :
            The data

        """

        fname = str(file_path)
        L = len(fname)
        f = None
        for fmt in cls.DATA_FILE_FORMATS:

            l = len(fmt)
            if fname[L - l :] == fmt:

                if fmt[:3] == "csv":
                    f = pd.read_csv

                elif fmt == "h5":
                    f = pd.read_hdf

                elif fmt == "nc":
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
        data : pandas.DataFrame
            The data
        file_path : str
            The path to the file
        format_dict : dict
            Dictionary with format entries for
            columns, e.g. '{:.4f}'
        **kwargs : dict, optional
            Parameters forwarded to the pandas writing method.

        """

        fdict = cls.DEFAULT_FORMAT_DICT
        fdict.update(format_dict)

        out = pd.DataFrame(index=data.index)
        for c in data.columns:
            if c in fdict:
                out[c] = data[c].map(
                    lambda x: fdict[c].format(x) if not pd.isna(x) else x
                )
            else:
                out[c] = data[c]

        L = len(file_path)
        f = None
        for fmt in cls.DATA_FILE_FORMATS:

            l = len(fmt)
            if file_path[L - l :] == fmt:

                if fmt[:3] == "csv":
                    f = out.to_csv

                elif fmt == "h5":
                    f = out.to_hdf

                elif fmt == "nc":
                    f = out.to_netcdf

                if f is not None:

                    pars = cls.DEFAULT_WRITING_PARAMETERS[fmt]
                    pars.update(kwargs)

                    f(file_path, **pars)

                    return

        raise KeyError(
            f"Unknown file format '{file_path}'. Supported formats: {cls.DATA_FILE_FORMATS}"
        )
