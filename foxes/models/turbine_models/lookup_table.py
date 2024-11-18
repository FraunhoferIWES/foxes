import numpy as np
import pandas as pd
import xarray as xr

from foxes.core import TurbineModel
from foxes.utils import PandasFileHelper
from foxes.config import config, get_path
import foxes.constants as FC


class LookupTable(TurbineModel):
    """
    Calculates the data by interpolation of
    lookup-table data

    Attributes
    ----------
    data_source: str or pandas.DataFrame
        The lookup-table data
    input_vars: list of str
        The foxes input variables
    output_vars: list of str
        The foxes output variables
    varmap: dict
        Mapping from foxes variable names
        to column names in the data_source

    :group: models.turbine_models

    """

    def __init__(
        self,
        data_source,
        input_vars,
        output_vars,
        varmap={},
        pd_file_read_pars={},
        xr_interp_args={},
        interpn_args={},
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        data_source: str or pandas.DataFrame
            The lookup-table data
        input_vars: list of str
            The foxes input variables
        output_vars: list of str
            The foxes output variables
        varmap: dict
            Mapping from foxes variable names
            to column names in the data_source
        pd_file_read_pars: dict
            Parameters for pandas file reading
        xr_interp_args: dict
            Parameters for xarray interpolation method
        interpn_args: dict
            Parameters for scipy intern or interp1d
        kwargs: dict, optional
            Additional parameters, added as default
            values if not in data

        """
        super().__init__()

        self.data_source = data_source
        self.input_vars = input_vars
        self.output_vars = output_vars
        self.varmap = varmap

        self._rpars = pd_file_read_pars
        self._xargs = xr_interp_args
        self._iargs = interpn_args
        self._data = None

        for v, d in kwargs.items():
            if v not in input_vars:
                raise KeyError(
                    f"{self.name}: Default input parameter '{v}' not in list of inputs {input_vars}"
                )
            setattr(self, v, d)

    def output_farm_vars(self, algo):
        """
        The variables which are being modified by the model.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        output_vars: list of str
            The output variable names

        """
        return self.output_vars

    def load_data(self, algo, verbosity=0):
        """
        Load and/or create all model data that is subject to chunking.

        Such data should not be stored under self, for memory reasons. The
        data returned here will automatically be chunked and then provided
        as part of the mdata object during calculations.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        verbosity: int
            The verbosity level, 0 = silent

        Returns
        -------
        idata: dict
            The dict has exactly two entries: `data_vars`,
            a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and `coords`, a dict with entries `dim_name_str -> dim_array`

        """
        if self._data is None:
            if isinstance(self.data_source, pd.DataFrame):
                data = self.data_source
            else:
                fpath = get_path(self.data_source)
                if verbosity > 0:
                    print(f"{self.name}: Reading file {fpath}")
                data = PandasFileHelper.read_file(fpath, **self._rpars)

            if verbosity > 0:
                print(f"{self.name}: Preparing interpolation data")
            if len(self.varmap):
                data = data.rename(columns={c: v for v, c in self.varmap.items()})
            data = data[self.input_vars + self.output_vars]
            data.sort_values(by=self.input_vars, inplace=True)
            coords = {
                v: np.asarray(data[v].unique(), dtype=config.dtype_double)
                for v in self.input_vars
            }

            dvars = {}
            for oname in self.output_vars:
                pivot_matrix = data.pivot_table(index=self.input_vars, values=[oname])
                dvars[oname] = (
                    self.input_vars,
                    pivot_matrix.to_numpy(config.dtype_double).reshape(
                        pivot_matrix.index.levshape
                    ),
                )

            self._data = xr.Dataset(coords=coords, data_vars=dvars)

            if verbosity > 1:
                print()
                print(self._data)
                print()

        return super().load_data(algo, verbosity)

    def calculate(self, algo, mdata, fdata, st_sel):
        """
        The main model calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        mdata: foxes.core.MData
            The model data
        fdata: foxes.core.FData
            The farm data
        st_sel: slice or numpy.ndarray of bool
            The state-turbine selection,
            for shape: (n_states, n_turbines)

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_turbines)

        """
        data = {
            v: self.get_data(
                v,
                FC.STATE_TURBINE,
                lookup="fs",
                fdata=fdata,
                upcast=False,
                selection=st_sel,
            )
            for v in self.input_vars
        }

        dims = {
            v: ("_z") if len(data[v].shape) == 1 else ("_z", "_u")
            for v in self.input_vars
        }
        indata = {
            v: xr.DataArray(
                data[v],
                dims=dims[v],
            )
            for v in self.input_vars
        }
        del data, dims

        iargs = dict(bounds_error=True)
        iargs.update(self._iargs)
        try:
            odata = self._data.interp(**indata, kwargs=iargs, **self._xargs)
        except ValueError as e:
            print("\nBOUNDS ERROR", self.name)
            print("Variables:", list(indata.keys()))
            print(
                "DATA min/max:",
                [float(np.min(self._data[v].to_numpy())) for v in indata.keys()],
                [float(np.max(self._data[v].to_numpy())) for v in indata.keys()],
            )
            print(
                "EVAL min/max:",
                [float(np.min(d.to_numpy())) for d in indata.values()],
                [float(np.max(d.to_numpy())) for d in indata.values()],
            )
            print(
                "\nMaybe you want to try the options 'bounds_error=False, fill_value=None'? This will extrapolate the data.\n"
            )
            raise e

        out = {}
        for v in self.output_vars:
            out[v] = fdata[v]
            out[v][st_sel] = odata[v]

        return out
