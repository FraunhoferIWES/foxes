import numpy as np
import pandas as pd
from os import PathLike
from xarray import Dataset, open_dataset

from foxes.data import STATES
from foxes.utils import PandasFileHelper, weibull_weights
from foxes.config import config, get_input_path
import foxes.variables as FV
import foxes.constants as FC

from .states_table import StatesTable


class WeibullSectors(StatesTable):
    """
    States with wind speed from Weibull parameters
    from a NetCDF file

    Attributes
    ----------
    ws_bins: numpy.ndarray
        The wind speed bins, including
        lower and upper bounds, shape: (n_ws_bins+1,)
    var2ncvar: dict
        Mapping from variable names to variable names
        in the nc file
    sel: dict
        Subset selection via xr.Dataset.sel()
    isel: dict
        Subset selection via xr.Dataset.isel()
    RDICT: dict
        Default xarray file reading parameters

    :group: input.states

    """

    RDICT = {}

    def __init__(
        self,
        data_source,
        output_vars,
        ws_bins=None,
        var2ncvar={},
        sel=None,
        isel=None,
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        data_source: str or xarray.Dataset or pandas.DataFrame
            Either path to NetCDF or csv file or data
        output_vars: list of str
            The output variables
        ws_bins: list of float, optional
            The wind speed bins, including
            lower and upper bounds
        var2ncvar: dict
            Mapping from variable names to variable names
            in the nc file
        sel: dict, optional
            Subset selection via xr.Dataset.sel()
        isel: dict, optional
            Subset selection via xr.Dataset.isel()
        kwargs: dict, optional
            Additional arguments for the base class

        """
        super().__init__(data_source, output_vars, var2col={}, **kwargs)
        self.ws_bins = None if ws_bins is None else np.asarray(ws_bins)
        self.var2ncvar = var2ncvar
        self.sel = sel if sel is not None else {}
        self.isel = isel if isel is not None else {}

        if FV.WS not in self.ovars:
            raise ValueError(f"States '{self.name}': Expecting output variable '{FV.WS}', got {self.ovars}")
        for v in [FV.WEIBULL_A, FV.WEIBULL_k, FV.WEIGHT]:
            if v in self.ovars:
                raise ValueError(f"States '{self.name}': Cannot have '{v}' as output variable")

    def __repr__(self):
        return f"{type(self).__name__}(ws_bins={self._n_ws})"
    
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

        if isinstance(self.data_source, (str, PathLike)):
            self._data_source = get_input_path(self.data_source)
            if not self.data_source.is_file():
                if verbosity > 0:
                    print(
                        f"States '{self.name}': Reading static data '{self.data_source}' from context '{STATES}'"
                    )
                self._data_source = algo.dbook.get_file_path(
                    STATES, self.data_source.name, check_raw=False
                )
                if verbosity > 0:
                    print(f"Path: {self.data_source}")
            elif verbosity:
                print(f"States '{self.name}': Reading file {self.data_source}")
            rpars = dict(self.RDICT, **self.rpars)
            if self.data_source.suffix == ".nc":
                data = open_dataset(self.data_source, engine=config.nc_engine, **rpars)
            else:
                data = PandasFileHelper().read_file(self.data_source, **rpars).to_xarray()

        elif isinstance(self.data_source, Dataset):
            data = self.data_source
        
        elif isinstance(self.data_source, pd.DataFrame):
            data = self.data_source.to_xarray()

        if self.isel is not None and len(self.isel):
            data = data.isel(**self.isel)
        if self.sel is not None and len(self.sel):
            data = data.sel(**self.sel)

        wsn = self.var2ncvar.get(FV.WS, FV.WS)
        if self.ws_bins is not None:
            wsb = self.ws_bins
        elif wsn in data:
            wsb = data[wsn].to_numpy()
        else:
            raise ValueError(f"States '{self.name}': Expecting ws_bins argument, since '{wsn}' not found in data")
        wss = 0.5 * (wsb[:-1] + wsb[1:])
        wsd = wsb[1:] - wsb[:-1]
        n_ws = len(wss)
        self._n_ws = n_ws
        del wsb

        secn = None
        n_secs = None
        self._data_source = {}
        for v in [FV.WEIBULL_A, FV.WEIBULL_k, FV.WEIGHT] + self.ovars:
            if v != FV.WS and v not in self.fixed_vars:
                c = self.var2ncvar.get(v, v)
                if c not in data:
                    raise KeyError(f"States '{self.name}': Missing variable '{c}' in data, found {list(data.data_vars.keys())}")
                d = data[c]
                if len(d.dims) == 0:
                    self.fixed_vars[v] = float(d.to_numpy())
                    continue
                elif len(d.dims) != 1:
                    raise ValueError(f"States '{self.name}': Expecting single dimension for variable '{c}', got {d.dims}")
                elif secn is None:
                    secn = d.dims[0]
                    n_secs = data.sizes[secn]
                elif d.dims[0] != secn:
                    raise ValueError(f"States '{self.name}': Expecting dimension '{secn}' for variable '{c}', got {d.dims}")
                self._data_source[v] = np.zeros((n_secs, n_ws), dtype=config.dtype_double)
                self._data_source[v][:] = d.to_numpy()[:, None]
        self._data_source[FV.WS] = np.zeros((n_secs, n_ws), dtype=config.dtype_double)
        self._data_source[FV.WS][:] = wss[None, :]
        del wss

        self._data_source[FV.WEIGHT] *= weibull_weights(
            ws=self._data_source[FV.WS], 
            ws_deltas=wsd[None, :],
            A=self._data_source.pop(FV.WEIBULL_A), 
            k=self._data_source.pop(FV.WEIBULL_k), 
        )

        # remove wd 360 from the end, if wd 0 is given:
        if FV.WD in self._data_source:
            if (
                np.all(self._data_source[FV.WD][0] == 0.) and
                np.all(self._data_source[FV.WD][-1] == 360.)
            ):
                for v in self._data_source.keys():
                    self._data_source[v] = self._data_source[v][:-1]
                n_secs -= 1

        N = n_secs*n_ws
        for v in self._data_source.keys():
            self._data_source[v] = self._data_source[v].reshape(N)
        self._data_source = pd.DataFrame(data=self._data_source, index=np.arange(N))
        self._data_source.index.name = FC.STATE
        
        return super().load_data(algo, verbosity)
