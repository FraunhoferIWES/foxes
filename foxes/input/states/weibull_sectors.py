from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from os import PathLike
from pathlib import Path
from typing import Any, cast
from xarray import Dataset, open_dataset

from foxes.data import STATES
from foxes.core import Algorithm, LoadedData
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
    ws_bins
        The wind speed bins, including
        lower and upper bounds, shape: (n_ws_bins+1,)
    var2ncvar
        Mapping from variable names to variable names
        in the nc file
    sel
        Subset selection via xr.Dataset.sel()
    isel
        Subset selection via xr.Dataset.isel()
    rpars
        Additional parameters for reading the file
    RDICT
        Default xarray file reading parameters

    :group: input.states

    """

    RDICT = {}

    def __init__(
        self,
        data_source: str | Path | Dataset | pd.DataFrame,
        output_vars: list[str],
        ws_bins: ArrayLike | None = None,
        var2ncvar: dict[str, str] | None = None,
        sel: dict[str, object] | None = None,
        isel: dict[str, object] | None = None,
        read_pars: dict[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        data_source
            Either path to NetCDF or csv file or data
        output_vars
            The output variables
        ws_bins
            The wind speed bins, including
            lower and upper bounds
        var2ncvar
            Mapping from variable names to variable names
            in the nc file
        sel
            Subset selection via xr.Dataset.sel()
        isel
            Subset selection via xr.Dataset.isel()
        read_pars
            Additional parameters for reading the file
        kwargs
            Additional arguments for the base class

        """
        super().__init__(data_source, output_vars, var2col={}, **kwargs)  # type: ignore[arg-type]
        self.ws_bins = None if ws_bins is None else np.asarray(ws_bins)
        self.var2ncvar = {} if var2ncvar is None else var2ncvar
        self.sel = sel if sel is not None else {}
        self.isel = isel if isel is not None else {}
        self.rpars = {} if read_pars is None else read_pars

        if FV.WS not in self.ovars:
            raise ValueError(
                f"States '{self.name}': Expecting output variable '{FV.WS}', got {self.ovars}"
            )
        for v in [FV.WEIBULL_A, FV.WEIBULL_k, FV.WEIGHT]:
            if v in self.ovars:
                raise ValueError(
                    f"States '{self.name}': Cannot have '{v}' as output variable"
                )

        self._original_data: Dataset | pd.DataFrame | str | Path | None = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_wd={self._n_wd}, n_ws={self._n_ws})"

    def _read_data(
        self,
        algo: Algorithm,
        point_coord: str | None = None,
        verbosity: int = 0,
    ) -> Dataset:
        """
        Extracts data from file or Dataset.

        Parameters
        ----------
        algo
            The calculation algorithm
        point_coord
            The coordinate name representing the point index
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        data
            The input data

        """
        # store original data
        if self._original_data is not None:
            self._data = self._original_data
            self._original_data = None

        # read file or grab data
        cwd = self.var2ncvar.get(FV.WD, FV.WD)
        if isinstance(self.data_source, (str, PathLike)):
            fpath = get_input_path(Path(self.data_source))
            if not fpath.is_file():
                if verbosity > 0:
                    print(
                        f"States '{self.name}': Reading static data '{fpath}' from context '{STATES}'"
                    )
                fpath = algo.dbook.get_file_path(STATES, fpath.name, check_raw=False)
                if verbosity > 0:
                    print(f"Path: {fpath}")
            elif verbosity > 0:
                print(f"States '{self.name}': Reading file {fpath}")
            rpars: dict[str, Any] = dict(self.RDICT, **self.rpars)
            if fpath.suffix == ".nc":
                data = open_dataset(fpath, engine=config.nc_engine, **rpars)
            else:
                data = PandasFileHelper().read_file(fpath, **rpars)
                data.index.name = cwd
                data = data.to_xarray()
            self._original_data = data

        elif isinstance(self.data_source, Dataset):
            data = self.data_source

        elif isinstance(self.data_source, pd.DataFrame):
            data = self.data_source
            data.index.name = cwd
            data = data.to_xarray()

        # optionally select a subset
        if self.isel is not None and len(self.isel):
            isel: dict[str, Any] = dict(self.isel)
            data = data.isel(**isel)
        if self.sel is not None and len(self.sel):
            sel: dict[str, Any] = dict(self.sel)
            data = data.sel(**sel)

        # remove wd 360 from the end, if wd 0 is given:
        wd = data[cwd].to_numpy()
        if wd[0] == 0.0 and wd[-1] == 360.0:
            data = data.isel({cwd: np.s_[:-1]})

        # construct wind speed bins and bin deltas
        cws = self.var2ncvar.get(FV.WS, FV.WS)
        if self.ws_bins is not None:
            wsb = self.ws_bins
            wss = 0.5 * (wsb[:-1] + wsb[1:])
        elif cws in data:
            wss = data[cws].to_numpy()
            wsb = np.zeros((len(wss) + 1,), dtype=config.dtype_double)
            wsb[1:-1] = 0.5 * (wss[1:] + wss[:-1])
            wsb[0] = wss[0] - 0.5 * wsb[1]
            wsb[-1] = wss[-1] + 0.5 * wsb[-2]
            self.ws_bins = wsb
        else:
            raise ValueError(
                f"States '{self.name}': Expecting ws_bins argument, since '{cws}' not found in data"
            )
        wsd = wsb[1:] - wsb[:-1]
        n_ws = len(wss)
        del wsb

        # prepare data binning
        if self._original_data is None:
            self._original_data = self.data_source
        cpt: str | None = (
            self.var2ncvar.get(FC.POINT, FC.POINT) if point_coord is not None else None
        )
        n_wd = data.sizes[cwd]
        self.BIN_WD = self.var("bin_wd")
        self.BIN_WS = self.var("bin_ws")
        self.POINT = self.var("point")
        shp: tuple[int, ...]
        dms: tuple[str, ...]
        if cpt is None:
            n_pt = 0
            shp = (n_wd, n_ws)
            dms = (self.BIN_WD, self.BIN_WS)
        else:
            n_pt = data.sizes[cpt]
            shp = (n_wd, n_ws, n_pt)
            dms = (self.BIN_WD, self.BIN_WS, self.POINT)

        # create binned data
        self._data = {
            FV.WD: np.zeros(shp, dtype=config.dtype_double),
            FV.WS: np.zeros(shp, dtype=config.dtype_double),
        }
        if cpt is None:
            self._data[FV.WD][:] = data[cwd].to_numpy()[:, None]
            self._data[FV.WS][:] = wss[None, :]
        else:
            self._data[FV.WD][:] = data[cwd].to_numpy()[:, None, None]
            self._data[FV.WS][:] = wss[None, :, None]
        for v in [FV.WEIBULL_A, FV.WEIBULL_k, FV.WEIGHT] + self.ovars:
            if v not in [FV.WS, FV.WD] and v not in self.fixed_vars:
                w = self.var2ncvar.get(v, v)
                if w not in data:
                    raise KeyError(
                        f"States '{self.name}': Missing variable '{w}' in data, found {list(data.data_vars.keys())}"
                    )
                data_array = data[w]
                iws = data_array.dims.index(cws) if cws in data_array.dims else -1
                iwd = data_array.dims.index(cwd) if cwd in data_array.dims else -1
                ipt = (
                    data_array.dims.index(cpt)
                    if cpt is not None and cpt in data_array.dims
                    else -1
                )
                d_array: np.ndarray = data_array.to_numpy()
                if v in [FV.WEIBULL_A, FV.WEIBULL_k, FV.WEIGHT]:
                    if cws in data_array.dims:
                        raise ValueError(
                            f"States '{self.name}': Cannot have '{cws}' as dimension in variable '{v}', got {data_array.dims}"
                        )
                    if cwd not in data_array.dims:
                        raise ValueError(
                            f"States '{self.name}': Expecting '{cwd}' as dimension in variable '{v}', got {data_array.dims}"
                        )
                if iws < 0 and iwd < 0 and ipt < 0:
                    self.fixed_vars[v] = cast(Any, d_array)
                elif iws >= 0 and iwd >= 0 and ipt >= 0:
                    self._data[v] = np.moveaxis(d_array, [iwd, iws, ipt], [0, 1, 2])
                elif iws >= 0 and iwd >= 0 and ipt < 0:
                    if cpt is None:
                        self._data[v] = np.moveaxis(d_array, [iwd, iws], [0, 1])
                    else:
                        self._data[v] = np.zeros(shp, dtype=config.dtype_double)
                        self._data[v][:] = np.moveaxis(d_array, [iwd, iws], [0, 1])[
                            :, :, None
                        ]
                elif iws >= 0 and iwd < 0 and ipt >= 0:
                    self._data[v] = np.zeros(shp, dtype=config.dtype_double)
                    self._data[v][:] = np.moveaxis(d_array, [iws, ipt], [0, 1])[
                        None, :, :
                    ]
                elif iws < 0 and iwd >= 0 and ipt >= 0:
                    self._data[v] = np.zeros(shp, dtype=config.dtype_double)
                    self._data[v][:] = np.moveaxis(d_array, [iwd, ipt], [0, 1])[
                        :, None, :
                    ]
                elif iws >= 0 and iwd < 0 and ipt < 0:
                    self._data[v] = np.zeros(shp, dtype=config.dtype_double)
                    if cpt is None:
                        self._data[v][:] = d_array[None, :]
                    else:
                        self._data[v][:] = d_array[None, :, None]
                elif iws < 0 and iwd >= 0 and ipt < 0:
                    self._data[v] = np.zeros(shp, dtype=config.dtype_double)
                    if cpt is None:
                        self._data[v][:] = d_array[:, None]
                    else:
                        self._data[v][:] = d_array[:, None, None]
                elif iws < 0 and iwd < 0 and ipt >= 0:
                    self._data[v] = np.zeros(shp, dtype=config.dtype_double)
                    self._data[v][:] = d_array[None, None, :]

        # compute Weibull weights
        self._data[FV.WEIGHT] *= weibull_weights(
            ws=wss[None, :, None] if cpt is not None else wss[None, :],
            ws_deltas=wsd[None, :, None] if cpt is not None else wsd[None, :],
            A=self._data.pop(FV.WEIBULL_A),
            k=self._data.pop(FV.WEIBULL_k),
        )

        # translate binned data to states
        self._n_pt = n_pt
        self._n_wd = n_wd
        self._n_ws = n_ws
        self._N = n_wd * n_ws
        self._data = Dataset(
            data_vars={v: (dms, d) for v, d in self._data.items()},
        )

        return data

    def load_data(
        self,
        algo: Algorithm,
        loaded_data: LoadedData,
        force: bool = False,
        verbosity: int = 0,
    ) -> None:
        """
        Load and/or create all model data that is subject to chunking.

        Such data should not be stored under self, for memory reasons. The
        data returned here will automatically be chunked and then provided
        as part of the mdata object during calculations.

        Parameters
        ----------
        algo
            The calculation algorithm
        loaded_data
            Data that has already been loaded, to be extended by this function.
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.
        force
            Overwrite existing data
        verbosity
            The verbosity level, 0 = silent

        """
        self._read_data(algo, verbosity=0)
        assert isinstance(self._data, Dataset)

        tmp = {}
        for v, d in self._data.data_vars.items():
            if self.POINT in d.dims:
                raise TypeError(
                    f"States '{self.name}': Variable '{v}' has unsupported dimension '{self.POINT}', dims = {d.dims}"
                )
            else:
                tmp[v] = d.to_numpy().reshape(self._N)
        self._data = tmp
        del tmp

        state_data = pd.DataFrame(data=self._data, index=np.arange(self._N))
        state_data.index.name = FC.STATE
        self._data = state_data

        super().load_data(algo, loaded_data, force=force, verbosity=verbosity)
