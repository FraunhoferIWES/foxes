import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import interpn

from foxes.core import States
from foxes.utils import wd2uv, uv2wd, import_module
from foxes.data import STATES, StaticData
from foxes.config import config, get_path
import foxes.variables as FV
import foxes.constants as FC


class SliceDataNC(States):
    """
    Heterogeneous ambient states on a regular
    horizontal grid in NetCDF format, independent
    of height.

    Attributes
    ----------
    data_source: str or xarray.Dataset
        The data or the file search pattern, should end with
        suffix '.nc'. One or many files.
    ovars: list of str
        The output variables
    var2ncvar: dict
        Mapping from variable names to variable names
        in the nc file
    fixed_vars: dict
        Uniform values for output variables, instead
        of reading from data
    states_coord: str
        The states coordinate name in the data
    x_coord: str
        The x coordinate name in the data
    y_coord: str
        The y coordinate name in the data
    pre_load: bool
        Flag for loading all data into memory during
        initialization
    weight_ncvar: str
        Name of the weight data variable in the nc file(s)
    bounds_error: bool
        Flag for raising errors if bounds are exceeded
    fill_value: number
        Fill value in case of exceeding bounds, if no bounds error
    time_format: str
        The datetime parsing format string
    interp_nans: bool
        Linearly interpolate nan values
    interpn_pars: dict, optional
        Additional parameters for scipy.interpolate.interpn

    :group: input.states

    """

    def __init__(
        self,
        data_source,
        output_vars,
        var2ncvar={},
        fixed_vars={},
        states_coord="Time",
        x_coord="UTMX",
        y_coord="UTMY",
        pre_load=True,
        weight_ncvar=None,
        time_format="%Y-%m-%d_%H:%M:%S",
        sel=None,
        isel=None,
        interp_nans=False,
        verbosity=1,
        **interpn_pars,
    ):
        """
        Constructor.

        Parameters
        ----------
        data_source: str or xarray.Dataset
            The data or the file search pattern, should end with
            suffix '.nc'. One or many files.
        output_vars: list of str
            The output variables
        var2ncvar: dict, optional
            Mapping from variable names to variable names
            in the nc file
        fixed_vars: dict, optional
            Uniform values for output variables, instead
            of reading from data
        states_coord: str
            The states coordinate name in the data
        x_coord: str
            The x coordinate name in the data
        y_coord: str
            The y coordinate name in the data
        pre_load: bool
            Flag for loading all data into memory during
            initialization
        weight_ncvar: str, optional
            Name of the weight data variable in the nc file(s)
        time_format: str
            The datetime parsing format string
        sel: dict, optional
            Subset selection via xr.Dataset.sel()
        isel: dict, optional
            Subset selection via xr.Dataset.isel()
        interp_nans: bool
            Linearly interpolate nan values
        verbosity: int
            Verbosity level for pre_load file reading
        interpn_pars: dict, optional
            Additional parameters for scipy.interpolate.interpn

        """
        super().__init__()

        self.states_coord = states_coord
        self.ovars = output_vars
        self.fixed_vars = fixed_vars
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.weight_ncvar = weight_ncvar
        self.pre_load = pre_load
        self.time_format = time_format
        self.sel = sel
        self.isel = isel
        self.interpn_pars = interpn_pars
        self.interp_nans = interp_nans

        self.var2ncvar = {
            v: var2ncvar.get(v, v) for v in output_vars if v not in fixed_vars
        }

        self._N = None

        self.__data_source = data_source
        self.__weights = None
        self.__inds = None

        # pre-load file reading:
        if not isinstance(self.data_source, xr.Dataset):
            if "*" in str(self.data_source):
                pass
            else:
                self.__data_source = get_path(self.data_source)
                if not self.data_source.is_file():
                    self.__data_source = StaticData().get_file_path(
                        STATES, self.data_source.name, check_raw=False
                    )
            if verbosity:
                if pre_load:
                    print(
                        f"States '{self.name}': Reading data from '{self.data_source}'"
                    )
                else:
                    print(
                        f"States '{self.name}': Reading index from '{self.data_source}'"
                    )

            def _read_ds():
                self._data_source = get_path(self.data_source)
                if self.data_source.is_file():
                    return xr.open_dataset(self.data_source)
                else:
                    # try to read multiple files, needs dask:
                    try:
                        return xr.open_mfdataset(
                            str(self.data_source),
                            parallel=False,
                            concat_dim=self.states_coord,
                            combine="nested",
                            data_vars="minimal",
                            coords="minimal",
                            compat="override",
                        )
                    except ValueError as e:
                        import_module("dask", hint="pip install dask")
                        raise e

            self.__data_source = _read_ds()

        if sel is not None:
            self.__data_source = self.data_source.sel(self.sel)
        if isel is not None:
            self.__data_source = self.data_source.isel(self.isel)
        if pre_load:
            self.__data_source.load()

        self._get_inds(self.data_source)

    @property
    def data_source(self):
        """
        The data source

        Returns
        -------
        s: object
            The data source

        """
        if self.pre_load and self.running:
            raise ValueError(
                f"States '{self.name}': Cannot acces data_source while running"
            )
        return self.__data_source

    def _get_inds(self, ds):
        """
        Helper function for index and weights
        reading
        """
        for c in [self.states_coord, self.x_coord, self.y_coord]:
            if not c in ds:
                raise KeyError(
                    f"States '{self.name}': Missing coordinate '{c}' in data"
                )

        self.__inds = ds[self.states_coord].to_numpy()
        if self.time_format is not None:
            self.__inds = pd.to_datetime(
                self.__inds, format=self.time_format
            ).to_numpy()
        self._N = len(self.__inds)

        if self.weight_ncvar is not None:
            self.__weights = ds[self.weight_ncvar].to_numpy()

        for v in self.ovars:
            if v in self.var2ncvar:
                ncv = self.var2ncvar[v]
                if not ncv in ds:
                    raise KeyError(
                        f"States '{self.name}': nc variable '{ncv}' not found in data, found: {sorted(list(ds.keys()))}"
                    )
            elif v not in self.fixed_vars:
                raise ValueError(
                    f"States '{self.name}': Variable '{v}' neither found in var2ncvar not in fixed_vars"
                )

    def _get_data(self, ds, verbosity):
        """
        Helper function for data extraction
        """
        x = ds[self.x_coord].to_numpy()
        y = ds[self.y_coord].to_numpy()
        n_x = len(x)
        n_y = len(y)
        n_sts = ds.sizes[self.states_coord]

        cor_sxy = (self.states_coord, self.x_coord, self.y_coord)
        cor_syx = (self.states_coord, self.y_coord, self.x_coord)
        cor_s = (self.states_coord,)
        vars_syx = []
        vars_s = []
        for v, ncv in self.var2ncvar.items():
            if ds[ncv].dims == cor_syx or ds[ncv].dims == cor_sxy:
                vars_syx.append(v)
            elif ds[ncv].dims == cor_s:
                vars_s.append(v)
            else:
                raise ValueError(
                    f"States '{self.name}': Wrong coordinate order for variable '{ncv}': Found {ds[ncv].dims}, expecting {cor_sxy}, {cor_syx}, or {cor_s}"
                )

        data = np.zeros(
            (n_sts, n_y, n_x, len(self.var2ncvar)), dtype=config.dtype_double
        )
        for v in vars_syx:
            ncv = self.var2ncvar[v]
            if ds[ncv].dims == cor_syx:
                data[..., self._dkys[v]] = ds[ncv][:]
            else:
                data[..., self._dkys[v]] = np.swapaxes(ds[ncv].to_numpy(), 1, 2)
        for v in vars_s:
            ncv = self.var2ncvar[v]
            data[..., self._dkys[v]] = ds[ncv].to_numpy()[:, None, None]
        if FV.WD in self.fixed_vars:
            data[..., self._dkys[FV.WD]] = np.full(
                (n_sts, n_y, n_x), self.fixed_vars[FV.WD], dtype=config.dtype_double
            )

        if verbosity > 1:
            print(f"\n{self.name}: Data ranges")
            for v, i in self._dkys.items():
                d = data[..., i]
                nn = np.sum(np.isnan(d))
                print(
                    f"  {v}: {np.nanmin(d)} --> {np.nanmax(d)}, nans: {nn} ({100*nn/len(d.flat):.2f}%)"
                )

        return data

    def output_point_vars(self, algo):
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
        return self.ovars

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

        if (FV.WS in self.ovars and FV.WD not in self.ovars) or (
            FV.WS not in self.ovars and FV.WD in self.ovars
        ):
            raise KeyError(
                f"States '{self.name}': Missing '{FV.WS}' or '{FV.WD}' in output variables {self.ovars}"
            )

        # ensure WD and WS get the first two slots of data:
        self._dkys = {}
        if FV.WS in self.ovars:
            self._dkys[FV.WD] = 0
        if FV.WS in self.var2ncvar:
            self._dkys[FV.WS] = 1
        for v in self.var2ncvar:
            if v not in self._dkys:
                self._dkys[v] = len(self._dkys)
        self._n_dvars = len(self._dkys)

        if self.__weights is None:
            self.__weights = np.full(
                (self._N, algo.n_turbines), 1.0 / self._N, dtype=config.dtype_double
            )

        idata = super().load_data(algo, verbosity)

        if self.pre_load:
            self.X = self.var(FV.X)
            self.Y = self.var(FV.Y)
            self.VARS = self.var("vars")
            self.DATA = self.var("data")

            ds = self.data_source

            y = ds[self.y_coord].to_numpy()
            x = ds[self.x_coord].to_numpy()
            v = list(self._dkys.keys())
            coos = (FC.STATE, self.Y, self.X, self.VARS)
            data = self._get_data(ds, verbosity)
            data = (coos, data)

            idata["coords"][self.Y] = y
            idata["coords"][self.X] = x
            idata["coords"][self.VARS] = v
            idata["data_vars"][self.DATA] = data

        return idata

    def set_running(
        self,
        algo,
        data_stash,
        sel=None,
        isel=None,
        verbosity=0,
    ):
        """
        Sets this model status to running, and moves
        all large data to stash.

        The stashed data will be returned by the
        unset_running() function after running calculations.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        data_stash: dict
            Large data stash, this function adds data here.
            Key: model name. Value: dict, large model data
        sel: dict, optional
            The subset selection dictionary
        isel: dict, optional
            The index subset selection dictionary
        verbosity: int
            The verbosity level, 0 = silent

        """
        super().set_running(algo, data_stash, sel, isel, verbosity)

        data_stash[self.name] = dict(
            weights=self.__weights,
            inds=self.__inds,
        )
        del self.__weights, self.__inds

        if self.pre_load:
            data_stash[self.name]["data_source"] = self.__data_source
            del self.__data_source

    def unset_running(
        self,
        algo,
        data_stash,
        sel=None,
        isel=None,
        verbosity=0,
    ):
        """
        Sets this model status to not running, recovering large data
        from stash

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        data_stash: dict
            Large data stash, this function adds data here.
            Key: model name. Value: dict, large model data
        sel: dict, optional
            The subset selection dictionary
        isel: dict, optional
            The index subset selection dictionary
        verbosity: int
            The verbosity level, 0 = silent

        """
        super().unset_running(algo, data_stash, sel, isel, verbosity)

        data = data_stash[self.name]
        self.__weights = data.pop("weights")
        self.__inds = data.pop("inds")

        if self.pre_load:
            self.__data_source = data.pop("data_source")

    def size(self):
        """
        The total number of states.

        Returns
        -------
        int:
            The total number of states

        """
        return self._N

    def index(self):
        """
        The index list

        Returns
        -------
        indices: array_like
            The index labels of states, or None for default integers

        """
        if self.running:
            raise ValueError(f"States '{self.name}': Cannot acces index while running")
        return self.__inds

    def output_point_vars(self, algo):
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
        return self.ovars

    def weights(self, algo):
        """
        The statistical weights of all states.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        weights: numpy.ndarray
            The weights, shape: (n_states, n_turbines)

        """
        if self.running:
            raise ValueError(
                f"States '{self.name}': Cannot acces weights while running"
            )
        return self.__weights

    def calculate(self, algo, mdata, fdata, tdata):
        """ "
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
        tdata: foxes.core.TData
            The target point data

        Returns
        -------
        results: dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape
            (n_states, n_targets, n_tpoints)

        """
        # prepare:
        n_states = tdata.n_states
        n_targets = tdata.n_targets
        n_tpoints = tdata.n_tpoints
        points = tdata[FC.TARGETS].reshape(n_states, n_targets * n_tpoints, 3)
        n_pts = points.shape[1]
        n_states = fdata.n_states

        # pick pre-loaded data:
        if self.pre_load:
            x = mdata[self.X]
            y = mdata[self.Y]
            data = mdata[self.DATA].copy()

        # read data for this chunk:
        else:
            i0 = mdata.states_i0(counter=True)
            s = slice(i0, i0 + n_states)
            ds = self.data_source.isel({self.states_coord: s}).load()

            x = ds[self.x_coord].to_numpy()
            y = ds[self.y_coord].to_numpy()
            data = self._get_data(ds, verbosity=0)

            del ds
        n_y = len(y)
        n_x = len(x)

        # translate WS, WD into U, V:
        if FV.WD in self.ovars and FV.WS in self.ovars:
            wd = data[..., self._dkys[FV.WD]]
            ws = (
                data[..., self._dkys[FV.WS]]
                if FV.WS in self._dkys
                else self.fixed_vars[FV.WS]
            )
            wdwsi = [self._dkys[FV.WD], self._dkys[FV.WS]]
            data[..., wdwsi] = wd2uv(wd, ws, axis=-1)
            del ws, wd

        # prepare points:
        sts = np.arange(n_states)
        pts = np.append(
            points, np.zeros((n_states, n_pts, 1), dtype=config.dtype_double), axis=2
        )
        pts[:, :, 3] = sts[:, None]
        pts = pts.reshape(n_states * n_pts, 3)
        pts = np.flip(pts, axis=1)
        gvars = (sts, y, x)

        # interpolate nan values:
        if self.interp_nans and np.any(np.isnan(data)):
            df = pd.DataFrame(
                index=pd.MultiIndex.from_product(gvars, names=["state", "y", "x"]),
                data={
                    v: data[..., vi].reshape(n_states * n_y * n_x)
                    for v, vi in self._dkys.items()
                },
            )
            df.interpolate(
                axis=0, method="linear", limit_direction="forward", inplace=True
            )
            df.interpolate(
                axis=0, method="linear", limit_direction="backward", inplace=True
            )
            data = df.to_numpy().reshape(n_states, n_y, n_x, self._n_dvars)
            del df

        # interpolate:
        try:
            ipars = dict(bounds_error=True, fill_value=None)
            ipars.update(self.interpn_pars)
            data = interpn(gvars, data, pts, **ipars).reshape(
                n_states, n_pts, self._n_dvars
            )
        except ValueError as e:
            print(f"\nStates '{self.name}': Interpolation error")
            print("INPUT VARS: (state, y, x)")
            print(
                "DATA BOUNDS:",
                [float(np.min(d)) for d in gvars],
                [float(np.max(d)) for d in gvars],
            )
            print(
                "EVAL BOUNDS:",
                [float(np.min(p)) for p in pts.T],
                [float(np.max(p)) for p in pts.T],
            )
            print(
                "\nMaybe you want to try the option 'bounds_error=False'? This will extrapolate the data.\n"
            )
            raise e
        del pts, x, y, gvars

        # interpolate nan values:
        if self.interp_nans and np.any(np.isnan(data)):
            df = pd.DataFrame(
                index=pd.MultiIndex.from_product(
                    (sts, range(n_pts)), names=["state", "point"]
                ),
                data={
                    v: data[:, :, vi].reshape(n_states * n_pts)
                    for v, vi in self._dkys.items()
                },
            )
            df["x"] = points[:, :, 0].reshape(n_states * n_pts)
            df["y"] = points[:, :, 1].reshape(n_states * n_pts)
            df = df.reset_index().set_index(["state", "x", "y"])
            df.interpolate(
                axis=0, method="linear", limit_direction="forward", inplace=True
            )
            df.interpolate(
                axis=0, method="linear", limit_direction="backward", inplace=True
            )
            df = df.reset_index().drop(["x", "y"], axis=1).set_index(["state", "point"])
            data = df.to_numpy().reshape(n_states, n_pts, self._n_dvars)
            del df

        # set output:
        out = {}
        if FV.WD in self.ovars and FV.WS in self.ovars:
            uv = data[..., wdwsi]
            out[FV.WS] = np.linalg.norm(uv, axis=-1)
            out[FV.WD] = uv2wd(uv, axis=-1)
            del uv
        for v in self.ovars:
            if v not in out:
                if v in self._dkys:
                    out[v] = data[..., self._dkys[v]]
                else:
                    out[v] = np.full(
                        (n_states, n_pts), self.fixed_vars[v], dtype=config.dtype_double
                    )

        return {v: d.reshape(n_states, n_targets, n_tpoints) for v, d in out.items()}
