import numpy as np
import pandas as pd
import xarray as xr
from copy import copy
from scipy.interpolate import interpn

from foxes.core import States, get_engine
from foxes.utils import wd2uv, uv2wd, import_module
from foxes.data import STATES, StaticData
import foxes.variables as FV
import foxes.constants as FC
from foxes.config import config, get_input_path


def _read_nc_file(
    fpath,
    coords,
    vars,
    nc_engine,
    sel,
    isel,
    minimal,
):
    """Helper function for nc file reading"""
    data = xr.open_dataset(fpath, engine=nc_engine)
    for c in coords:
        if c is not None and c not in data:
            raise KeyError(
                f"Missing coordinate '{c}' in file {fpath}, got: {list(data.coords.keys())}"
            )
    if minimal:
        return data[coords[0]].to_numpy()
    else:
        data = data[vars]
        data.attrs = {}
        if isel is not None and len(isel):
            data = data.isel(**isel)
        if sel is not None and len(sel):
            data = data.sel(**sel)
        return data


class FieldDataNC(States):
    """
    Heterogeneous ambient states on a regular
    horizontal grid in NetCDF format.

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
    h_coord: str
        The height coordinate name in the data
    load_mode: str
        The load mode, choices: preload, lazy, fly.
        preload loads all data during initialization,
        lazy lazy-loads the data using dask, and fly
        reads only states index and weights during initialization
        and then opens the relevant files again within
        the chunk calculation
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
    bounds_extra_space: float or str
        The extra space, either float in m,
        or str for units of D, e.g. '2.5D'

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
        h_coord="height",
        load_mode="preload",
        weight_ncvar=None,
        time_format="%Y-%m-%d_%H:%M:%S",
        sel=None,
        isel=None,
        interp_nans=False,
        bounds_extra_space=1000,
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
        h_coord: str, optional
            The height coordinate name in the data
        load_mode: str
            The load mode, choices: preload, lazy, fly.
            preload loads all data during initialization,
            lazy lazy-loads the data using dask, and fly
            reads only states index and weights during initialization
            and then opens the relevant files again within
            the chunk calculation
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
        bounds_extra_space: float or str, optional
            The extra space, either float in m,
            or str for units of D, e.g. '2.5D'
        interpn_pars: dict, optional
            Additional parameters for scipy.interpolate.interpn

        """
        super().__init__()

        self.states_coord = states_coord
        self.ovars = list(output_vars)
        self.fixed_vars = fixed_vars
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.h_coord = h_coord
        self.weight_ncvar = weight_ncvar
        self.load_mode = load_mode
        self.time_format = time_format
        self.sel = sel if sel is not None else {}
        self.isel = isel if isel is not None else {}
        self.interpn_pars = interpn_pars
        self.interp_nans = interp_nans
        self.bounds_extra_space = bounds_extra_space

        self.var2ncvar = {
            v: var2ncvar.get(v, v) for v in output_vars if v not in fixed_vars
        }

        self._N = None

        self.__data_source = data_source
        self.__inds = None

    @property
    def data_source(self):
        """
        The data source

        Returns
        -------
        s: object
            The data source

        """
        if self.load_mode in ["preload", "fly"] and self.running:
            raise ValueError(
                f"States '{self.name}': Cannot access data_source while running for load mode '{self.load_mode}'"
            )
        return self.__data_source

    def _get_data(self, ds, coords, verbosity):
        """
        Helper function for data extraction
        """
        for ci, c in enumerate(coords):
            found = False
            if c is not None:
                for v in ds.data_vars.values():
                    if c in v.dims:
                        found = True
                        break
            if not found:
                coords[ci] = None

        dlst = []
        for c in coords:
            if c is not None:
                dlst.append(np.atleast_1d(ds[c].to_numpy()))
            else:
                dlst.append(np.array([0], dtype=config.dtype_double))
        sts, h, y, x = dlst
        n_sts, n_h, n_y, n_x = [len(u) for u in dlst]

        cor_shxy = (self.states_coord, self.h_coord, self.x_coord, self.y_coord)
        cor_shyx = (self.states_coord, self.h_coord, self.y_coord, self.x_coord)
        cor_sxy = (self.states_coord, self.x_coord, self.y_coord)
        cor_syx = (self.states_coord, self.y_coord, self.x_coord)
        cor_sh = (self.states_coord, self.h_coord)
        cor_s = (self.states_coord,)
        vars_shyx = []
        vars_syx = []
        vars_sh = []
        vars_s = []
        for v, ncv in self.var2ncvar.items():
            if ds[ncv].dims == cor_shyx or ds[ncv].dims == cor_shxy:
                vars_shyx.append(v)
            elif ds[ncv].dims == cor_syx or ds[ncv].dims == cor_sxy:
                vars_syx.append(v)
            elif ds[ncv].dims == cor_sh:
                vars_sh.append(v)
            elif ds[ncv].dims == cor_s:
                vars_s.append(v)
            else:
                expc = [
                    c
                    for c in [cor_shxy, cor_shyx, cor_sxy, cor_syx, cor_sh, cor_s]
                    if None not in c
                ]
                raise ValueError(
                    f"States '{self.name}': Wrong coordinates for variable '{ncv}': Found {ds[ncv].dims}, expecting one of {expc}"
                )

        data = np.zeros(
            (n_sts, n_h, n_y, n_x, len(self.var2ncvar)), dtype=config.dtype_double
        )
        for v in vars_shyx:
            ncv = self.var2ncvar[v]
            if ds[ncv].dims == cor_shyx:
                data[..., self._dkys[v]] = ds[ncv][:]
            else:
                data[..., self._dkys[v]] = np.swapaxes(ds[ncv].to_numpy(), 2, 3)
        for v in vars_syx:
            ncv = self.var2ncvar[v]
            if ds[ncv].dims == cor_syx:
                data[..., self._dkys[v]] = ds[ncv].to_numpy()[:, None]
            else:
                data[..., self._dkys[v]] = np.swapaxes(ds[ncv].to_numpy(), 1, 2)[
                    :, None
                ]
        for v in vars_sh:
            ncv = self.var2ncvar[v]
            data[..., self._dkys[v]] = ds[ncv].to_numpy()[:, :, None, None]
        for v in vars_s:
            ncv = self.var2ncvar[v]
            data[..., self._dkys[v]] = ds[ncv].to_numpy()[:, None, None, None]
        if FV.WD in self.fixed_vars:
            data[..., self._dkys[FV.WD]] = np.full(
                (n_sts, n_h, n_y, n_x),
                self.fixed_vars[FV.WD],
                dtype=config.dtype_double,
            )

        weights = None
        if self.weight_ncvar is not None:
            if self.weight_ncvar not in ds.data_vars:
                raise KeyError(f"States '{self.name}': Missing weights variable '{self.weight_ncvar}' in data, found {sorted(list(ds.data_vars.keys()))}")
            if ds[self.weight_ncvar].dims != (self.states_coord,):
                raise ValueError(f"States '{self.name}': Weights variable '{self.weight_ncvar}' has wrong dimensions. Expecting {(self.states_coord,)}, got {ds[self.weight_ncvar].dims}")
            weights = ds[self.weight_ncvar].to_numpy()

        if verbosity > 1:
            print(f"\n{self.name}: Data ranges")
            for v, i in self._dkys.items():
                d = data[..., i]
                nn = np.sum(np.isnan(d))
                print(
                    f"  {v}: {np.nanmin(d)} --> {np.nanmax(d)}, nans: {nn} ({100*nn/len(d.flat):.2f}%)"
                )

        return sts, h, y, x, data, weights

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

        # pre-load file reading:
        coords = [self.states_coord, self.h_coord, self.y_coord, self.x_coord]
        if not isinstance(self.data_source, xr.Dataset):

            # check variables:
            for v in self.ovars:
                if v == FV.WEIGHT and self.weight_ncvar is None:
                    pass
                elif v not in self.var2ncvar and v not in self.fixed_vars:
                    raise ValueError(
                        f"States '{self.name}': Variable '{v}' neither found in var2ncvar not in fixed_vars"
                    )
            if (FV.WS in self.ovars and FV.WD not in self.ovars) or (
                FV.WS not in self.ovars and FV.WD in self.ovars
            ):
                raise KeyError(
                    f"States '{self.name}': Missing '{FV.WS}' or '{FV.WD}' in output variables {self.ovars}"
                )

            # check static data:
            fpath = get_input_path(self.data_source)
            if "*" not in str(self.data_source):
                if not fpath.is_file():
                    fpath = StaticData().get_file_path(
                        STATES, fpath.name, check_raw=False
                    )

            # find bounds:
            if self.x_coord is not None and self.x_coord not in self.sel:
                xy_min, xy_max = algo.farm.get_xy_bounds(
                    extra_space=self.bounds_extra_space, algo=algo
                )
                if verbosity > 0:
                    print(
                        f"States '{self.name}': Restricting to bounds {xy_min} - {xy_max}"
                    )
                self.sel.update(
                    {
                        self.x_coord: slice(xy_min[0], xy_max[1]),
                        self.y_coord: slice(xy_min[1], xy_max[1]),
                    }
                )

            # read files:
            if verbosity > 0:
                if self.load_mode == "preload":
                    print(
                        f"States '{self.name}': Reading data from '{self.data_source}'"
                    )
                elif self.load_mode == "lazy":
                    print(
                        f"States '{self.name}': Reading header from '{self.data_source}'"
                    )
                else:
                    print(
                        f"States '{self.name}': Reading states from '{self.data_source}'"
                    )
            files = sorted(list(fpath.resolve().parent.glob(fpath.name)))
            vars = list(self.var2ncvar.values())
            if self.weight_ncvar is not None:
                vars += [self.weight_ncvar]
            self.__data_source = get_engine().map(
                _read_nc_file,
                files,
                coords=coords,
                vars=vars,
                nc_engine=config.nc_engine,
                isel=self.isel,
                sel=self.sel,
                minimal=self.load_mode == "fly",
            )

            if self.load_mode in ["preload", "lazy"]:

                if self.load_mode == "lazy":
                    try:
                        self.__data_source = [ds.chunk() for ds in self.__data_source]
                    except (ModuleNotFoundError, ValueError) as e:
                        import_module("dask")
                        raise e
                self.__data_source = xr.concat(
                    self.__data_source,
                    dim=self.states_coord,
                    coords="minimal",
                    data_vars="minimal",
                    compat="equals",
                    join="exact",
                    combine_attrs="drop",
                )
                if self.load_mode == "preload":
                    self.__data_source.load()
                self.__inds = self.__data_source[self.states_coord].to_numpy()
                self._N = len(self.__inds)

            elif self.load_mode == "fly":
                self.__inds = self.__data_source
                self.__data_source = fpath
                self._files_maxi = {f: len(inds) for f, inds in zip(files, self.__inds)}
                self.__inds = np.concatenate(self.__inds, axis=0)
                self._N = len(self.__inds)

            else:
                raise KeyError(
                    f"States '{self.name}': Unknown load_mode '{self.load_mode}', choices: preload, lazy, fly"
                )

            if self.time_format is not None:
                self.__inds = pd.to_datetime(
                    self.__inds, format=self.time_format
                ).to_numpy()

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

        idata = super().load_data(algo, verbosity)

        if self.load_mode == "preload":
            self.X = self.var(FV.X)
            self.Y = self.var(FV.Y)
            self.H = self.var(FV.H)
            self.VARS = self.var("vars")
            self.DATA = self.var("data")
            self.WEIGHT = self.var(FV.WEIGHT)

            __, h, y, x, data, weights = self._get_data(self.data_source, coords, verbosity)
            self._prl_coords = coords

            coos = (FC.STATE, self.H, self.Y, self.X, self.VARS)
            data = (coos, data)

            idata["coords"][self.H] = h
            idata["coords"][self.Y] = y
            idata["coords"][self.X] = x
            idata["coords"][self.VARS] = list(self._dkys.keys())
            idata["data_vars"][self.DATA] = data
            if weights is not None:
                idata["data_vars"][self.WEIGHT] = ((FC.STATE,), weights)

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
            inds=self.__inds,
        )
        del self.__inds

        if self.load_mode == "preload":
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
        self.__inds = data.pop("inds")

        if self.load_mode == "preload":
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
            raise ValueError(f"States '{self.name}': Cannot access index while running")
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

    def calculate(self, algo, mdata, fdata, tdata):
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
        coords = [self.states_coord, self.h_coord, self.y_coord, self.x_coord]

        # case preload:
        if self.load_mode == "preload":
            x = mdata[self.X]
            y = mdata[self.Y]
            h = mdata[self.H]
            data = mdata[self.DATA].copy()
            weights = mdata.get(self.WEIGHT, None)
            coords = self._prl_coords

        # case lazy:
        elif self.load_mode == "lazy":
            i0 = mdata.states_i0(counter=True)
            s = slice(i0, i0 + n_states)
            ds = self.data_source.isel({self.states_coord: s}).load()
            __, h, y, x, data, weights = self._get_data(ds, coords, verbosity=0)
            del ds

        # case fly:
        elif self.load_mode == "fly":
            vars = list(self.var2ncvar.values())
            if self.weight_ncvar is not None:
                vars += [self.weight_ncvar]

            i0 = mdata.states_i0(counter=True)
            i1 = i0 + n_states
            j0 = 0
            data = []
            for fpath, n in self._files_maxi.items():
                if i0 < j0:
                    break
                else:
                    j1 = j0 + n
                    if i0 < j1:
                        a = i0 - j0
                        b = min(i1, j1) - j0
                        isel = copy(self.isel)
                        isel[self.states_coord] = slice(a, b)

                        data.append(
                            _read_nc_file(
                                fpath,
                                coords=coords,
                                vars=vars,
                                nc_engine=config.nc_engine,
                                isel=isel,
                                sel=self.sel,
                                minimal=False,
                            )
                        )

                        i0 += b - a
                    j0 = j1

            assert (
                i0 == i1
            ), f"States '{self.name}': Missing states for load_mode '{self.load_mode}': (i0, i1) = {(i0, i1)}"

            data = xr.concat(data, dim=self.states_coord)
            __, h, y, x, data, weights = self._get_data(data, coords, verbosity=0)

        else:
            raise KeyError(
                f"States '{self.name}': Unknown load_mode '{self.load_mode}', choices: preload, lazy, fly"
            )

        n_h = len(h)
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
        pts = pts.reshape(n_states * n_pts, 4)
        pts = np.flip(pts, axis=1)
        gvars = (sts, h, y, x)

        # reset None coordinate data, since that should not be interpolated:
        for i, (c, g) in enumerate(zip(coords, gvars)):
            if c is None:
                pts[..., i] = g[0]

        # interpolate nan values:
        if self.interp_nans and np.any(np.isnan(data)):
            df = pd.DataFrame(
                index=pd.MultiIndex.from_product(
                    gvars, names=["state", "height", "y", "x"]
                ),
                data={
                    v: data[..., vi].reshape(n_states * n_h * n_y * n_x)
                    for v, vi in self._dkys.items()
                },
            )
            df.interpolate(
                axis=0, method="linear", limit_direction="forward", inplace=True
            )
            df.interpolate(
                axis=0, method="linear", limit_direction="backward", inplace=True
            )
            data = df.to_numpy().reshape(n_states, n_h, n_y, n_x, self._n_dvars)
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
            print("INPUT VARS: (state, heights, y, x)")
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
        del pts, x, y, h, gvars

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
            if v != FV.WEIGHT and v not in out:
                if v in self._dkys:
                    out[v] = data[..., self._dkys[v]]
                else:
                    out[v] = np.full(
                        (n_states, n_pts), self.fixed_vars[v], dtype=config.dtype_double
                    )

        # add weights:
        if weights is not None:
            tdata[FV.WEIGHT] = weights[:, None, None]
        else:
            tdata[FV.WEIGHT] = np.full((mdata.n_states, 1, 1), 1/self._N, dtype=config.dtype_double)
        tdata.dims[FV.WEIGHT] = (FC.STATE, FC.TARGET, FC.TPOINT)

        return {v: d.reshape(n_states, n_targets, n_tpoints) for v, d in out.items()}
