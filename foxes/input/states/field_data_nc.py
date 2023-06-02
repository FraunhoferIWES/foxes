import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from scipy.interpolate import RegularGridInterpolator
from copy import deepcopy

from foxes.core import States
from foxes.utils import wd2uv, uv2wd
from foxes.data import STATES, StaticData
import foxes.variables as FV
import foxes.constants as FC


class FieldDataNC(States):
    """
    Heterogeneous ambient states on a regular
    horizontal grid in NetCDF format.

    Parameters
    ----------
    data_source : str or xarray.Dataset
        The data or the file search pattern, should end with
        suffix '.nc'. One or many files.
    output_vars : list of str
        The output variables
    var2ncvar : dict, optional
        Mapping from variable names to variable names
        in the nc file
    fixed_vars : dict, optional
        Uniform values for output variables, instead
        of reading from data
    states_coord : str
        The states coordinate name in the data
    x_coord : str
        The x coordinate name in the data
    y_coord : str
        The y coordinate name in the data
    h_coord : str
        The height coordinate name in the data
    pre_load : bool
        Flag for loading all data into memory during
        initialization
    weight_ncvar : str, optional
        Name of the weight data variable in the nc file(s)
    bounds_error : bool
        Flag for raising errors if bounds are exceeded
    fill_value : number, optional
        Fill value in case of exceeding bounds, if no bounds error
    time_format : str
        The datetime parsing format string
    sel : dict, optional
        Subset selection via xr.Dataset.sel()
    verbosity : int
        Verbosity level for pre_load file reading

    Attributes
    ----------
    data_source : str or xarray.Dataset
        The data or the file search pattern, should end with
        suffix '.nc'. One or many files.
    ovars : list of str
        The output variables
    var2ncvar : dict
        Mapping from variable names to variable names
        in the nc file
    fixed_vars : dict
        Uniform values for output variables, instead
        of reading from data
    states_coord : str
        The states coordinate name in the data
    x_coord : str
        The x coordinate name in the data
    y_coord : str
        The y coordinate name in the data
    h_coord : str
        The height coordinate name in the data
    pre_load : bool
        Flag for loading all data into memory during
        initialization
    weight_ncvar : str
        Name of the weight data variable in the nc file(s)
    bounds_error : bool
        Flag for raising errors if bounds are exceeded
    fill_value : number
        Fill value in case of exceeding bounds, if no bounds error
    time_format : str
        The datetime parsing format string
    sel : dict
        Subset selection via xr.Dataset.sel()

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
        pre_load=True,
        weight_ncvar=None,
        bounds_error=True,
        fill_value=None,
        time_format="%Y-%m-%d_%H:%M:%S",
        sel=None,
        verbosity=1,
    ):
        super().__init__()

        self.data_source = data_source
        self.states_coord = states_coord
        self.ovars = output_vars
        self.fixed_vars = fixed_vars
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.h_coord = h_coord
        self.weight_ncvar = weight_ncvar
        self.pre_load = pre_load
        self.bounds_error = bounds_error
        self.fill_value = fill_value
        self.time_format = time_format
        self.sel = sel

        self.var2ncvar = {
            v: var2ncvar.get(v, v) for v in output_vars if v not in fixed_vars
        }

        self._inds = None
        self._N = None
        self._weights = None

        # pre-load file reading, usually prior to DaskRunner:
        if not isinstance(self.data_source, xr.Dataset):
            if "*" in str(self.data_source):
                pass
            else:
                self.data_source = StaticData().get_file_path(
                    STATES, self.data_source, check_raw=True
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

            with xr.open_mfdataset(
                str(self.data_source),
                parallel=False,
                concat_dim=self.states_coord,
                combine="nested",
                data_vars="minimal",
                coords="minimal",
                compat="override",
            ) as ds:
                dss = ds if self.sel is None else ds.sel(self.sel)
                if pre_load:
                    self.data_source = dss.load()
                else:
                    self.data_source = dss
                self._get_inds(dss)

    def _get_inds(self, ds):
        """
        Helper function for index and weights
        reading
        """
        for c in [self.states_coord, self.x_coord, self.y_coord, self.h_coord]:
            if not c in ds:
                raise KeyError(
                    f"States '{self.name}': Missing coordinate '{c}' in data"
                )

        self._inds = ds[self.states_coord].to_numpy()
        if self.time_format is not None:
            self._inds = pd.to_datetime(self._inds, format=self.time_format).to_numpy()
        self._N = len(self._inds)

        if self.weight_ncvar is not None:
            self._weights = ds[self.weight_ncvar].to_numpy()

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
        h = ds[self.h_coord].to_numpy()
        n_x = len(x)
        n_y = len(y)
        n_h = len(h)
        n_sts = ds.sizes[self.states_coord]

        cor_shxy = (self.states_coord, self.h_coord, self.x_coord, self.y_coord)
        cor_shyx = (self.states_coord, self.h_coord, self.y_coord, self.x_coord)
        cor_sh = (self.states_coord, self.h_coord)
        cor_s = (self.states_coord,)
        vars_shyx = []
        vars_sh = []
        vars_s = []
        for v, ncv in self.var2ncvar.items():
            if ds[ncv].dims == cor_shyx or ds[ncv].dims == cor_shxy:
                vars_shyx.append(v)
            elif ds[ncv].dims == cor_sh:
                vars_sh.append(v)
            elif ds[ncv].dims == cor_s:
                vars_s.append(v)
            else:
                raise ValueError(
                    f"States '{self.name}': Wrong coordinate order for variable '{ncv}': Found {ds[ncv].dims}, expecting {cor_shxy}, {cor_shyx}, {cor_sh} or {cor_s}"
                )

        data = np.zeros((n_sts, n_h, n_y, n_x, len(self.var2ncvar)), dtype=FC.DTYPE)
        for v in vars_shyx:
            ncv = self.var2ncvar[v]
            if ds[ncv].dims == cor_shyx:
                data[..., self._dkys[v]] = ds[ncv][:]
            else:
                data[..., self._dkys[v]] = np.swapaxes(ds[ncv].to_numpy(), 2, 3)
        for v in vars_sh:
            ncv = self.var2ncvar[v]
            data[..., self._dkys[v]] = ds[ncv].to_numpy()[:, :, None, None]
        for v in vars_s:
            ncv = self.var2ncvar[v]
            data[..., self._dkys[v]] = ds[ncv].to_numpy()[:, None, None, None]
        if FV.WD in self.fixed_vars:
            data[..., self._dkys[FV.WD]] = np.full(
                (n_sts, n_h, n_y, n_x), self.fixed_vars[FV.WD], dtype=FC.DTYPE
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

    def initialize(self, algo, verbosity=0):
        """
        Initializes the model.

        This includes loading all required data from files. The model
        should return all array type data as part of the idata return
        dictionary (and not store it under self, for memory reasons). This
        data will then be chunked and provided as part of the mdata object
        during calculations.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        idata : dict
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

        if self._weights is None:
            self._weights = np.full(
                (self._N, algo.n_turbines), 1.0 / self._N, dtype=FC.DTYPE
            )

        idata = super().initialize(algo, verbosity)
        self._update_idata(algo, idata)

        if self.pre_load:
            self.X = self.var(FV.X)
            self.Y = self.var(FV.Y)
            self.H = self.var(FV.H)
            self.VARS = self.var("vars")
            self.DATA = self.var("data")

            ds = self.data_source

            h = ds[self.h_coord].to_numpy()
            y = ds[self.y_coord].to_numpy()
            x = ds[self.x_coord].to_numpy()
            v = list(self._dkys.keys())
            coos = (FC.STATE, self.H, self.Y, self.X, self.VARS)
            data = self._get_data(ds, verbosity)
            data = (coos, data)

            idata["coords"][self.H] = h
            idata["coords"][self.Y] = y
            idata["coords"][self.X] = x
            idata["coords"][self.VARS] = v
            idata["data_vars"][self.DATA] = data

        return idata

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
        indices : array_like
            The index labels of states, or None for default integers

        """
        return self._inds

    def output_point_vars(self, algo):
        """
        The variables which are being modified by the model.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        output_vars : list of str
            The output variable names

        """
        return self.ovars

    def weights(self, algo):
        """
        The statistical weights of all states.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm

        Returns
        -------
        weights : numpy.ndarray
            The weights, shape: (n_states, n_turbines)

        """
        return self._weights

    def calculate(self, algo, mdata, fdata, pdata):
        """ "
        The main model calculation.

        This function is executed on a single chunk of data,
        all computations should be based on numpy arrays.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
        pdata : foxes.core.Data
            The point data

        Returns
        -------
        results : dict
            The resulting data, keys: output variable str.
            Values: numpy.ndarray with shape (n_states, n_points)

        """
        # prepare:
        points = pdata[FC.POINTS]
        n_pts = points.shape[1]
        n_states = fdata.n_states

        # pick pre-loaded data:
        if self.pre_load:
            x = mdata[self.X]
            y = mdata[self.Y]
            h = mdata[self.H]
            data = mdata[self.DATA].copy()

        # read data for this chunk:
        else:
            i0 = np.where(self._inds == mdata[FC.STATE][0])[0][0]
            s = slice(i0, i0 + n_states)
            ds = self.data_source.isel({self.states_coord: s}).load()

            x = ds[self.x_coord].to_numpy()
            y = ds[self.y_coord].to_numpy()
            h = ds[self.h_coord].to_numpy()
            data = self._get_data(ds, verbosity=0)

            del ds

        # translate WS, WD into U, V:
        if FV.WD in self.ovars and FV.WS in self.ovars:
            wd = data[..., self._dkys[FV.WD]]
            ws = (
                data[..., self._dkys[FV.WS]]
                if FV.WS in self._dkys
                else self.fixed_vars[FV.WS]
            )
            data[..., :2] = wd2uv(wd, ws, axis=-1)
            del ws, wd

        # prepare points:
        sts = np.arange(n_states)
        pts = np.append(points, np.zeros((n_states, n_pts, 1), dtype=FC.DTYPE), axis=2)
        pts[:, :, 3] = sts[:, None]
        pts = pts.reshape(n_states * n_pts, 4)
        pts = np.flip(pts, axis=1)

        # interpolate:
        gvars = (sts, h, y, x)
        iterp = RegularGridInterpolator(
            gvars, data, bounds_error=self.bounds_error, fill_value=self.fill_value
        )
        try:
            data = iterp(pts).reshape(n_states, n_pts, self._n_dvars)
        except ValueError as e:
            print(f"\n\nStates '{self.name}': Interpolation error")
            print("INPUT VARS : (state, heights, y, x)")
            print(
                "DATA BOUNDS:", [np.min(d) for d in gvars], [np.max(d) for d in gvars]
            )
            print(
                "EVAL BOUNDS:", [np.min(p) for p in pts.T], [np.max(p) for p in pts.T]
            )
            raise e
        del pts, iterp, x, y, h, gvars

        # set output:
        out = {}
        if FV.WD in self.ovars and FV.WS in self.ovars:
            uv = data[..., :2]
            out[FV.WS] = np.linalg.norm(uv, axis=-1)
            out[FV.WD] = uv2wd(uv, axis=-1)
            del uv
        for v in self.ovars:
            if v not in out:
                if v in self._dkys:
                    out[v] = data[..., self._dkys[v]]
                else:
                    out[v] = np.full(
                        (n_states, n_pts), self.fixed_vars[v], dtype=FC.DTYPE
                    )

        return out
