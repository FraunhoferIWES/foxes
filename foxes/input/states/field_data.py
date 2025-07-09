import numpy as np
import pandas as pd
import xarray as xr
from copy import copy, deepcopy
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
        assert min(data.sizes.values()) > 0, (
            f"States: No data in file {fpath}, isel={isel}, sel={sel}, resulting sizes={data.sizes}"
        )
        return data


class FieldData(States):
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
    interpn_pars: dict, optional
        Additional parameters for scipy.interpolate.interpn
    bounds_extra_space: float or str
        The extra space, either float in m,
        or str for units of D, e.g. '2.5D'
    weight_factor: float
        The factor to multiply the weights with

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
        bounds_extra_space=1000,
        weight_factor=None,
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
        bounds_extra_space: float or str, optional
            The extra space, either float in m,
            or str for units of D, e.g. '2.5D'
        weight_factor: float, optional
            The factor to multiply the weights with
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
        self.bounds_extra_space = bounds_extra_space
        self.var2ncvar = var2ncvar
        self.weight_factor = weight_factor

        assert FV.WEIGHT not in output_vars, (
            f"States '{self.name}': Cannot have '{FV.WEIGHT}' as output variable, got {output_vars}"
        )
        self.variables = [v for v in output_vars if v not in fixed_vars]
        if weight_ncvar is not None:
            self.var2ncvar[FV.WEIGHT] = weight_ncvar
            self.variables.append(FV.WEIGHT)
        elif FV.WEIGHT in var2ncvar:
            raise KeyError(
                f"States '{self.name}': Cannot have '{FV.WEIGHT}' in var2ncvar, use weight_ncvar instead"
            )

        self._N = None
        self._inds = None
        self.__data_source = data_source

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
    
    def _read_ds(self, ds, variables, verbosity=0):
        """
        Extract data from the original Dataset.

        Parameters
        ----------
        ds: xarray.Dataset
            The Dataset to read data from
        variables: list of str
            The variables to extract from the Dataset
        verbosity: int
            The verbosity level, 0 = silent
        
        Returns
        -------
        s: numpy.ndarray
            The states coordinate values
        x: numpy.ndarray
            The x coordinate values
        y: numpy.ndarray
            The y coordinate values
        h: numpy.ndarray or None
            The height coordinate values, or None if not available
        data: dict
            The extracted data, keys are variable names,
            values are tuples (dims, data_array)
            where dims is a tuple of dimension names and
            data_array is a numpy.ndarray with the data values

        """
        shp_s = (self.states_coord,)
        shp_xy = (self.x_coord, self.y_coord)
        shp_h = (self.h_coord,) if self.h_coord is not None else None
        shp_xyh = (self.x_coord, self.y_coord, self.h_coord) if self.h_coord is not None else None
        shp_sh = (self.states_coord, self.h_coord) if self.h_coord is not None else None
        shp_sxy = (self.states_coord, self.x_coord, self.y_coord)
        shp_sxyh = (self.states_coord, self.x_coord, self.y_coord, self.h_coord) if self.h_coord is not None else None
        shps = [shp_s, shp_xy, shp_h, shp_xyh, shp_sh, shp_sxy, shp_sxyh]
        shps = [s for s in shps if s is not None]
        shpso = [sorted(s) for s in shps]

        cmap = {
            self.states_coord: FC.STATE,
            self.x_coord: FV.X,
            self.y_coord: FV.Y,
            self.h_coord: FV.H
        }

        data = {}
        for v in variables:
            w = self.var2ncvar.get(v, v)
            if v in self.fixed_vars:
                continue
            if w not in ds.data_vars:
                raise KeyError(f"States '{self.name}': Missing data variable '{w}' in Dataset, got '{list(ds.data_vars.keys())}'")
            
            d = ds[w]
            dims = tuple([cmap[c] for c in d.dims])
            if d.dims in shps:
                data[v] = (dims, d.to_numpy())
            elif sorted(d.dims) in shpso:
                s = shps[shpso.index(sorted(d.dims))]
                i = [d.dims.index(c) for c in s]
                s = tuple([cmap[c] for c in s])
                data[v] = (s, np.moveaxis(d.to_numpy(), i, np.arange(len(i))))
            else:
                raise ValueError(f"States '{self.name}': Failed to map variable '{w}' with dimensions {d.dims} to expected dimensions {shps}")

        if self.weight_factor is not None and FV.WEIGHT in data:
            data[FV.WEIGHT][1][:] *= self.weight_factor

        s = ds[self.states_coord].to_numpy() if self.states_coord in ds else None
        x = ds[self.x_coord].to_numpy()
        y = ds[self.y_coord].to_numpy()
        h = ds[self.h_coord].to_numpy() if self.h_coord is not None else None

        if verbosity > 1 and data is not None:
            print(f"\n{self.name}: Data ranges")
            for v, d in data.items():
                nn = np.sum(np.isnan(d))
                print(
                    f"  {v}: {np.nanmin(d)} --> {np.nanmax(d)}, nans: {nn} ({100 * nn / len(d.flat):.2f}%)"
                )

        return s, x, y, h, data

    def _get_data(self, ds, variables, verbosity=0):
        """
        Gets the data from the Dataset and prepares it for calculations.

        Parameters
        ----------
        ds: xarray.Dataset
            The Dataset to read data from
        variables: list of str
            The variables to extract from the Dataset
        verbosity: int
            The verbosity level, 0 = silent 

        Returns
        -------
        s: numpy.ndarray
            The states coordinate values
        x: numpy.ndarray
            The x coordinate values
        y: numpy.ndarray
            The y coordinate values
        h: numpy.ndarray or None
            The height coordinate values, or None if not available
        data: dict
            The extracted data, keys are dimension tuples,
            values are tuples (DATA key, variables, data_array)     
            where DATA key is the name in the mdata object,
            variables is a list of variable names, and
            data_array is a numpy.ndarray with the data values,
            the last dimension corresponds to the variables

        """
        s, x, y, h, data0 = self._read_ds(ds, variables, verbosity)

        weights = None
        if FV.WEIGHT in variables:
            if FV.WEIGHT not in data0:
                raise KeyError(
                    f"States '{self.name}': Missing weights variable '{self.weight_ncvar}' in data, found {sorted(list(data0.keys()))}"
                )
            elif data0[FV.WEIGHT][0] == (self.states_coord,):
                weights = data0.pop(FV.WEIGHT)[1]

        vmap = {
            FC.STATE: FC.STATE,
            FV.X: self.var(FV.X),
            FV.Y: self.var(FV.Y),
            FV.H: self.var(FV.H)
        }

        data = {} # dim: [DATA key, variables, data array]
        for v, (dims, d) in data0.items():
            if dims not in data:
                i = len(data)
                data[dims] = [self.var(f"data{i}"), [], []]
            data[dims][1].append(v)
            data[dims][2].append(d)
        for dims in data.keys():
            data[dims][2] = np.stack(data[dims][2], axis=-1)
        data = {
            tuple([vmap[c] for c in dims] + [self.var(f"vars{i}")]): d
            for i, (dims, d) in enumerate(data.items())
        }

        return s, x, y, h, data, weights

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
            vars = [self.var2ncvar.get(v, v) for v in self.variables]
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
                self._inds = self.__data_source[self.states_coord].to_numpy()
                self._N = len(self._inds)

            elif self.load_mode == "fly":
                self._inds = self.__data_source
                self.__data_source = fpath
                self._files_maxi = {f: len(inds) for f, inds in zip(files, self._inds)}
                self._inds = np.concatenate(self._inds, axis=0)
                self._N = len(self._inds)

            else:
                raise KeyError(
                    f"States '{self.name}': Unknown load_mode '{self.load_mode}', choices: preload, lazy, fly"
                )

            if self.time_format is not None:
                self._inds = pd.to_datetime(
                    self._inds, format=self.time_format
                ).to_numpy()

        # given data is already Dataset:
        else:
            self._inds = self.data_source[self.states_coord].to_numpy()
            self._N = len(self._inds)

        idata = super().load_data(algo, verbosity)

        if self.load_mode == "preload":

            s, self._x, self._y, self._h, data, w = self._get_data(
                self.data_source, self.variables, verbosity)

            if s is not None:
                idata["coords"][FC.STATE] = s
            if w is not None:
                idata["data_vars"][FV.WEIGHT] = ((FC.STATE,), w)

            self._data_state_keys = []
            self._data_nostate = {}
            for dims, d in data.items():
                if FC.STATE in dims:
                    self._data_state_keys.append(d[0])
                    idata["coords"][dims[-1]] = d[1]
                    idata["data_vars"][d[0]] = (dims, d[2])
                else:
                    self._data_nostate[dims] = (d[1], d[2])
            del data

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
            inds=self._inds,
        )
        del self._inds

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
        self._inds = data.pop("inds")

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
        return self._inds

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
            x = self._x
            y = self._y
            h = self._h
            weights = mdata[FV.WEIGHT] if FV.WEIGHT in mdata else None
            data = deepcopy(self._data_nostate)
            for DATA in self._data_state_keys:
                dims = mdata.dims[DATA]
                vrs = mdata[dims[-1]].tolist()
                data[dims] = (vrs, mdata[DATA].copy())

        # case lazy:
        elif self.load_mode == "lazy":
            i0 = mdata.states_i0(counter=True)
            s = slice(i0, i0 + n_states)
            ds = self.data_source.isel({self.states_coord: s}).load()
            __, x, y, h, data, weights = self._get_data(
                ds, self.variables, verbosity=0)
            del ds
            data = {dims: (d[1], d[2]) for dims, d in data.items()}

        # case fly:
        elif self.load_mode == "fly":
            vars = [self.var2ncvar.get(v, v) for v in self.variables]
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

            assert i0 == i1, (
                f"States '{self.name}': Missing states for load_mode '{self.load_mode}': (i0, i1) = {(i0, i1)}"
            )

            data = xr.concat(data, dim=self.states_coord)
            __, x, y, h, data, weights = self._get_data(
                data, self.variables, verbosity=0)
            data = {dims: (d[1], d[2]) for dims, d in data.items()}

        else:
            raise KeyError(
                f"States '{self.name}': Unknown load_mode '{self.load_mode}', choices: preload, lazy, fly"
            )
        
        # interpolate data to points:
        out = {}
        gmap = {
            FC.STATE: np.arange(n_states),
            self.var(FV.X): x,
            self.var(FV.Y): y,
            self.var(FV.H): h,
        }
        for dims, (vrs, data) in data.items():

            # translate (WD, WS) to (U, V):
            if FV.WD in vrs or FV.WS in vrs:
                assert FV.WD in vrs and (FV.WS in vrs or FV.WS in self.fixed_vars), (
                    f"States '{self.name}': Missing '{FV.WD}' or '{FV.WS}' in data variables {vrs} for dimensions {dims}"
                )
                iwd = vrs.index(FV.WD)
                iws = vrs.index(FV.WS)
                ws = (
                    data[..., iws]
                    if FV.WS in vrs
                    else self.fixed_vars[FV.WS]
                )
                data[..., [iwd, iws]] = wd2uv(data[..., iwd], ws, axis=-1)
                del ws

            # prepare grid:
            idims = dims[:-1]
            gvars = tuple([gmap[c] for c in idims])
            
            # prepare points:
            n_vrs = len(vrs)
            tdims = [n_states, n_pts, n_vrs]
            if idims == (FC.STATE, self.var(FV.X), self.var(FV.Y), self.var(FV.H)):
                pts = np.append(
                    np.zeros((n_states, n_pts, 1), dtype=config.dtype_double), 
                    points, 
                    axis=2,
                )
                pts[..., 0] = np.arange(n_states)[:, None]
                pts = pts.reshape(n_states * n_pts, 4)
            elif idims == (FC.STATE, self.var(FV.X), self.var(FV.Y)):
                pts = np.append(
                    np.zeros((n_states, n_pts, 1), dtype=config.dtype_double), 
                    points[..., :2], 
                    axis=2,
                )
                pts[..., 0] = np.arange(n_states)[:, None]
                pts = pts.reshape(n_states * n_pts, 3)
            elif idims == (FC.STATE, self.var(FV.H)):
                pts = np.append(
                    np.zeros((n_states, n_pts, 1), dtype=config.dtype_double), 
                    points[..., 2, None], 
                    axis=2,
                )
                pts[..., 0] = np.arange(n_states)[:, None]
                pts = pts.reshape(n_states * n_pts, 2)
            elif idims == (FC.STATE,):
                if FV.WD in vrs:
                    uv = data[..., [iwd, iws]]
                    data[..., iwd] = uv2wd(uv)
                    data[..., iws] = np.linalg.norm(uv, axis=-1)
                    del uv
                for i, v in enumerate(vrs):
                    if v in self.ovars:
                        out[v] = np.zeros((n_states, n_pts), dtype=config.dtype_double)
                        out[v][:] = data[:, None, i]
                continue
            elif idims == (self.var(FV.X), self.var(FV.Y), self.var(FV.H)):
                pts = points[0]
                tdims = (1, n_pts, n_vrs)
            elif idims == (self.var(FV.X), self.var(FV.Y)):
                pts = points[0][:, :2]
                tdims = (1, n_pts, n_vrs)
            elif idims == (self.var(FV.H),):
                pts = points[0][:, 2]
                tdims = (1, n_pts, n_vrs)
            else:
                raise ValueError(f"States '{self.name}': Unsupported dimensions {dims} for variables {vrs}")

            # interpolate:
            try:
                ipars = dict(bounds_error=True, fill_value=None)
                ipars.update(self.interpn_pars)
                data = interpn(gvars, data, pts, **ipars).reshape(tdims)
            except ValueError as e:
                print(f"\nStates '{self.name}': Interpolation error")
                print("INPUT VARS: (state, x, y, height)")
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
            del pts, gvars

            # translate (U, V) into (WD, WS):
            if FV.WD in vrs:
                uv = data[..., [iwd, iws]]
                data[..., iwd] = uv2wd(uv)
                data[..., iws] = np.linalg.norm(uv, axis=-1)
                del uv
            
            # broadcast if needed:
            if tdims != (n_states, n_pts, n_vrs):
                tmp = data
                data = np.zeros((n_states, n_pts, n_vrs), dtype=config.dtype_double)
                data[:] = tmp
                del tmp

            # set output:
            for i, v in enumerate(vrs):
                if v in self.ovars:
                    out[v] = data[..., i]

        # set fixed variables:
        for v, d in self.fixed_vars.items():
            out[v] = np.full(
                (n_states, n_pts), d, dtype=config.dtype_double
            )

        # add weights:
        if weights is not None:
            tdata[FV.WEIGHT] = weights[:, None, None]
        elif FV.WEIGHT in out:
            tdata[FV.WEIGHT] = out.pop(FV.WEIGHT).reshape(n_states, n_targets, n_tpoints)
        else:
            tdata[FV.WEIGHT] = np.full(
                (mdata.n_states, 1, 1), 1 / self._N, dtype=config.dtype_double
            )
        tdata.dims[FV.WEIGHT] = (FC.STATE, FC.TARGET, FC.TPOINT)

        return {v: d.reshape(n_states, n_targets, n_tpoints) for v, d in out.items()}
