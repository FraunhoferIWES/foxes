import numpy as np
from scipy.interpolate import interpn

from foxes.utils import wd2uv, uv2wd, weibull_weights
from foxes.config import config
import foxes.variables as FV
import foxes.constants as FC

from .dataset_states import DatasetStates


class FieldData(DatasetStates):
    """
    Heterogeneous ambient states on a regular
    horizontal grid in NetCDF format.

    Attributes
    ----------
    states_coord: str
        The states coordinate name in the data
    x_coord: str
        The x coordinate name in the data
    y_coord: str
        The y coordinate name in the data
    h_coord: str
        The height coordinate name in the data
    weight_ncvar: str
        Name of the weight data variable in the nc file(s)
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
        *args,
        states_coord="Time",
        x_coord="UTMX",
        y_coord="UTMY",
        h_coord="height",
        weight_ncvar=None,
        bounds_extra_space=1000,
        weight_factor=None,
        interpn_pars={},
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        args: tuple, optional
            Arguments for the base class
        states_coord: str
            The states coordinate name in the data
        x_coord: str
            The x coordinate name in the data
        y_coord: str
            The y coordinate name in the data
        h_coord: str, optional
            The height coordinate name in the data
        weight_ncvar: str, optional
            Name of the weight data variable in the nc file(s)
        bounds_extra_space: float or str, optional
            The extra space, either float in m,
            or str for units of D, e.g. '2.5D'
        weight_factor: float, optional
            The factor to multiply the weights with
        interpn_pars: dict
            Parameters for scipy.interpolate.interpn
        kwargs: dict, optional
            Additional parameters for the base class

        """
        super().__init__(*args, **kwargs)

        self.states_coord = states_coord
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.h_coord = h_coord
        self.weight_ncvar = weight_ncvar
        self.interpn_pars = interpn_pars
        self.bounds_extra_space = bounds_extra_space
        self.weight_factor = weight_factor

        assert FV.WEIGHT not in self.ovars, (
            f"States '{self.name}': Cannot have '{FV.WEIGHT}' as output variable, got {self.ovars}"
        )
        self.variables = [v for v in self.ovars if v not in self.fixed_vars]
        if weight_ncvar is not None:
            self.var2ncvar[FV.WEIGHT] = weight_ncvar
            self.variables.append(FV.WEIGHT)
        elif FV.WEIGHT in self.var2ncvar:
            raise KeyError(
                f"States '{self.name}': Cannot have '{FV.WEIGHT}' in var2ncvar, use weight_ncvar instead"
            )
        
        if bounds_extra_space is not None:
            self._filter_xy = dict(
                x_coord=self.x_coord,
                y_coord=self.y_coord,
                bounds_extra_space=self.bounds_extra_space,
            )
        else:
            self._filter_xy = None
    
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
        gpts: tuple
            Grid point data extracted from the Dataset, like
            (x, y, h) or similar
        data: dict
            The extracted data, keys are dimension tuples,
            values are tuples (DATA key, variables, data_array)     
            where DATA key is the name in the mdata object,
            variables is a list of variable names, and
            data_array is a numpy.ndarray with the data values,
            the last dimension corresponds to the variables
        weights: numpy.ndarray or None
            The weights array, if only state dependent, otherwise
            weights are among data. Shape: (n_states,)

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

        return s, (x, y, h), data, weights

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
        return super().load_data(
            algo, 
            coords=[self.states_coord, self.h_coord, self.y_coord, self.x_coord],
            variables=self.variables,
            filter_xy=self._filter_xy,
            verbosity=verbosity,
        )

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
        # prepare
        n_states = tdata.n_states
        n_targets = tdata.n_targets
        n_tpoints = tdata.n_tpoints
        points = tdata[FC.TARGETS].reshape(n_states, n_targets * n_tpoints, 3)
        n_pts = points.shape[1]
        n_states = fdata.n_states
        coords = [self.states_coord, self.h_coord, self.y_coord, self.x_coord]

        # get data for calculation
        (x, y, h), data, weights = self.get_calc_data(mdata, coords, self.variables)
        
        # interpolate data to points:
        out = {}
        gmap = {
            FC.STATE: np.arange(n_states),
            self.var(FV.X): x,
            self.var(FV.Y): y,
            self.var(FV.H): h,
        }
        for dims, (vrs, d) in data.items():

            # translate (WD, WS) to (U, V):
            if FV.WD in vrs or FV.WS in vrs:
                assert FV.WD in vrs and (FV.WS in vrs or FV.WS in self.fixed_vars), (
                    f"States '{self.name}': Missing '{FV.WD}' or '{FV.WS}' in data variables {vrs} for dimensions {dims}"
                )
                iwd = vrs.index(FV.WD)
                iws = vrs.index(FV.WS)
                ws = (
                    d[..., iws]
                    if FV.WS in vrs
                    else self.fixed_vars[FV.WS]
                )
                d[..., [iwd, iws]] = wd2uv(d[..., iwd], ws, axis=-1)
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
                    uv = d[..., [iwd, iws]]
                    d[..., iwd] = uv2wd(uv)
                    d[..., iws] = np.linalg.norm(uv, axis=-1)
                    del uv
                for i, v in enumerate(vrs):
                    if v in self.ovars:
                        out[v] = np.zeros((n_states, n_pts), dtype=config.dtype_double)
                        out[v][:] = d[:, None, i]
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
                d = interpn(gvars, d, pts, **ipars).reshape(tdims)
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
                uv = d[..., [iwd, iws]]
                d[..., iwd] = uv2wd(uv)
                d[..., iws] = np.linalg.norm(uv, axis=-1)
                del uv
            
            # broadcast if needed:
            if tdims != (n_states, n_pts, n_vrs):
                tmp = d
                d = np.zeros((n_states, n_pts, n_vrs), dtype=config.dtype_double)
                d[:] = tmp
                del tmp

            # set output:
            for i, v in enumerate(vrs):
                if v in self.ovars:
                    out[v] = d[..., i]

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


class WeibullField(FieldData):
    """
    Weibull sectors at regular grid points

    Attributes
    ----------
    wd_coord: str
        The wind direction coordinate name
    ws_coord: str
        The wind speed coordinate name, if wind speed bin
        centres are in data, else None
    ws_bins: numpy.ndarray
        The wind speed bins, including
        lower and upper bounds, shape: (n_ws_bins+1,)

    :group: input.states

    """

    def __init__(
        self,
        *args,
        wd_coord,
        ws_coord=None,
        ws_bins=None,
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        args: tuple, optional
            Positional arguments for the base class
        wd_coord: str
            The wind direction coordinate name
        ws_coord: str, optional
            The wind speed coordinate name, if wind speed bin
            centres are in data
        ws_bins: list of float, optional
            The wind speed bins, including
            lower and upper bounds
        kwargs: dict, optional
            Keyword arguments for the base class

        """
        super().__init__(
            *args,
            states_coord=wd_coord, 
            time_format=None, 
            **kwargs,
        )
        self.ws_bins = None if ws_bins is None else np.sort(np.asarray(ws_bins))
        self.ws_coord = ws_coord

        assert ws_coord is None or ws_bins is None, (
            f"States '{self.name}': Cannot have both ws_coord '{ws_coord}' and ws_bins {ws_bins}"
        )
        assert ws_coord is not None or ws_bins is not None, (
            f"States '{self.name}': Expecting either ws_coord or ws_bins"
        )

        if FV.WD not in self.ovars:
            raise ValueError(
                f"States '{self.name}': Expecting output variable '{FV.WD}', got {self.ovars}"
            )
        for v in [FV.WEIBULL_A, FV.WEIBULL_k, FV.WEIGHT]:
            if v in self.ovars:
                raise ValueError(
                    f"States '{self.name}': Cannot have '{v}' as output variable"
                )
            if v not in self.variables:
                self.variables.append(v)

        for v in [FV.WS, FV.WD]:
            if v in self.variables:
                self.variables.remove(v)

        self._n_wd = None
        self._n_ws = None

    def __repr__(self):
        return f"{type(self).__name__}(n_wd={self._n_wd}, n_ws={self._n_ws})"

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
        # check for ws_coord
        if self.ws_coord is not None:
            assert self.ws_coord in ds.coords, (
                f"States '{self.name}': Expecting ws_coord '{self.ws_coord}' to be among data coordinates, got {list(ds.coords.keys())}"
            )
            for v, d in ds.data_vars.items():
                if self.ws_coord in d.dims:
                    raise NotImplementedError(
                        f"States '{self.name}': Cannot handle variable '{v}' with dimension '{self.ws_coord}' in data, dims = {d.dims}"
                    )
        
        # read data, using wd_coord as state coordinate
        wd, x, y, h, data0 = super()._read_ds(ds, variables, verbosity)

        # replace state by wd coordinate
        data0 = {
            v: (tuple({FC.STATE: FV.WD}.get(c, c) for c in dims), d) 
            for v, (dims, d) in data0.items()
        }

        # check weights
        if FV.WEIGHT not in data0:
            raise KeyError(
                f"States '{self.name}': Missing weights variable '{FV.WEIGHT}' in data, found {sorted(list(data0.keys()))}"
            )
        else:
            dims = data0[FV.WEIGHT][0]
            if FV.WD not in dims:
                raise KeyError(
                    f"States '{self.name}': Expecting weights variable '{FV.WEIGHT}' to contain dimension '{FV.WD}', got {dims}"
                )
            if FV.WS in dims:
                raise KeyError(
                    f"States '{self.name}': Expecting weights variable '{FV.WEIGHT}' to not contain dimension '{FV.WS}', got {dims}"
                )

        # construct wind speed bins and bin deltas
        assert FV.WS not in data0, (
            f"States '{self.name}': Cannot have '{FV.WS}' in data, found variables {list(data0.keys())}"
        )
        if self.ws_bins is not None:
            wsb = self.ws_bins
            wss = 0.5 * (wsb[:-1] + wsb[1:])
        elif self.ws_coord in ds.coords:
            wss = ds[self.ws_coord].to_numpy()
            wsb = np.zeros((len(wss) + 1,), dtype=config.dtype_double)
            wsb[1:-1] = 0.5 * (wss[1:] + wss[:-1])
            wsb[0] = wss[0] - 0.5 * wsb[1]
            wsb[-1] = wss[-1] + 0.5 * wsb[-2]
            self.ws_bins = wsb
        else:
            raise ValueError(
                f"States '{self.name}': Expecting ws_bins argument, or '{self.ws_coord}' among data coordinates, got {list(ds.coords.keys())}"
            )
        wsd = wsb[1:] - wsb[:-1]
        n_ws = len(wss)
        n_wd = len(wd)
        del wsb, ds

        # calculate Weibull weights
        dms = [FV.WS, FV.WD]
        shp = [n_ws, n_wd]
        for c in [FV.X, FV.Y, FV.H]:
            for v in [FV.WEIBULL_A, FV.WEIBULL_k]:
                if c in data0[v][0]:
                    dms.append(c)
                    shp.append(data0[v][1].shape[data0[v][0].index(c)])
                    break
        dms = tuple(dms)
        shp = tuple(shp)
        if data0[FV.WEIGHT][0] == dms:
            w = data0.pop(FV.WEIGHT)[1]
        else:
            s_w = tuple([np.s_[:] if c in data0[FV.WEIGHT][0] else None for c in dms])
            w = np.zeros(shp, dtype=config.dtype_double)
            w[:] = data0.pop(FV.WEIGHT)[1][s_w]
        s_ws = tuple([np.s_[:], None] + [None] * (len(dms) - 2))
        s_A = tuple([np.s_[:] if c in data0[FV.WEIBULL_A][0] else None for c in dms])
        s_k = tuple([np.s_[:] if c in data0[FV.WEIBULL_A][0] else None for c in dms])
        data0[FV.WEIGHT] = (
            dms,
            w * weibull_weights(
                ws=wss[s_ws],
                ws_deltas=wsd[s_ws],
                A=data0.pop(FV.WEIBULL_A)[1][s_A],
                k=data0.pop(FV.WEIBULL_k)[1][s_k],
            )
        )
        del w, s_ws, s_A, s_k

        # translate binned data to states
        self._N = n_ws * n_wd
        self._inds = None
        data = {
            FV.WS: np.zeros((n_ws, n_wd), dtype=config.dtype_double),
            FV.WD: np.zeros((n_ws, n_wd), dtype=config.dtype_double),
        }
        data[FV.WS][:] = wss[:, None]
        data[FV.WD][:] = wd[None, :]
        data[FV.WS] = ((FC.STATE,), data[FV.WS].reshape(self._N))
        data[FV.WD] = ((FC.STATE,), data[FV.WD].reshape(self._N))
        for v in list(data0.keys()):
            dims, d = data0.pop(v)
            if dims[0] == FV.WD:
                dms = tuple([FC.STATE] + list(dims[1:]))
                shp = [n_ws] + list(d.shape)
                data[v] = np.zeros(shp, dtype=config.dtype_double)
                data[v][:] = d[None, ...]
                data[v] = (dms, data[v].reshape([self._N] + shp[2:]))
            elif len(dims) >=2 and dims[:2] == (FV.WS, FV.WD):
                dms = tuple([FC.STATE] + list(dims[2:]))
                shp = [self._N] + list(d.shape[2:])
                data[v] = (dms, d.reshape(shp))
            else:
                data[v] = (dims, d)
        data0 = data

        return None, x, y, h, data
    