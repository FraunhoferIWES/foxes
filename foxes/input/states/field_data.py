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
        
        self._cmap = {
            FC.STATE: self.states_coord,
            FV.X: self.x_coord,
            FV.Y: self.y_coord,
        }
        if self.h_coord is not None:
            self._cmap[FV.H] = self.h_coord   

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
            cmap=self._cmap,
            variables=self.variables,
            bounds_extra_space=self.bounds_extra_space,
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
        self.ensure_output_vars(algo, tdata)
        n_states = tdata.n_states
        n_targets = tdata.n_targets
        n_tpoints = tdata.n_tpoints
        points = tdata[FC.TARGETS].reshape(n_states, n_targets * n_tpoints, 3)
        n_pts = points.shape[1]

        # get data for calculation
        coords, data, weights = self.get_calc_data(mdata, self._cmap, self.variables)
        coords[FC.STATE] = np.arange(n_states, dtype=config.dtype_int)
        
        # interpolate data to points:
        out = {}
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
            gvars = tuple([coords[c] for c in idims])
            
            # prepare points:
            n_vrs = len(vrs)
            tdims = [n_states, n_pts, n_vrs]
            if idims == (FC.STATE, FV.X, FV.Y, FV.H):
                pts = np.append(
                    np.zeros((n_states, n_pts, 1), dtype=config.dtype_double), 
                    points, 
                    axis=2,
                )
                pts[..., 0] = np.arange(n_states)[:, None]
                pts = pts.reshape(n_states * n_pts, 4)
            elif idims == (FC.STATE, FV.X, FV.Y):
                pts = np.append(
                    np.zeros((n_states, n_pts, 1), dtype=config.dtype_double), 
                    points[..., :2], 
                    axis=2,
                )
                pts[..., 0] = np.arange(n_states)[:, None]
                pts = pts.reshape(n_states * n_pts, 3)
            elif idims == (FC.STATE, FV.H):
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
            elif idims == (FV.X, FV.Y, FV.H):
                pts = points[0]
                tdims = (1, n_pts, n_vrs)
            elif idims == (FV.X, FV.Y):
                pts = points[0][:, :2]
                tdims = (1, n_pts, n_vrs)
            elif idims == (FV.H,):
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
            load_mode="preload",
            **kwargs,
        )
        self.wd_coord = wd_coord
        self.ws_coord = ws_coord
        self.ws_bins = None if ws_bins is None else np.sort(np.asarray(ws_bins))

        assert ws_coord is None or ws_bins is None, (
            f"States '{self.name}': Cannot have both ws_coord '{ws_coord}' and ws_bins {ws_bins}"
        )
        assert ws_coord is not None or ws_bins is not None, (
            f"States '{self.name}': Expecting either ws_coord or ws_bins"
        )
        if ws_coord is not None:
            self._cmap[FV.WS] = ws_coord

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

    def _read_ds(self, ds, cmap, variables, verbosity=0):
        """
        Helper function for _get_data, extracts data from the original Dataset.

        Parameters
        ----------
        ds: xarray.Dataset
            The Dataset to read data from
        cmap: dict
            A mapping from foxes variable names to Dataset dimension names
        variables: list of str
            The variables to extract from the Dataset
        verbosity: int
            The verbosity level, 0 = silent
        
        Returns
        -------
        coords: dict
            keys: Foxes variable names, values: 1D coordinate value arrays
        data: dict
            The extracted data, keys are variable names,
            values are tuples (dims, data_array)
            where dims is a tuple of dimension names and
            data_array is a numpy.ndarray with the data values

        """       
        # read data, using wd_coord as state coordinate
        coords, data0 = super()._read_ds(ds, cmap, variables, verbosity)
        wd = coords.pop(FC.STATE)

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

        return coords, data
    