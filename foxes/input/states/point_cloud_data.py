import numpy as np
from scipy.interpolate import griddata

from foxes.config import config
from foxes.utils import weibull_weights
import foxes.variables as FV
import foxes.constants as FC

from .dataset_states import DatasetStates


class PointCloudData(DatasetStates):
    """
    Inflow data with point cloud support.

    Attributes
    ----------
    states_coord: str
        The states coordinate name in the data
    point_coord: str
        The point coordinate name in the data
    x_ncvar: str
        The x variable name in the data
    y_ncvar: str
        The y variable name in the data
    h_ncvar: str, optional
        The height variable name in the data
    weight_ncvar: str, optional
        The name of the weights variable in the data
    interp_pars: dict
        Additional arguments for the interpolation

    Examples
    --------
    Example of the NetCDF input files with point cloud data:

    >>>    Dimensions:  (point: 30, state: 100)
    >>>    Dimensions without coordinates: point, state
    >>>    Data variables:
    >>>        x        (point) float32 120B ...
    >>>        y        (point) float32 120B ...
    >>>        ws       (state, point) float32 12kB ...
    >>>        wd       (state, point) float32 12kB ...
    >>>        ti       (point) float32 120B ...
    >>>        rho      (state) float32 400B ...

    :group: input.states

    """

    def __init__(
        self,
        *args,
        states_coord="Time",
        point_coord="point",
        x_ncvar="x",
        y_ncvar="y",
        h_ncvar=None,
        weight_ncvar=None,
        interp_pars={},
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
        point_coord: str
            The point coordinate name in the data
        x_ncvar: str
            The x variable name in the data
        y_ncvar: str
            The y variable name in the data
        h_ncvar: str, optional
            The height variable name in the data
        weight_ncvar: str, optional
            The name of the weights variable in the data
        interp_pars: dict
            Additional arguments for the interpolation
        kwargs: dict, optional
            Additional parameters for the base class

        """
        super().__init__(*args, **kwargs)

        self.states_coord = states_coord
        self.point_coord = point_coord
        self.x_ncvar = x_ncvar
        self.y_ncvar = y_ncvar
        self.h_ncvar = h_ncvar
        self.weight_ncvar = weight_ncvar
        self.interp_pars = interp_pars

        self.variables = [FV.X, FV.Y]
        self.variables += [v for v in self.ovars if v not in self.fixed_vars]
        self.var2ncvar[FV.X] = x_ncvar
        self.var2ncvar[FV.Y] = y_ncvar
        if weight_ncvar is not None:
            self.var2ncvar[FV.WEIGHT] = weight_ncvar
            self.variables.append(FV.WEIGHT)
        elif FV.WEIGHT in self.var2ncvar:
            raise KeyError(
                f"States '{self.name}': Cannot have '{FV.WEIGHT}' in var2ncvar, use weight_ncvar instead"
            )

        self._n_pt = None
        self._n_wd = None
        self._n_ws = None

        if FV.WS not in self.ovars:
            raise ValueError(
                f"States '{self.name}': Expecting output variable '{FV.WS}', got {self.ovars}"
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

        self._cmap = {
            FC.STATE: self.states_coord,
            FC.POINT: self.point_coord,
        }

    def __repr__(self):
        return f"{type(self).__name__}(n_pt={self._n_pt}, n_wd={self._n_wd}, n_ws={self._n_ws})"

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
        coords, data = super()._read_ds(ds, cmap, variables, verbosity)

        assert FV.X in data and FV.Y in data, (
            f"States '{self.name}': Expecting variables '{FV.X}' and '{FV.Y}' in data, found {list(data.keys())}"
        )
        assert data[FV.X][0] == (FC.POINT,), (
            f"States '{self.name}': Expecting variable '{FV.X}' to have dimensions '({FC.POINT},)', got {data[FV.X][0]}"
        )
        assert data[FV.Y][0] == (FC.POINT,), (
            f"States '{self.name}': Expecting variable '{FV.Y}' to have dimensions '({FC.POINT},)', got {data[FV.Y][0]}"
        )
        if FV.H in data:
            assert data[FV.H][0] == (FC.POINT,), (
                f"States '{self.name}': Expecting variable '{FV.H}' to have dimensions '({FC.POINT},)', got {data[FV.H][0]}"
            )

        points = [data.pop(FV.X)[1], data.pop(FV.Y)[1]]
        if FV.H in data:
            points.append(data.pop(FV.H)[1])
        coords[FC.POINT] = np.stack(points, axis=-1)

        return coords, data

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
            bounds_extra_space=None,
            verbosity=verbosity,
        )

    def interpolate_data(self, idims, icrds, d, pts, vrs, times):
        """
        Interpolates data to points.

        This function should be implemented in derived classes.

        Parameters
        ----------
        idims: list of str
            The input dimensions, e.g. [x, y, height]
        icrds: list of numpy.ndarray
            The input coordinates, each with shape (n_i,)
            where n_i is the number of grid points in dimension i
        d: numpy.ndarray
            The data array, with shape (n1, n2, ..., nv)
            where ni represents the dimension sizes and
            nv is the number of variables
        pts: numpy.ndarray
            The points to interpolate to, with shape (n_pts, n_idims)
        vrs: list of str
            The variable names, length nv
        times: numpy.ndarray
            The time coordinates of the states, with shape (n_states,)
        Returns
        -------
        d_interp: numpy.ndarray
            The interpolated data array with shape (n_pts, nv)

        """

        # prepare interpolation parameters:
        ipars = dict(
            method="linear",
            rescale=True,
            fill_value=np.nan,
        )
        ipars.update(self.interp_pars)

        def _check_nan(gpts, d, pts, idims, results):
            """ Checks for NaN results and raises errors. """
            if np.isnan(ipars.get("fill_value", np.nan)):
                sel = np.isnan(results)
                if np.any(sel):
                    i = [j[0] for j in np.where(sel)]
                    p = pts[i[0]]
                    qmin = np.min(gpts, axis=0)
                    qmax = np.max(gpts, axis=0)
                    isin = (p >= qmin) & (p <= qmax)
                    method = "linear"
                    print("\n\nInterpolation error")
                    print("dims:   ", idims[1:] if FC.STATE in idims else idims)
                    print(f"point {i[0]}: ", p)
                    print("qmin:   ", qmin)
                    print("qmax:   ", qmax)
                    print("Inside: ", isin, "\n\n")

                    if not np.all(isin):
                        raise ValueError(
                            f"States '{self.name}': Interpolation method '{method}' failed for {np.sum(sel)} points, e.g. for point {p}, outside of bounds {qmin} - {qmax}, dimensions = {idims}. "
                        )
                    else:
                        sel2 = np.isnan(d)
                        if np.any(sel2):
                            i = np.where(sel2)
                            p = gpts[i[0][0]]
                            v = vrs[i[1][0]]
                            print(
                                f"NaN data found in input data during interpolation, e.g. for variable '{v}' at point:"
                            )
                            for ic, c in enumerate(idims):
                                print(f"  {c}: {p[ic]}")
                            for iw, w in enumerate(vrs):
                                print(f"  {w}: {d[i[0][0], iw]}")
                            print("\n\n")
                            raise ValueError(
                                f"States '{self.name}': Interpolation method '{method}' failed, NaN values found in input data for {np.sum(sel)} grid points, e.g. {gpts[i[0]]} with {v} = {d[i[0][0], i[1][0]]}."
                            )
                        raise ValueError(
                            f"States '{self.name}': Interpolation method '{method}' failed for {np.sum(sel)} points, for unknown reason."
                        )
        if FC.STATE in idims:
            raise NotImplementedError(
                f"States '{self.name}': Interpolation with state dimension not implemented."
            )

        # prepare grid points:
        assert len(idims) == 1 and idims[0] == FC.POINT, (
            f"States '{self.name}': Only point cloud interpolation supported, got dimensions {idims}"
        )
        gpts = icrds[0]

        # remove NaN data points:
        if not self.check_input_nans:
            sel = np.any(np.isnan(d), axis=tuple(range(1, d.ndim)))
            if np.any(sel):
                gpts = gpts[~sel]
                d = d[~sel]

        # interpolate:
        results = griddata(gpts, d, pts, **ipars)

        # check for NaN results:
        _check_nan(gpts, d, pts, idims, results)

        return results


class WeibullPointCloud(PointCloudData):
    """
    Weibull sectors at point cloud support, e.g., at turbine locations.

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

    Examples
    --------
    Example of the NetCDF input files with point cloud data:

    >>>    Dimensions:               (wind_turbine: 8, wind_direction: 2, wind_speed: 2)
    >>>    Coordinates:
    >>>    * wind_turbine          (wind_turbine) int64 64B 0 1 2 3 4 5 6 7
    >>>    * wind_direction        (wind_direction) int64 16B 0 30
    >>>    * wind_speed            (wind_speed) int64 16B 8 10
    >>>    Data variables:
    >>>        sector_probability    (wind_turbine, wind_direction) float64 128B ...
    >>>        weibull_a             (wind_turbine, wind_direction) float64 128B ...
    >>>        weibull_k             (wind_turbine, wind_direction) float64 128B ...
    >>>        turbulence_intensity  (wind_turbine, wind_direction, wind_speed) float64 256B ...
    >>>        x                     (wind_turbine) float64 64B ...
    >>>        y                     (wind_turbine) float64 64B ...
    >>>        height                (wind_turbine) float64 64B ...

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

        assert ws_coord is not None or ws_bins is not None, (
            f"States '{self.name}': Expecting either ws_coord or ws_bins"
        )
        assert ws_coord is None or ws_bins is None, (
            f"States '{self.name}': Expecting either ws_coord or ws_bins, not both"
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
        hcmap = cmap.copy()
        if self.ws_coord is not None:
            hcmap = {FV.WS: self.ws_coord, **cmap}
        coords, data0 = super()._read_ds(ds, hcmap, variables, verbosity)
        wd = coords.pop(FC.STATE)
        wss = coords.pop(FV.WS, None)

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
        elif wss is not None:
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
        del wsb

        # calculate Weibull weights
        dms = [FV.WS, FV.WD]
        shp = [n_ws, n_wd]
        for v in [FV.WEIBULL_A, FV.WEIBULL_k]:
            if FC.POINT in data0[v][0]:
                dms.append(FC.POINT)
                shp.append(data0[v][1].shape[data0[v][0].index(FC.POINT)])
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
            w
            * weibull_weights(
                ws=wss[s_ws],
                ws_deltas=wsd[s_ws],
                A=data0.pop(FV.WEIBULL_A)[1][s_A],
                k=data0.pop(FV.WEIBULL_k)[1][s_k],
            ),
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
            if len(dims) >= 2 and dims[:2] == (FV.WS, FV.WD):
                dms = tuple([FC.STATE] + list(dims[2:]))
                shp = [self._N] + list(d.shape[2:])
                data[v] = (dms, d.reshape(shp))
            elif dims[0] == FV.WD:
                dms = tuple([FC.STATE] + list(dims[1:]))
                shp = [n_ws] + list(d.shape)
                data[v] = np.zeros(shp, dtype=config.dtype_double)
                data[v][:] = d[None, ...]
                data[v] = (dms, data[v].reshape([self._N] + shp[2:]))
            elif dims[0] == FV.WS:
                dms = tuple([FC.STATE] + list(dims[1:]))
                shp = [n_ws, n_wd] + list(d.shape[2:])
                data[v] = np.zeros(shp, dtype=config.dtype_double)
                data[v][:] = d[:, None, ...]
                data[v] = (dms, data[v].reshape([self._N] + shp[2:]))
            else:
                data[v] = (dims, d)

        return coords, data
