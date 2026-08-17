import numpy as np
import xarray as xr
from typing import Any, cast
from foxes.core import Algorithm, FData, LoadedData, MData
from scipy.interpolate import griddata
from scipy.spatial import QhullError

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
    states_coord
        The states coordinate name in the data
    point_coord
        The point coordinate name in the data
    x_ncvar
        The x variable name in the data
    y_ncvar
        The y variable name in the data
    h_ncvar
        The height variable name in the data
    weight_ncvar
        The name of the weights variable in the data

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
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        args
            Arguments for the base class
        states_coord
            The states coordinate name in the data
        point_coord
            The point coordinate name in the data
        x_ncvar
            The x variable name in the data
        y_ncvar
            The y variable name in the data
        h_ncvar
            The height variable name in the data
        weight_ncvar
            The name of the weights variable in the data
        kwargs
            Additional parameters for the base class

        """
        super().__init__(*args, bounds_extra_space=None, **kwargs)

        self.states_coord = states_coord
        self.point_coord = point_coord
        self.x_ncvar = x_ncvar
        self.y_ncvar = y_ncvar
        self.h_ncvar = h_ncvar
        self.weight_ncvar = weight_ncvar

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

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_pt={self._n_pt}, n_wd={self._n_wd}, n_ws={self._n_ws})"

    def get_grid_points(
        self,
        loaded_data: LoadedData | None = None,
        mdata: MData | None = None,
        all_heights: bool = True,
        height: float | None = None,
    ) -> np.ndarray:
        """
        Returns explicit point-cloud coordinates.

        Parameters
        ----------
            loaded_data
            The loaded data dictionary.
            mdata
            The model data.
            all_heights
            Must be True because point-cloud states do not expose a separate
            height axis.
            height
            Must be None because point-cloud states contain explicit points.

        Returns
        -------
        grid_points
            The explicit point coordinates, shape ``(n_points, n_coordinates)``.

        """
        assert loaded_data is not None or mdata is not None, (
            f"States '{self.name}': Either loaded_data or mdata must be provided"
        )
        assert loaded_data is None or mdata is None, (
            f"States '{self.name}': Either loaded_data or mdata must be provided, not both"
        )
        assert all_heights and height is None, (
            f"States '{self.name}': Point-cloud states do not support height selection"
        )

        source = cast(dict[str, Any], mdata if mdata is not None else loaded_data)
        point_coord = self.var(FC.POINT)
        if point_coord not in source and FC.POINT in source:
            point_coord = FC.POINT
        assert point_coord in source, (
            f"States '{self.name}': Missing point coordinates '{point_coord}'"
        )
        points = np.asarray(source[point_coord])
        return points.reshape(-1, points.shape[-1])

    def _read_ds(
        self,
        ds: xr.Dataset,
        cmap: dict[str, str] | None = None,
        verbosity: int = 0,
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, tuple[tuple[str, ...], np.ndarray]],
    ]:
        """
        Helper function for _get_data, extracts data from the original Dataset.

        Parameters
        ----------
        ds
            The Dataset to read data from
        cmap
            A mapping from foxes variable names to Dataset dimension names, if not given self._cmap will be used
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        coords
            keys: Foxes variable names, values: 1D coordinate value arrays
        data
            The extracted data, keys are variable names,
            values are tuples (dims, data_array)
            where dims is a tuple of dimension names and
            data_array is a numpy.ndarray with the data values

        """
        coords, data = super()._read_ds(ds, cmap=cmap, verbosity=verbosity)

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

        point_axes = [FV.X, FV.Y]
        points = [data.pop(FV.X)[1], data.pop(FV.Y)[1]]
        if FV.H in data:
            point_axes.append(FV.H)
            points.append(data.pop(FV.H)[1])
        coords[FC.XYH] = np.asarray(point_axes)
        cast(dict[str, Any], coords)[FC.POINT] = (
            (FC.POINT, FC.XYH),
            np.stack(points, axis=-1),
        )

        return coords, data

    def _check_nan(
        self,
        ipars: dict[str, bool | float | str | None],
        gpts: np.ndarray,
        d: np.ndarray,
        pts: np.ndarray,
        idims: list[str],
        vrs: list[str],
        results: np.ndarray,
    ) -> None:
        """Checks for NaN results and raises errors."""
        fill_value = ipars.get("fill_value", np.nan)
        if isinstance(fill_value, (int, float)) and np.isnan(fill_value):
            sel = np.isnan(results)
            if np.any(sel):
                point_indices = [j[0] for j in np.where(sel)]
                p = pts[point_indices[0]]
                qmin = np.min(gpts, axis=0)
                qmax = np.max(gpts, axis=0)
                isin = (p >= qmin) & (p <= qmax)
                method = "linear"
                print("\n\nInterpolation error")
                print("dims:   ", idims[1:] if FC.STATE in idims else idims)
                print(f"point {point_indices[0]}: ", p)
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
                        nan_indices = np.where(sel2)
                        p = gpts[nan_indices[0][0]]
                        v = vrs[nan_indices[1][0]]
                        print(
                            f"NaN data found in input data during interpolation, e.g. for variable '{v}' at point:"
                        )
                        for ic, c in enumerate(idims):
                            print(f"  {c}: {p[ic]}")
                        for iw, w in enumerate(vrs):
                            print(f"  {w}: {d[nan_indices[0][0], iw]}")
                        print("\n\n")
                        raise ValueError(
                            f"States '{self.name}': Interpolation method '{method}' failed, NaN values found in input data for {np.sum(sel)} grid points, e.g. {gpts[nan_indices[0][0]]} with {v} = {d[nan_indices[0][0], nan_indices[1][0]]}."
                        )
                    raise ValueError(
                        f"States '{self.name}': Interpolation method '{method}' failed for {np.sum(sel)} points, for unknown reason."
                    )

    def interpolate_data(
        self,
        mdata: MData,
        idims: list[str],
        d: np.ndarray,
        pts: np.ndarray,
        vrs: list[str],
        state_indices: np.ndarray | None = None,
        gpts: tuple[np.ndarray, ...] | np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Interpolates point-cloud data to evaluation points.

        Parameters
        ----------
        mdata
            The model data.
        idims
            The input dimensions, typically ``[FC.POINT]``.
        d
            The data array to interpolate.
        pts
            The points to interpolate to, shape ``(n_pts, n_idims)``.
        vrs
            Variable names.
        state_indices
            State indices, unused here.
        gpts
            Explicit grid points.

        Returns
        -------
        numpy.ndarray
            Interpolated data, shape ``(n_pts, ...)``.

        """
        # prepare interpolation parameters:
        ipars: dict[str, bool | float | str | None] = dict(
            method="linear",
            rescale=True,
            fill_value=np.nan,
        )
        ipars.update(self.interp_pars)

        if FC.STATE in idims:
            raise NotImplementedError(
                f"States '{self.name}': Interpolation with state dimension not implemented."
            )

        assert len(idims) == 1 and idims[0] == FC.POINT, (
            f"States '{self.name}': Only point cloud interpolation supported, got dimensions {idims}"
        )

        if gpts is None:
            gpts = self.get_interpolation_grid_data(mdata, idims)
        if isinstance(gpts, (tuple, list)):
            assert len(gpts) == 1, (
                f"States '{self.name}': Expecting one point-cloud coordinate array, got {gpts}"
            )
            gpts = gpts[0]
        gpts_array: np.ndarray = np.asarray(gpts)
        pts = np.asarray(pts)

        if gpts_array.ndim == 1:
            gpts_array = gpts_array[:, None]
        if pts.ndim == 1:
            pts = pts[None, :]

        # remove NaN data points:
        if not self.check_input_nans:
            sel = np.any(np.isnan(d), axis=tuple(range(1, d.ndim)))
            if np.any(sel):
                gpts_array = gpts_array[~sel]
                d = d[~sel]

        # interpolate
        try:
            results = griddata(gpts_array, d, pts, **ipars)
        except QhullError:
            if ipars.get("method", "linear") == "nearest":
                raise
            fpars = dict(ipars)
            fpars["method"] = "nearest"
            results = griddata(gpts_array, d, pts, **fpars)

        # check for NaN results:
        self._check_nan(ipars, gpts_array, d, pts, idims, vrs, results)

        return results


class WeibullPointCloud(PointCloudData):
    """
    Weibull sectors at point cloud support, e.g., at turbine locations.

    Attributes
    ----------
    wd_coord
        The wind direction coordinate name
    ws_coord
        The wind speed coordinate name, if wind speed bin
        centres are in data, else None
    ws_bins
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
        args
            Positional arguments for the base class
        wd_coord
            The wind direction coordinate name
        ws_coord
            The wind speed coordinate name, if wind speed bin
            centres are in data
        ws_bins
            The wind speed bins, including
            lower and upper bounds
        kwargs
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

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_wd={self._n_wd}, n_ws={self._n_ws})"

    def _read_ds(
        self,
        ds: xr.Dataset,
        cmap: dict[str, str] | None = None,
        verbosity: int = 0,
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, tuple[tuple[str, ...], np.ndarray]],
    ]:
        """
        Helper function for _get_data, extracts data from the original Dataset.

        Parameters
        ----------
        ds
            The Dataset to read data from
        cmap
            A mapping from foxes variable names to Dataset dimension names, if not given self._cmap will be used
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        coords
            keys: Foxes variable names, values: 1D coordinate value arrays
        data
            The extracted data, keys are variable names,
            values are tuples (dims, data_array)
            where dims is a tuple of dimension names and
            data_array is a numpy.ndarray with the data values

        """
        # read data, using wd_coord as state coordinate
        hcmap = self._cmap.copy() if cmap is None else cmap.copy()
        if self.ws_coord is not None:
            hcmap = {FV.WS: self.ws_coord, **hcmap}
        coords, data0 = super()._read_ds(ds, cmap=hcmap, verbosity=verbosity)
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
        dimension_names: list[str] = [FV.WS, FV.WD]
        shape: list[int] = [n_ws, n_wd]
        for v in [FV.WEIBULL_A, FV.WEIBULL_k]:
            if FC.POINT in data0[v][0]:
                dimension_names.append(FC.POINT)
                shape.append(data0[v][1].shape[data0[v][0].index(FC.POINT)])
                break
        dms = tuple(dimension_names)
        shp = tuple(shape)
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
        self._inds = np.arange(self._N, dtype=config.dtype_int)
        translated_data: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
        ws_data: np.ndarray = np.zeros((n_ws, n_wd), dtype=config.dtype_double)
        wd_data: np.ndarray = np.zeros((n_ws, n_wd), dtype=config.dtype_double)
        ws_data[:] = wss[:, None]
        wd_data[:] = wd[None, :]
        translated_data[FV.WS] = ((FC.STATE,), ws_data.reshape(self._N))
        translated_data[FV.WD] = ((FC.STATE,), wd_data.reshape(self._N))
        for v in list(data0.keys()):
            dims, d = data0.pop(v)
            if len(dims) >= 2 and dims[:2] == (FV.WS, FV.WD):
                dms = tuple([FC.STATE] + list(dims[2:]))
                shape = [self._N] + list(d.shape[2:])
                translated_data[v] = (dms, d.reshape(shape))
            elif dims[0] == FV.WD:
                dms = tuple([FC.STATE] + list(dims[1:]))
                shape = [n_ws] + list(d.shape)
                expanded_data: np.ndarray = np.zeros(
                    shape, dtype=config.dtype_double
                )
                expanded_data[:] = d[None, ...]
                translated_data[v] = (
                    dms,
                    expanded_data.reshape([self._N] + shape[2:]),
                )
            elif dims[0] == FV.WS:
                dms = tuple([FC.STATE] + list(dims[1:]))
                shape = [n_ws, n_wd] + list(d.shape[2:])
                expanded_data = np.zeros(shape, dtype=config.dtype_double)
                expanded_data[:] = d[:, None, ...]
                translated_data[v] = (
                    dms,
                    expanded_data.reshape([self._N] + shape[2:]),
                )
            else:
                translated_data[v] = (dims, d)

        return coords, translated_data


class TurbinePointCloud(DatasetStates):
    """
    Point cloud data at turbine locations, for wake calculations.

    Attributes
    ----------
    states_coord
        The coordinate name for the states dimension.
    turbine_coord
        The coordinate name for the turbine dimension.

    :group: input.states

    """

    def __init__(
        self,
        *args,
        states_coord=FC.STATE,
        turbine_coord=FC.TURBINE,
        weight_ncvar=None,
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        args
            Positional arguments for the base class
        states_coord
            The states coordinate name in the data
        turbine_coord
            The turbine coordinate name in the data
        weight_ncvar
            The name of the weights variable in the data
        kwargs
            Keyword arguments for the base class

        """
        # Turbine-point-cloud data is indexed by turbine, not by global X/Y grids.
        # Disable XY-bound filtering from DatasetStates to avoid requiring X/Y cmap.
        super().__init__(
            *args,
            load_mode="preload",
            bounds_extra_space=None,
            **kwargs,
        )

        self.states_coord = states_coord
        self.turbine_coord = turbine_coord

        if weight_ncvar is not None:
            self.var2ncvar[FV.WEIGHT] = weight_ncvar
            self.variables.append(FV.WEIGHT)
        elif FV.WEIGHT in self.var2ncvar:
            raise KeyError(
                f"States '{self.name}': Cannot have '{FV.WEIGHT}' in var2ncvar, use weight_ncvar instead"
            )

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
            FC.TURBINE: self.turbine_coord,
        }

    def load_data(  # type: ignore[override]
        self,
        algo: Algorithm,
        loaded_data: LoadedData,
        force: bool = False,
        bounds_extra_space: float | str | None = None,
        height_bounds: tuple[float, float] | None = None,
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
        bounds_extra_space
            Extra horizontal bounds; unsupported for turbine point-cloud data.
        height_bounds
            Height bounds; unsupported for turbine point-cloud data.
            Keys are "coords", a dict with entries `dim_name_str -> dim_array`;
            "data_vars", a dict with entries `name_str -> (dim_tuple, data_ndarray)`;
            and "extra_data", a dict with non-array additional data.
        force
            Overwrite existing data
        verbosity
            The verbosity level, 0 = silent

        """
        super().load_data(
            algo,
            loaded_data,
            force=force,
            bounds_extra_space=None,
            verbosity=verbosity,
        )

    def _update_dims(
        self,
        dims: tuple[str, ...],
        coords: dict[str, np.ndarray],
        vrs: list[str],
        d: np.ndarray,
        fdata: FData,
    ) -> tuple[tuple[str, ...], dict[str, np.ndarray]]:
        """Helper function for dimension adjustment, if needed"""
        coords[FC.TURBINE] = fdata[FV.TXYH]
        return dims, coords

    def get_grid_points(
        self,
        loaded_data: LoadedData | None = None,
        mdata: MData | None = None,
        all_heights: bool = True,
        height: float | None = None,
    ) -> np.ndarray:
        """
        Returns explicit turbine point-cloud coordinates.

        Parameters
        ----------
            loaded_data
            The loaded data dictionary.
            mdata
            The model data.
            all_heights
            Must be True because turbine point-cloud states do not expose a
            separate height axis.
            height
            Must be None because turbine heights are part of the explicit
            turbine coordinates.

        Returns
        -------
        grid_points
            The explicit turbine coordinates, shape
            ``(n_states * n_turbines, 3)``.

        """
        assert loaded_data is not None or mdata is not None, (
            f"States '{self.name}': Either loaded_data or mdata must be provided"
        )
        assert loaded_data is None or mdata is None, (
            f"States '{self.name}': Either loaded_data or mdata must be provided, not both"
        )
        assert all_heights and height is None, (
            f"States '{self.name}': Turbine point-cloud states do not support height selection"
        )

        source = cast(dict[str, Any], mdata if mdata is not None else loaded_data)
        if FV.TXYH in source:
            points = np.asarray(source[FV.TXYH])
        else:
            turbine_coord = self.var(FC.TURBINE)
            if turbine_coord not in source and FC.TURBINE in source:
                turbine_coord = FC.TURBINE
            assert turbine_coord in source, (
                f"States '{self.name}': Missing turbine coordinates '{turbine_coord}'"
            )
            points = np.asarray(source[turbine_coord])
        return points.reshape(-1, points.shape[-1])

    def interpolate_data(
        self,
        mdata: MData,
        idims: list[str],
        d: np.ndarray,
        pts: np.ndarray,
        vrs: list[str],
        state_indices: np.ndarray | None = None,
        gpts: tuple[np.ndarray, ...] | np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Interpolates turbine point-cloud data to the evaluation points.

        Parameters
        ----------
        mdata
            The model data.
        idims
            The input dimensions, typically ``[FC.TURBINE]``.
        d
            The turbine data array.
        pts
            The evaluation points, shape ``(n_pts, n_idims)``.
        vrs
            Variable names.
        state_indices
            State indices, unused here.
        gpts
            Explicit grid points.

        Returns
        -------
        numpy.ndarray
            Interpolated values, shape ``(n_states, n_turbines, n_vars)``.

        """
        # special case of time-only data:
        if len(idims) == 0:
            assert pts is None, (
                f"States '{self.name}': Expecting no points for time-only data, got {pts}"
            )
            return d[:, None, ...]

        assert len(idims) == 1 and idims[0] == FC.TURBINE, (
            f"States '{self.name}': Only turbine point cloud interpolation supported, got dimensions {idims}"
        )

        if gpts is None:
            gpts = (self.get_grid_points(mdata=mdata),)
        if isinstance(gpts, (tuple, list)):
            assert len(gpts) == 1, (
                f"States '{self.name}': Expecting one turbine coordinate array, got {gpts}"
            )
            gpts = gpts[0]
        gpts_array: np.ndarray = np.asarray(gpts)
        pts = np.asarray(pts)

        if gpts_array.ndim == 1:
            gpts_array = gpts_array[:, None]
        if pts.ndim == 1:
            pts = pts[None, :]

        if (
            gpts_array.ndim == 2
            and gpts_array.shape[0] == 1
            and pts.ndim == 2
            and gpts_array.shape[1] == pts.shape[0]
        ):
            return d

        # special case of evaluation at turbine locations:
        if np.allclose(gpts_array, pts):
            return d

        # prepare interpolation parameters:
        ipars: dict[str, bool | float | str | None] = dict(
            method="linear",
            rescale=True,
            fill_value=np.nan,
        )
        ipars.update(self.interp_pars)

        # normalize point shapes to the state-aware turbine grid:
        if gpts_array.ndim == 2 and gpts_array.shape[-1] == 3:
            gpts_array = gpts_array[None, ...]
        if pts.ndim == 2 and pts.shape[-1] == 3:
            pts = pts[None, ...]
        if gpts_array.ndim == 3 and pts.ndim == 2:
            pts = np.broadcast_to(pts[None, ...], gpts_array.shape)
        elif (
            gpts_array.ndim == 3
            and pts.ndim == 3
            and pts.shape[0] == 1
            and gpts_array.shape[0] > 1
        ):
            pts = np.broadcast_to(pts, gpts_array.shape)

        n_states, n_turbines = gpts_array.shape[:2]
        if pts.shape != gpts_array.shape:
            raise ValueError(
                f"States '{self.name}': Expecting evaluation points shape {gpts_array.shape}, got {pts.shape}"
            )

        gpts2 = np.concatenate(
            [
                np.arange(n_states)[:, None, None] * np.ones((n_states, n_turbines, 1)),
                gpts_array,
            ],
            axis=-1,
        )
        epts = np.concatenate(
            [
                np.arange(n_states)[:, None, None] * np.ones((n_states, n_turbines, 1)),
                pts,
            ],
            axis=-1,
        )

        # check redundant dimensions:
        rmvd = []
        for i in range(1, gpts2.shape[-1]):
            if np.abs(np.min(gpts2[..., i]) - np.max(gpts2[..., i])) < 1e-12:
                rmvd.append(i)
        if len(rmvd) > 0:
            gpts2 = np.delete(gpts2, rmvd, axis=-1)
            epts = np.delete(epts, rmvd, axis=-1)

        # interpolate:
        gpts2 = gpts2.reshape(n_states * n_turbines, gpts2.shape[-1])
        epts = epts.reshape(n_states * n_turbines, epts.shape[-1])
        d2 = np.asarray(d).reshape(n_states * n_turbines, d.shape[-1])
        try:
            results = griddata(gpts2, d2, epts, **ipars)
        except QhullError:
            if ipars.get("method", "linear") == "nearest":
                raise
            fpars = dict(ipars)
            fpars["method"] = "nearest"
            results = griddata(gpts2, d2, epts, **fpars)

        PointCloudData._check_nan(
            cast(PointCloudData, self), ipars, gpts2, d2, epts, idims, vrs, results
        )

        results = results.reshape(n_states, n_turbines, results.shape[-1])
        return results
