import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path

from foxes.core import Algorithm, LoadedData, MData
from scipy.interpolate import griddata

from foxes.utils.utm_utils import from_lonlat
from foxes.config import config, get_output_path
from foxes.output import FarmLayoutOutput
import foxes.variables as FV
import foxes.constants as FC

from .dataset_states import DatasetStates, InterpolationParameters


class NEWAStates(DatasetStates):
    """
    Heterogeneous ambient states in NEWA-WRF format.

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

    Examples
    --------
    Example of one of the NetCDF input files in NEWA format:

    >>>     Dimensions:      (time: 144, south_north: 165, west_east: 234, height: 15)
    >>>     Coordinates:
    >>>      * time         (time) datetime64[ns] 1kB 2006-01-04 ... 2006-01-04T23:54:00
    >>>      * south_north  (south_north) float32 660B -1.79e+05 -1.77e+05 ... 1.49e+05
    >>>      * west_east    (west_east) float32 936B -2.48e+05 -2.46e+05 ... 2.18e+05
    >>>      * height       (height) float32 60B 25.0 50.0 75.0 90.0 ... 400.0 500.0 1e+03
    >>>        XLAT         (south_north, west_east) float32 154kB ...
    >>>        XLON         (south_north, west_east) float32 154kB ...
    >>>    Data variables: (12/24)
    >>>        WS           (time, height, south_north, west_east) float32 334MB ...
    >>>        ...

    :group: input.states

    """

    def __init__(
        self,
        input_files_nc: str | Path | xr.Dataset,
        time_coord: str = "time",
        west_east_coord: str = "west_east",
        south_north_coord: str = "south_north",
        height_coord: str = "height",
        xlat_coord: str = "XLAT",
        xlon_coord: str = "XLON",
        output_vars: list[str] | None = None,
        var2ncvar: dict[str, str] | None = None,
        load_mode: str = "fly",
        time_format: str | None = None,
        interp_pars: InterpolationParameters = {},
        wrf_point_plot: str | Path | None = None,
        **kwargs: object,
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        input_files_nc: str or pathlib.Path or xarray.Dataset
            The input netcdf file(s), can contain
            wildcards, e.g. 'wrfout_2025*.nc'
        time_coord: str
            The time coordinate name in the data
        west_east_coord: str
            The west-east coordinate name in the data
        south_north_coord: str
            The south-north coordinate name in the data
        height_coord: str, optional
            The height coordinate name in the data
        xlat_coord: str
            The latitude coordinate name in the data
        xlon_coord: str
            The longitude coordinate name in the data
        output_vars: list of str, optional
            The output variables to load, if None,
            the default variables are loaded
            (FV.WS, FV.WD, FV.TI, FV.RHO)
        var2ncvar: dict[str, str], optional
            A dictionary mapping foxes variable names
            to the corresponding netcdf variable names.
        load_mode: str
            The load mode, choices: preload, lazy, fly.
            preload loads all data during initialization,
            lazy lazy-loads the data using dask, and fly
            reads only states index and weights during initialization
            and then opens the relevant files again within
            the chunk calculations.
        time_format: str or None, optional
            The datetime parsing format string
        interp_pars: dict[str, bool or float or str or None], optional
            Additional parameters for scipy.interpolate.griddata,
            e.g. {'method': 'linear', 'fill_value': None, 'rescale': True}
        wrf_point_plot: str or pathlib.Path or None, optional
            Path to a plot file, e.g. wrf_points.png, to visualize the
            selected WRF grid points and the layout of the farm.
        kwargs: object
            Additional parameters for the base class

        """
        if output_vars is None:
            ovars = [FV.WS, FV.WD, FV.TI, FV.RHO]
        else:
            ovars = output_vars

        if var2ncvar is None:
            var2ncvar = {
                FV.WS: "WS",
                FV.WD: "WD",
                FV.TKE: "TKE",
                FV.RHO: "RHO",
            }

        super().__init__(
            data_source=input_files_nc,
            output_vars=ovars,
            var2ncvar=var2ncvar,
            time_format=time_format,
            load_mode=load_mode,
            weight_factor=None,
            interp_pars=interp_pars,
            **kwargs,  # type: ignore[arg-type]
        )

        self.time_coord = time_coord
        self.west_east_coord = west_east_coord
        self.south_north_coord = south_north_coord
        self.height_coord = height_coord
        self.xlat_coord = xlat_coord
        self.xlon_coord = xlon_coord
        self.wrf_point_plot = wrf_point_plot
        self.variables = list(set([v if v != FV.TI else FV.TKE for v in ovars]))

        self._cmap = {
            FC.STATE: self.time_coord,
            FV.X: self.west_east_coord,
            FV.Y: self.south_north_coord,
            FV.H: self.height_coord,
        }

    def preproc_first(
        self,
        algo: Algorithm,
        data: xr.Dataset,
        bounds_extra_space: float | str | None = None,
        height_bounds: tuple[float, float] | None = None,
        loaded_data: LoadedData | None = None,
        verbosity: int = 0,
    ) -> None:
        """
        Preprocesses the first file.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        data: xarray.Dataset
            The dataset to preprocess
        bounds_extra_space: float or str or None, optional
            The extra space, either float in m,
            or str for units of D, e.g. '2.5D'
        height_bounds: tuple[float, float], optional
            The (h_min, h_max) height bounds in m. Defaults to H +/- 0.5*D
        loaded_data: LoadedData, optional
            If given, optionally add to this loaded data dict with entries
            {"coords": {}, "data_vars": {}, "extra_data": {}}
        verbosity: int
            The verbosity level, 0 = silent

        """

        super().preproc_first(
            algo,
            data,
            bounds_extra_space=None,
            height_bounds=height_bounds,
            loaded_data=None,
            verbosity=verbosity,
        )

        if verbosity > 0:
            print(
                f"States '{self.name}': Selected UTM zone: {config.utm_zone[0]}{config.utm_zone[1]}"
            )

        lonlat = np.stack(
            (data[self.xlon_coord].values, data[self.xlat_coord].values), axis=-1
        )
        lonlat = np.moveaxis(lonlat, 0, 1)  # (y, x, 2) to (x, y, 2)
        nx, ny = lonlat.shape[:2]
        lonlat = lonlat.reshape((nx * ny, 2))
        _xy = from_lonlat(lonlat)
        _xy = _xy.reshape((nx, ny, 2))
        nh = len(self._heights)
        self.XY = self.var(f"{FV.X}{FV.Y}")
        self.X = self.var(FV.X)
        self.Y = self.var(FV.Y)
        self.H = self.var(FV.H)

        # find horizontal bounds:
        if bounds_extra_space is not None:
            assert FV.X in self._cmap, (
                f"States '{self.name}': x coordinate '{FV.X}' not in cmap {self._cmap}"
            )
            assert FV.Y in self._cmap, (
                f"States '{self.name}': y coordinate '{FV.Y}' not in cmap {self._cmap}"
            )

            # if bounds and self.x_coord is not None and self.x_coord not in self.sel:
            xy_min, xy_max = algo.farm.get_xy_bounds(
                extra_space=bounds_extra_space, algo=algo
            )
            x0, x1 = xy_min[0], xy_max[0]
            y0, y1 = xy_min[1], xy_max[1]
            if verbosity > 0:
                print(
                    f"States '{self.name}': Restricting {FV.X} to bounds {x0:.2f} - {x1:.2f}"
                )
                print(
                    f"States '{self.name}': Restricting {FV.Y} to bounds {y0:.2f} - {y1:.2f}"
                )

            inds = np.argwhere(
                (_xy[..., 0] >= x0)
                & (_xy[..., 0] <= x1)
                & (_xy[..., 1] >= y0)
                & (_xy[..., 1] <= y1)
            )
            assert len(inds) > 0, (
                f"States '{self.name}': No grid points found within bounds (x0, x1)=({x0}, {x1}), (y0, y1)=({y0}, {y1})"
            )
            i0 = inds[:, 0].min()
            i1 = inds[:, 0].max()
            j0 = inds[:, 1].min()
            j1 = inds[:, 1].max()
            while True:
                xy = _xy[i0 : i1 + 1, j0 : j1 + 1]
                if i0 > 0 and x0 < np.min(xy[..., 0]):
                    i0 -= 1
                elif i1 < nx - 1 and x1 > np.max(xy[..., 0]):
                    i1 += 1
                elif j0 > 0 and y0 < np.min(xy[..., 1]):
                    j0 -= 1
                elif j1 < ny - 1 and y1 > np.max(xy[..., 1]):
                    j1 += 1
                else:
                    break
            nx, ny = xy.shape[:2]

            if self.isel is None:
                self.isel = {}
            self.isel.update(
                {
                    self.west_east_coord: slice(i0, i1 + 1),
                    self.south_north_coord: slice(j0, j1 + 1),
                }
            )
            if verbosity > 0:
                print(
                    f"States '{self.name}': Selected {FV.X} = {np.min(xy[..., 0]):.2f} - {np.max(xy[..., 0]):.2f} ({nx} points)"
                )
                print(
                    f"States '{self.name}': Selected {FV.Y} = {np.min(xy[..., 1]):.2f} - {np.max(xy[..., 1]):.2f} ({ny} points)"
                )
                print(
                    f"States '{self.name}': Selected {xy.shape[:2] + (nh,)} grid points"
                )
        else:
            xy = _xy
            if verbosity > 0:
                print(
                    f"States '{self.name}': Selecting all {xy.shape[:2] + (nh,)} grid points"
                )

        if loaded_data is not None:
            loaded_data["data_vars"][self.XY] = ((self.X, self.Y, FC.XY), xy)
            loaded_data["coords"][self.H] = self._heights

        if self.wrf_point_plot is not None:
            fpath = get_output_path(self.wrf_point_plot)
            if verbosity > 0:
                print(f"States '{self.name}': Writing WRF grid point plot to '{fpath}'")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(
                xy[..., 0].flatten(),
                xy[..., 1].flatten(),
                c="blue",
                alpha=0.2,
                marker=".",
                linestyle="None",
            )
            wind_farm_names = algo.farm.wind_farm_names
            assert wind_farm_names is not None
            anno = 3 if len(wind_farm_names) > 1 else 0
            FarmLayoutOutput(farm=algo.farm).get_figure(
                fig=fig, ax=ax, annotate=anno, fontsize=12
            )
            ax.set_xlabel(f"{FV.X} [m]")
            ax.set_ylabel(f"{FV.Y} [m]")
            ax.set_aspect("equal", adjustable="box")
            ax.autoscale_view(tight=True)
            fig.savefig(fpath, bbox_inches="tight")
            plt.close()

    def get_grid_points(
        self,
        loaded_data: LoadedData | None = None,
        mdata: MData | None = None,
        all_heights: bool = True,
        height: float | None = None,
    ) -> np.ndarray:
        """
        Returns the grid points from the mdata object.

        Parameters
        ----------
        loaded_data: LoadedData, optional
            The loaded data dictionary
        mdata: foxes.core.MData, optional
            The model data
        all_heights: bool, optional
            If True, return all heights, otherwise only the highest.
        height: float, optional
            The height to use. If None, the highest height is used if
            all_heights is False.

        Returns
        -------
        grid_points: numpy.ndarray
            The grid points, shape (n_points, 3)

        """
        assert loaded_data is not None or mdata is not None, (
            f"States '{self.name}': Either loaded_data or mdata must be provided"
        )
        assert loaded_data is None or mdata is None, (
            f"States '{self.name}': Either loaded_data or mdata must be provided, not both"
        )

        if mdata is not None:
            assert self.XY in mdata, (
                f"States '{self.name}': Missing grid points '{self.XY}' in mdata, got {list(mdata.keys())}"
            )
            xy = mdata[self.XY]

            if all_heights or height is None:
                assert self.H in mdata, (
                    f"States '{self.name}': Missing heights '{self.H}' in mdata, got {list(mdata.keys())}"
                )
                h = mdata[self.H]
                if height is None:
                    h = np.atleast_1d(np.max(h))
                elif all_heights:
                    raise ValueError(
                        f"States '{self.name}': Cannot specify both all_heights and height, got all_heights={all_heights}, height={height}"
                    )
            else:
                h = np.atleast_1d(height)

        else:
            assert loaded_data is not None
            assert self.XY in loaded_data["data_vars"], (
                f"States '{self.name}': Missing coordinates '{self.XY}' in loaded_data, got {list(loaded_data['data_vars'].keys())}"
            )
            xy = loaded_data["data_vars"][self.XY][1]

            if all_heights or height is None:
                assert self.H in loaded_data["coords"], (
                    f"States '{self.name}': Missing heights '{self.H}' in loaded_data, got {list(loaded_data['coords'].keys())}"
                )
                h = loaded_data["coords"][self.H]
                if height is None:
                    h = np.atleast_1d(np.max(h))
                elif all_heights:
                    raise ValueError(
                        f"States '{self.name}': Cannot specify both all_heights and height, got all_heights={all_heights}, height={height}"
                    )
            else:
                h = np.atleast_1d(height)

        nx, ny = xy.shape[:2]
        nh = len(h)
        gpts = np.zeros((nx * ny, nh, 3), dtype=config.dtype_double)
        gpts[:, :, :2] = xy.reshape((nx * ny, 1, 2))
        gpts[:, :, 2] = h.reshape((1, nh))
        gpts = gpts.reshape((nx * ny * nh, 3))

        return gpts

    def get_interpolation_grid_data(self, mdata: MData, idims: list[str]) -> np.ndarray:
        """
        Extracts interpolation grid data from chunk model data.

        Parameters
        ----------
        mdata: foxes.core.MData
            The model data
        idims: list of str
            The dimensions for interpolation, e.g. ['x', 'y', 'height']

        Returns
        -------
        gpts: numpy.ndarray
            A 2D array with shape (n_points, n_idims).

        """
        # get coordinates:
        icrds = []
        for c in idims:
            cc = self.var(c) if c not in [FC.STATE, FC.TURBINE] else c
            assert cc in mdata, (
                f"States '{self.name}': Missing coordinate '{cc}' in mdata, got {list(mdata.keys())}"
            )
            icrds.append(mdata[cc])

        # prepare grid points:
        n_dms = len(idims)
        gpts = np.zeros(
            tuple([len(c) for c in icrds]) + (n_dms,), dtype=config.dtype_double
        )
        n_gpts = 1
        ix = None
        for i, c in enumerate(icrds):
            if idims[i] not in (FV.X, FV.Y):
                shp = [1] * n_dms
                shp[i] = c.shape[0]
                gpts[..., i] = c.reshape(shp)
                n_gpts *= c.shape[0]
            elif idims[i] == FV.X:
                assert FV.Y in idims, (
                    f"States '{self.name}': {FV.X} found in dims {idims} but not {FV.Y}"
                )
                ix = i
            else:
                assert ix == i - 1, (
                    f"States '{self.name}': Unexpected dimension order {idims}, expected {FV.X} before {FV.Y}"
                )

        # sneak in xy instead of west_east and south_north coords:
        if ix is not None:
            xy = mdata[self.XY]
            shp = [1] * len(gpts.shape)
            shp[ix : ix + 2] = xy.shape[:2]
            shp[-1] = 2
            gpts[..., ix : ix + 2] = xy.reshape(shp)
            n_gpts *= xy.shape[0] * xy.shape[1]

        # reshape:
        return gpts.reshape((n_gpts, n_dms))

    def interpolate_data(
        self,
        mdata: MData,
        idims: list[str],
        d: np.ndarray,
        pts: np.ndarray,
        vrs: list[str],
        state_indices: np.ndarray | None = None,
        gpts: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Interpolates data to points.

        Parameters
        ----------
        mdata: foxes.core.MData
            The model data
        idims: list of str
            The input dimensions, e.g. ['x', 'y', 'height']
        d: numpy.ndarray
            The data array, with shape (n1, n2, ..., nv)
            where ni represents the dimension sizes of the ordered
            icoords keys, and nv is the number of variables
        pts: numpy.ndarray
            The points to interpolate to, with shape (n_pts, n_idims)
        vrs: list of str
            The variable names, length nv
        state_indices: numpy.ndarray, optional
            The indices of the states, with shape (n_states,)
        gpts: numpy.ndarray or None, optional
            A 2D array with shape (n_points, n_dims), or None to extract the
            grid points from mdata.

        Returns
        -------
        d_interp: numpy.ndarray
            The interpolated data array with shape (n_pts, nv)

        """
        if FC.STATE in idims:
            raise NotImplementedError(
                f"States '{self.name}': Interpolation with state dimension not implemented."
            )

        # prepare interpolation parameters:
        ipars = dict(
            method="linear",
            rescale=True,
            fill_value=np.nan,
        )
        ipars.update(self.interp_pars)

        # get grid points if not provided:
        if gpts is None:
            gpts = self.get_interpolation_grid_data(mdata, idims)
        else:
            assert (
                isinstance(gpts, np.ndarray)
                and gpts.ndim == 2
                and gpts.shape[1] == len(idims)
            ), (
                f"States '{self.name}': gpts must be a 2D numpy array with shape (n_points, {len(idims)}), got {gpts.shape}"
            )

        # check and reshape d, data is on a non-regular grid:
        n_gpts, n_dms = gpts.shape
        if d.shape[0] != n_gpts:
            try:
                d = d.reshape((n_gpts,) + d.shape[n_dms:])
            except Exception as e:
                raise ValueError(
                    f"States '{self.name}': Cannot reshape d with shape {d.shape} to match gpts with shape {gpts.shape} and vrs with length {len(vrs)}"
                ) from e

        def _check_nan(
            gpts: np.ndarray,
            d: np.ndarray,
            pts: np.ndarray,
            idims: list[str],
            results: np.ndarray,
        ) -> None:
            """Checks for NaN results and raises errors."""
            if np.isnan(ipars.get("fill_value", np.nan)):
                assert state_indices is not None, (
                    f"States '{self.name}': state_indices must be provided for NaN check, got None"
                )
                sel = np.isnan(results)
                if np.any(sel):
                    i = [j[0] for j in np.where(sel)]
                    t = state_indices[i.pop(-2)] if len(results.shape) == 3 else None
                    p = pts[tuple(i[:-1])]
                    qmin = np.min(gpts, axis=0)
                    qmax = np.max(gpts, axis=0)
                    isin = (p >= qmin) & (p <= qmax)
                    method = "linear"
                    print("\n\nInterpolation error")
                    print("time:   ", t)
                    print("dims:   ", idims[1:] if FC.STATE in idims else idims)
                    print("point:  ", p)
                    print("qmin:   ", qmin)
                    print("qmax:   ", qmax)
                    print("Inside: ", isin, "\n\n")

                    if not np.all(isin):
                        raise ValueError(
                            f"States '{self.name}': Interpolation method '{method}' failed for {np.sum(sel)} points, e.g. for point {p} at time {t}, outside of bounds {qmin} - {qmax}, dimensions = {idims}. "
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
                            print("   time:   ", t)
                            for ic, c in enumerate(idims):
                                print(f"  {c}: {p[ic]}")
                            for iw, w in enumerate(vrs):
                                print(f"  {w}: {d[i[0][0], iw]}")
                            print("\n\n")
                            raise ValueError(
                                f"States '{self.name}': Interpolation method '{method}' failed, NaN values found in input data for {np.sum(sel)} grid points, e.g. {gpts[i[0]]} at time {t} with {v} = {d[i[0][0], i[1][0]]}."
                            )
                        raise ValueError(
                            f"States '{self.name}': Interpolation method '{method}' failed for {np.sum(sel)} points, for unknown reason."
                        )

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
