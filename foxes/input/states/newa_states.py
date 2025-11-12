import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from foxes.utils.utm_utils import from_lonlat
from foxes.config import config, get_output_path
from foxes.output import FarmLayoutOutput
import foxes.variables as FV
import foxes.constants as FC

from .dataset_states import DatasetStates


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
    bounds_extra_space: float or str
        The extra space, either float in m,
        or str for units of D, e.g. '2.5D'
    height_bounds: tuple, optional
        The (h_min, h_max) height bounds in m. Defaults to H +/- 0.5*D

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
        input_files_nc,
        time_coord="time",
        west_east_coord="west_east",
        south_north_coord="south_north",
        height_coord="height",
        xlat_coord="XLAT",
        xlon_coord="XLON",
        output_vars=None,
        var2ncvar=None,
        load_mode="fly",
        time_format=None, 
        bounds_extra_space=0.0,
        height_bounds=None,
        interp_pars=None,
        wrf_point_plot=None,
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        input_files_nc: str
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
        var2ncvar: dict, optional
            A dictionary mapping foxes variable names
            to the corresponding netcdf variable names.
        load_mode: str
            The load mode, choices: preload, lazy, fly.
            preload loads all data during initialization,
            lazy lazy-loads the data using dask, and fly
            reads only states index and weights during initialization
            and then opens the relevant files again within
            the chunk calculations.
        time_format: str
            The datetime parsing format string
        bounds_extra_space: float or str, optional
            The extra space, either float in m,
            or str for units of D, e.g. '2.5D'
        height_bounds: tuple, optional
            The (h_min, h_max) height bounds in m. Defaults to H +/- 0.5*D
        kwargs: dict, optional
            Additional parameters for the base class
        interp_pars: dict, optional
            Additional parameters for scipy.interpolate.griddata,
            e.g. {'method': 'linear', 'fill_value': None, 'rescale': True}
        wrf_point_plot: str, optional
            Path to a plot file, e.g. wrf_points.png, to visualize the
            selected WRF grid points and the layout of the farm.
        kwargs: dict, optional
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
            **kwargs,
        )

        self.time_coord = time_coord
        self.west_east_coord = west_east_coord
        self.south_north_coord = south_north_coord
        self.height_coord = height_coord
        self.xlat_coord = xlat_coord
        self.xlon_coord = xlon_coord
        self.bounds_extra_space = bounds_extra_space
        self.height_bounds = height_bounds
        self.wrf_point_plot = wrf_point_plot
        self.interp_pars = interp_pars if interp_pars is not None else {}
        self.variables = list(set([v if v != FV.TI else FV.TKE for v in ovars]))

        self._cmap = {
            FC.STATE: self.time_coord,
            FV.X: self.west_east_coord,
            FV.Y: self.south_north_coord,
            FV.H: self.height_coord,
        }

    def preproc_first(
        self,
        algo,
        data,
        cmap,
        vars,
        bounds_extra_space,
        height_bounds,
        verbosity=0,
    ):
        """
        Preprocesses the first file.

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The calculation algorithm
        data: xarray.Dataset
            The dataset to preprocess
        cmap: dict
            A mapping from foxes variable names to Dataset dimension names
        vars: list
            The list of variable names
        bounds_extra_space: float or str, optional
            The extra space, either float in m,
            or str for units of D, e.g. '2.5D'
        height_bounds: tuple, optional
            The (h_min, h_max) height bounds in m. Defaults to H +/-
        verbosity: int
            The verbosity level, 0 = silent

        """

        super().preproc_first(
            algo,
            data,
            cmap,
            vars,
            bounds_extra_space=None,
            height_bounds=height_bounds,
            verbosity=verbosity,
        )

        lonlat = np.stack(
            (data[self.xlon_coord].values, data[self.xlat_coord].values), axis=-1
        )
        lonlat = np.moveaxis(lonlat, 0, 1)  # (y, x, 2) to (x, y, 2)
        nx, ny = lonlat.shape[:2]
        lonlat = lonlat.reshape((nx * ny, 2))
        xy = from_lonlat(lonlat)
        xy = xy.reshape((nx, ny, 2))
        nh = len(self._heights)

        # find horizontal bounds:
        if bounds_extra_space is not None:
            assert FV.X in cmap, (
                f"States '{self.name}': x coordinate '{FV.X}' not in cmap {cmap}"
            )
            assert FV.Y in cmap, (
                f"States '{self.name}': y coordinate '{FV.Y}' not in cmap {cmap}"
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
                (xy[..., 0] >= x0)
                & (xy[..., 0] <= x1)
                & (xy[..., 1] >= y0)
                & (xy[..., 1] <= y1)
            )
            assert len(inds) > 0, (
                f"States '{self.name}': No grid points found within bounds (x0, x1)=({x0}, {x1}), (y0, y1)=({y0}, {y1})"
            )
            i0 = inds[:, 0].min()
            i1 = inds[:, 0].max()
            j0 = inds[:, 1].min()
            j1 = inds[:, 1].max()
            while True:
                self._xy = xy[i0 : i1 + 1, j0 : j1 + 1]
                if i0 > 0 and x0 < np.min(self._xy[..., 0]):
                    i0 -= 1
                elif i1 < nx - 1 and x1 > np.max(self._xy[..., 0]):
                    i1 += 1
                elif j0 > 0 and y0 < np.min(self._xy[..., 1]):
                    j0 -= 1
                elif j1 < ny - 1 and y1 > np.max(self._xy[..., 1]):
                    j1 += 1
                else:
                    break
            nx, ny = self._xy.shape[:2]

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
                    f"States '{self.name}': Selected {FV.X} = {np.min(self._xy[..., 0]):.2f} - {np.max(self._xy[..., 0]):.2f} ({nx} points)"
                )
                print(
                    f"States '{self.name}': Selected {FV.Y} = {np.min(self._xy[..., 1]):.2f} - {np.max(self._xy[..., 1]):.2f} ({ny} points)"
                )
                print(
                    f"States '{self.name}': Selected {self._xy.shape[:2] + (nh,)} grid points"
                )
        else:
            self._xy = xy
            if verbosity > 0:
                print(
                    f"States '{self.name}': Selecting all {self._xy.shape[:2] + (nh,)} grid points"
                )

        if self.wrf_point_plot is not None:
            fpath = get_output_path(self.wrf_point_plot)
            if verbosity > 0:
                print(f"States '{self.name}': Writing WRF grid point plot to '{fpath}'")
            fig, ax = plt.subplots(figsize=(8, 8))
            ax.plot(
                self._xy[..., 0].flatten(),
                self._xy[..., 1].flatten(),
                c="blue",
                alpha=0.2,
                marker=".",
                linestyle="None",
            )
            anno = 3 if len(algo.farm.wind_farm_names) > 1 else 0
            FarmLayoutOutput(farm=algo.farm).get_figure(
                fig=fig, ax=ax, annotate=anno, fontsize=12
            )
            ax.set_xlabel(f"{FV.X} [m]")
            ax.set_ylabel(f"{FV.Y} [m]")
            ax.set_aspect("equal", adjustable="box")
            ax.autoscale_view(tight=True)
            fig.savefig(fpath, bbox_inches="tight")
            plt.close()

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
            height_bounds=self.height_bounds,
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
                    t = times[i.pop(-2)] if len(results.shape) == 3 else None
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
        if FC.STATE in idims:
            raise NotImplementedError(
                f"States '{self.name}': Interpolation with state dimension not implemented."
            )

        # prepare grid points:
        n_dms = len(idims)
        gpts = np.zeros(d.shape[:n_dms] + (n_dms,), dtype=config.dtype_double)
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

        # sneak in self._xy instead of west_east and south_north coords:
        if ix is not None:
            shp = [1] * len(gpts.shape)
            shp[ix : ix + 2] = self._xy.shape[:2]
            shp[-1] = 2
            gpts[..., ix : ix + 2] = self._xy.reshape(shp)
            n_gpts *= self._xy.shape[0] * self._xy.shape[1]

        # reshape:
        gpts = gpts.reshape((n_gpts, n_dms))
        d = d.reshape((n_gpts,) + d.shape[n_dms:])

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
        results = super().calculate(algo, mdata, fdata, tdata)

        # convert TKE to TI if needed:
        if FV.TI in self.ovars and FV.TI not in results:
            assert FV.WS in results, (
                f"States '{self.name}': Cannot calculate {FV.TI} without {FV.WS}"
            )
            assert FV.TKE in results or FV.TKE in self.ovars, (
                f"States '{self.name}': Cannot calculate {FV.TI} without {FV.TKE}"
            )
            if FV.TKE not in self.ovars:
                tke = results.pop(FV.TKE)
            else:
                tke = results[FV.TKE]
            ws = results[FV.WS]
            results[FV.TI] = np.sqrt(1.5 * tke) / ws

        return results
