import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pandas import read_csv

from foxes.utils import get_utm_zone, to_lonlat
from foxes.config import config, get_output_path
from foxes.output import FarmLayoutOutput
from foxes.data import MODEL_DATA
import foxes.variables as FV
import foxes.constants as FC

from .dataset_states import DatasetStates


class ICONStates(DatasetStates):
    """
    Heterogeneous ambient states in DWD-ICON format.

    Attributes
    ----------
    height_coord_default: str, optional
        The default height level coordinate name in the data
    height_coord_tke: str, optional
        The height level coordinate name for TKE in the data
    time_coord: str
        The time coordinate name in the data
    lat_coord: str
        The latitude coordinate name in the data
    lon_coord: str
        The longitude coordinate name in the data
    bounds_extra_space: float or str, optional
        The extra space, either float in m,
        or str for units of D, e.g. '2.5D'
    height_bounds: tuple, optional
        The (h_min, h_max) height bounds in m. Defaults to H +/- 0.5*D
    interp_pars: dict, optional
        Additional parameters for scipy.interpolate.griddata,
        e.g. {'method': 'linear', 'fill_value': None, 'rescale': True}
    icon_point_plot: str, optional
        Path to a plot file, e.g. wrf_points.png, to visualize the
        selected ICON grid points and the layout of the farm.
    utm_zone: str or tuple, optional
        Method for setting UTM zone in config, if not already set.
        Options are:
        - "from_grid": get UTM zone from the centre of the (lon, lat) grid
        - "XA": use given number X, letter A
        - (lon, lat): use given lon, lat values
        - None: do not set UTM zone, assume it is already set, 
        typically during the wind farm creation.

    :group: input.states

    """

    def __init__(
        self,
        input_files_nc,
        height_coord_default="height",
        height_coord_tke="height_2",
        time_coord="time",
        lat_coord="lat",
        lon_coord="lon",
        output_vars=None,
        var2ncvar=None,
        load_mode="fly",
        time_format=None,
        bounds_extra_space=0.0,
        height_bounds=None,
        interp_pars=None,
        icon_point_plot=None,
        utm_zone=None,
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        input_files_nc: str
            The input netcdf file(s) containing, can contain
            wildcards, e.g. '2025*_icon.nc'
        height_coord_default: str, optional
            The default height level coordinate name in the data
        height_coord_tke: str, optional
            The height level coordinate name for TKE in the data
        time_coord: str
            The time coordinate name in the data
        lat_coord: str
            The latitude coordinate name in the data
        lon_coord: str
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
        interp_pars: dict, optional
            Additional parameters for scipy.interpolate.griddata,
            e.g. {'method': 'linear', 'fill_value': None, 'rescale': True}
        icon_point_plot: str, optional
            Path to a plot file, e.g. wrf_points.png, to visualize the
            selected ICON grid points and the layout of the farm.
        utm_zone: str or tuple, optional
            Method for setting UTM zone in config, if not already set.
            Options are:
            - "from_grid": get UTM zone from the centre of the (lon, lat) grid
            - "XA": use given number X, letter A
            - (lon, lat): use given lon, lat values
            - None: do not set UTM zone, assume it is already set, 
            typically during the wind farm creation.
        kwargs: dict, optional
            Additional parameters for the base class

        """
        if output_vars is None:
            ovars = [FV.WS, FV.WD, FV.TI, FV.RHO]
        else:
            ovars = output_vars

        if var2ncvar is None:
            var2ncvar = {
                FV.U: "U",
                FV.V: "V",
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
        self.lat_coord = lat_coord
        self.lon_coord = lon_coord
        self.bounds_extra_space = bounds_extra_space
        self.height_bounds = height_bounds
        self.icon_point_plot = icon_point_plot
        self.interp_pars = interp_pars if interp_pars is not None else {}
        self._prepr0 = self.preprocess_nc
        self.preprocess_nc = self._preproc_icon_nc
        self.__utm_zone = utm_zone

        self.variables = []
        for v in ovars:
            if v == FV.TI:
                self.variables.append(FV.TKE)
            elif v == FV.WS or v == FV.WD:
                if FV.U not in self.variables:
                    assert FV.WS in ovars and FV.WD in ovars, (
                        f"{self.name}: Both FV.WS and FV.WD must be requested if one of them is requested."
                    )
                    self.variables.extend([FV.U, FV.V])
            elif v == FV.RHO:
                self.variables.append(FV.T)
                self.variables.append(FV.p)
            elif v not in self.variables:
                self.variables.append(v)

        # longitude and latitude play the role of x and y here:
        self._cmap = {
            FC.STATE: self.time_coord,
            FV.X: self.lon_coord,
            FV.Y: self.lat_coord,
        }
        if height_coord_default is not None:
            self._cmap[FV.H] = height_coord_default
        if height_coord_tke is not None:
            self.H_TKE = FV.H + "_tke"
            self._cmap[self.H_TKE] = height_coord_tke

    def _preproc_icon_nc(self, ds):
        """Preprocess ICON netcdf dataset."""
        if FV.H in self._cmap and self._cmap[FV.H] in ds.sizes:
            c = ds[self._cmap[FV.H]].values.astype(int)
            ds = ds.assign_coords({self._cmap[FV.H]: self.__icon_heights_default[c]})
        if self.H_TKE in self._cmap and self._cmap[self.H_TKE] in ds.sizes:
            c = ds[self._cmap[self.H_TKE]].values.astype(int)
            ds = ds.assign_coords({self._cmap[self.H_TKE]: self.__icon_heights_TKE[c]})
        return self._prepr0(ds) if self._prepr0 is not None else ds

    def _find_xy_bounds(self, algo, bounds_extra_space):
        """Helper function to determine x/y bounds with extra space."""
        xy_min, xy_max = algo.farm.get_xy_bounds(
            extra_space=bounds_extra_space, algo=algo
        )
        xy_min = to_lonlat(xy_min[None, :])[0]
        xy_max = to_lonlat(xy_max[None, :])[0]
        return xy_min, xy_max

    def preproc_first(
        self, algo, data, cmap, vars, bounds_extra_space, height_bounds, verbosity=0
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
            algo, data, cmap, vars, bounds_extra_space, height_bounds, verbosity
        )
        if not config.utm_zone_set and self.__utm_zone is None:
            raise ValueError(
                f"States '{self.name}': config.utm_zone is not set and no utm_zone argument given."
            )
        if self.__utm_zone is None:
            zone = config.utm_zone
        elif self.__utm_zone == "from_grid":
            lonlat = np.stack(
                [
                    0.5 * (data[cmap[FV.X]].values.min() + data[cmap[FV.X]].values.max()),
                    0.5 * (data[cmap[FV.Y]].values.min() + data[cmap[FV.Y]].values.max()),
                ]
            )
            zone = get_utm_zone(lonlat[None, :])
        elif isinstance(self.__utm_zone, str):
            zone = (int(self.__utm_zone[:-1]), self.__utm_zone[-1])
        elif len(self.__utm_zone) == 2:
            lonlat = np.asarray(self.__utm_zone)
            zone = get_utm_zone(lonlat[None, :])
        else:
            raise ValueError(
                f"States '{self.name}': invalid utm_zone argument: {self.__utm_zone}"
            )
        if not config.utm_zone_set:
            config.set_utm_zone(*zone)
        elif config.utm_zone != zone:
            raise ValueError(
                f"States '{self.name}': config.utm_zone = {config.utm_zone} differs from determined zone {zone}"
            )
        
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
        # read mapping from height levels to heights from static csv files:
        hdata = {}
        for A in ("A1", "A2"):
            fpath =  algo.dbook.get_file_path(
                MODEL_DATA, f"icon_heights_{A}.csv", check_raw=False
            )
            algo.print(f"States '{self.name}': Loading '{fpath.stem}'")
            hdata[A] = read_csv(fpath, index_col="eu_nest")["height"].to_numpy()
        self.__icon_heights_TKE = hdata["A1"]
        self.__icon_heights_default = hdata["A2"]

        return super().load_data(
            algo,
            cmap=self._cmap,
            variables=self.variables,
            bounds_extra_space=self.bounds_extra_space,
            height_bounds=self.height_bounds,
            verbosity=verbosity,
        )

    def _update_dims(self, dims, coords, vrs, d):
        """Helper function for dimension adjustment, if needed"""
        if self.H_TKE in dims:
            assert FV.H not in dims, (
                f"States {self.name}: Cannot have both {FV.H} and {self.H_TKE} in dims for variables {vrs}, got dims = {dims}"
            )
            dims_new = list(dims)
            dims_new[dims.index(self.H_TKE)] = FV.H
            dims = tuple(dims_new)
            coords = coords.copy()
            coords[FV.H] = coords[self.H_TKE]
        return dims, coords

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
        # convert (x, y) to (lon, lat) before interpolation:
        if FV.X in idims:
            ix = idims.index(FV.X)
            assert len(idims) > ix + 1 and idims[ix + 1] == FV.Y, (
                f"States {self.name}: Expecting subsequent ({FV.X}, {FV.Y}) in idims, got {idims}"
            )
            pts[:, ix:ix + 2] = to_lonlat(pts[:, ix:ix + 2])

        return super().interpolate_data(idims, icrds, d, pts, vrs, times)
    