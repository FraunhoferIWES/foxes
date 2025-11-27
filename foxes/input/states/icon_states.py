import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

from foxes.utils import get_utm_zone, from_lonlat
from foxes.config import config, get_output_path
from foxes.output import FarmLayoutOutput
import foxes.variables as FV
import foxes.constants as FC

from .dataset_states import DatasetStates


class ICONStates(DatasetStates):
    """
    Heterogeneous ambient states in DWD-ICON format.

    Attributes
    ----------
    height_level_coord: str, optional
        The height level coordinate name in the data
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
        height_level_coord="height",
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
        height_level_coord: str, optional
            The height level coordinate name in the data
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

        self.height_level_coord = height_level_coord
        self.time_coord = time_coord
        self.lat_coord = lat_coord
        self.lon_coord = lon_coord
        self.bounds_extra_space = bounds_extra_space
        self.height_bounds = height_bounds
        self.icon_point_plot = icon_point_plot
        self.interp_pars = interp_pars if interp_pars is not None else {}
        self.variables = list(set([v if v != FV.TI else FV.TKE for v in ovars]))
        self.utm_zone = utm_zone

        self._cmap = {
            FC.STATE: self.time_coord,
            FV.X: self.lon_coord,
            FV.Y: self.lat_coord,
            FV.H: self.height_level_coord,
        }

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
        