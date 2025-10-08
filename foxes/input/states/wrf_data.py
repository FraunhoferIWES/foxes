import foxes.variables as FV

from .field_data import FieldData


class WRFData(FieldData):
    """
    Heterogeneous ambient states on a regular
    horizontal grid in longitude, latitude coordinates.

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
        input_files_nc,
        time_coord="Time",
        lon_coord="lon",
        lat_coord="lat",
        h_coord="height",
        load_mode="fly",
        output_vars=None,
        # *args,
        # states_coord="Time",
        # x_coord="UTMX",
        # y_coord="UTMY",
        # h_coord="height",
        # weight_ncvar=None,
        # bounds_extra_space=1000,
        # interpn_pars={},
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
        lon_coord: str
            The longitude coordinate name in the data
        lat_coord: str
            The latitude coordinate name in the data
        h_coord: str, optional
            The height coordinate name in the data
        load_mode: str
            The load mode, choices: preload, lazy, fly.
            preload loads all data during initialization,
            lazy lazy-loads the data using dask, and fly
            reads only states index and weights during initialization
            and then opens the relevant files again within
            the chunk calculation

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
        if output_vars is None:
            ovars = [FV.WS, FV.WD, FV.TI, FV.RHO]
        else:
            ovars = output_vars
            
        super().__init__(
            data_source=input_files_nc,
            states_coord=time_coord,
            x_coord=lon_coord,
            y_coord=lat_coord,
            h_coord=h_coord,
            weight_ncvar=None,
            load_mode=load_mode,
            output_vars=ovars,
            **kwargs,
        )
