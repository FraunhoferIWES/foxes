from xarray import Dataset

from foxes.config import config
from foxes.utils import write_nc
import foxes.constants as FC

from .output import Output


class StateTurbineTable(Output):
    """
    Creates tables of state-turbine type data

    Attributes
    ----------
    farm_results: xarray.Dataset
        The farm results

    :group: output

    """

    def __init__(self, farm_results, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        farm_results: xarray.Dataset
            The farm results
        kwargs: dict, optional
            Additional parameters for the base class

        """
        super().__init__(**kwargs)
        self.farm_results = farm_results

    def get_dataset(
        self,
        variables,
        name_map={},
        to_file=None,
        isel=None,
        sel=None,
        transpose=False,
        **kwargs,
    ):
        """
        Creates a dataset object

        Parameters
        ----------
        variables: list of str
            The output variables
        name_map: dict
            Map from foxes to output names
        to_file: str, optional
            Name of the output file, if writing is desired
        isel: dict, optional
            Parameters for xarray.Dataset.isel
        sel: dict, optional
            Parameters for xarray.Dataset.sel
        transpose: bool, optional
            Whether to transpose the dataset
        kwargs: dict, optional
            Additional parameters for write_nc

        Returns
        -------
        table: xarray.Dataset
            The state-turbine data table

        """
        state = name_map.get(FC.STATE, FC.STATE)
        turbine = name_map.get(FC.TURBINE, FC.TURBINE)

        dvars = {}
        for v in variables:
            data = self.farm_results[v].to_numpy()
            dims = (state, turbine)
            if transpose:
                data = data.T
                dims = (turbine, state)
            dvars[name_map.get(v, v)] = (dims, data)

        ds = Dataset(
            coords={
                state: self.farm_results[FC.STATE].to_numpy(),
                turbine: self.farm_results[FC.TURBINE].to_numpy(),
            },
            data_vars=dvars,
        )

        if isel is not None:
            ds = ds.isel(**isel)
        if sel is not None:
            ds = ds.sel(**sel)

        if to_file is not None:
            fpath = self.get_fpath(to_file)
            write_nc(ds=ds, fpath=fpath, nc_engine=config.nc_engine, **kwargs)

        return ds
