from xarray import Dataset

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
    def __init__(self, farm_results):
        """
        Constructor.

        Parameters
        ----------
        farm_results: xarray.Dataset
            The farm results

        """
        self.farm_results = farm_results
    
    def get_dataset(
        self, 
        variables, 
        name_map={}, 
        to_file=None, 
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
            The output file path, if writing is desired
        kwargs: dict, optional
            Additional parameters for write_nc
        
        Returns
        -------
        table: xarray.Dataset
            The state-turbine data table
            
        """
        state = name_map.get(FC.STATE, FC.STATE)
        turbine = name_map.get(FC.TURBINE, FC.TURBINE)

        ds = Dataset(
            coords={
                state: self.farm_results[FC.STATE].to_numpy(),
                turbine: self.farm_results[FC.TURBINE].to_numpy(),
            },
            data_vars={
                name_map.get(v, v): (
                    (state, turbine), self.farm_results[v].to_numpy())
                for v in variables
            }
        )
        
        if to_file is not None:
            write_nc(ds=ds, fpath=to_file, **kwargs)
        
        return ds
    