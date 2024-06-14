import pandas as pd

from foxes.input.states import Timeseries
import foxes.constants as FC
import foxes.variables as FV

ovars = [FV.WS, FV.WD, FV.TI, FV.RHO]
fixval = {FV.TI: 0.05, FV.RHO: 1.225}

def _create_Timeseries(coords, fields, dims, results, verbosity):
    """ Create time series 
    :group: input.windio
    """
    if len(coords) == 1 and FC.TIME in coords:
        if verbosity > 1:
            print("        states class:", Timeseries.__name__)
        data = {f: d for f, d in fields.items() if dims[f] == (FC.TIME,)}
        sdata = pd.DataFrame(index=coords[FC.TIME], data=data)
        sdata.index.name = FC.TIME
        fix = {v: d for v, d in fixval.items() if v not in data}
        results["states"] = Timeseries(sdata, output_vars=ovars, fixed_vars=fix)
        return True
    return False

def create_states(coords, fields, dims, verbosity=1):
    """
    Creates states from windio input

    Parameters
    ----------
    coords: dict
        The coordinates data
    fields: dict
        The fields data
    dims: dict
        The dimensions data
    verbosity: int
        The verbosity level
    
    Returns
    -------
    states: foxes.core.States
        The states object
    
    :group: input.windio

    """
    if verbosity > 1:
        print("      Creating states")
    
    results = {}
    if (
        _create_Timeseries(coords, fields, dims, results, verbosity)
    ):
        return results["states"]
    else:
        raise ValueError(f"Failed to create states for coords {list(coords.keys())} and fields {list(fields.keys())} with dims {dims}")