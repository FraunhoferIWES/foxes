import numpy as np

import foxes.constants as FC
import foxes.variables as FV

""" Mapping from windio to foxes variables
:group: input.windio
"""
wio2foxes = {
    "time": FC.TIME,
    "x": FV.X,
    "y": FV.Y,
    "height": FV.H,
    "wind_turbine": FC.TURBINE,
    "wind_direction": FV.WD, 
    "wind_speed": FV.WS,
    "probability": FV.WEIGHT, 
    "sector_probability": "sector_probability", 
    "turbulence_intensity": FV.TI,
    "LMO": FV.MOL,
    "z0": FV.Z0,
}

""" Mapping from foxes to windio variables
:group: input.windio
"""
foxes2wio = {d: k for k, d in wio2foxes.items()}

def _read_nondimensional_coordinate(name, data, coords, fields, dims):
    """ read nondimensional coordinate 
    :group: input.windio
    """
    if isinstance(data, (int, float, list, tuple)):
        data = np.array(data)
    if isinstance(data, np.ndarray) and len(data.shape) == 0:
        raise NotImplementedError("nondimensional coordinates are not implemented yet")
    return False

def _read_dimensional_coordinate(name, data, coords, fields, dims):
    """ read dimensional coordinate 
    :group: input.windio
    """
    if isinstance(data, list):
        data = np.array(data)
    if isinstance(data, np.ndarray) and len(data.shape) == 1:
        coords[wio2foxes[name]] = data
        return True
    return False

def _read_multi_dimensional_coordinate(*args, **kwargs):
    """ Read multi dimensional coordinate 
    :group: input.windio
    """
    return (
        _read_nondimensional_coordinate(*args, **kwargs) or
        _read_dimensional_coordinate(*args, **kwargs)
    )

def _read_nondimensional_data(name, data, coords, fields, dims):
    """ read nondimensional data 
    :group: input.windio
    """
    if isinstance(data, (int, float, list, tuple)):
        data = np.array(data)
    if isinstance(data, np.ndarray) and len(data.shape) == 0:
        v = wio2foxes[name]
        fields[v] = data
        dims[v] = []
        return True
    return False

def _read_dimensional_data(name, data, coords, fields, dims):
    """ read dimensional data 
    :group: input.windio
    """
    if isinstance(data, dict) and "data" in data and "dims" in data:
        d = data["data"]
        v = wio2foxes[name]
        fields[v] = d if isinstance(d, np.ndarray) else np.array(d)
        dims[v] = tuple([wio2foxes[c] for c in data["dims"]])
        if len(dims[v]) != len(fields[v].shape):
            raise ValueError(f"Field '{name}': Dimensions {dims[v]} do not match shape {fields[v].shape}")
        return True
    return False

def _read_multi_dimensional_data(*args, **kwargs):
    """ Read multi dimensional data 
    :group: input.windio 
    """
    return (
        _read_nondimensional_data(*args, **kwargs) or
        _read_dimensional_data(*args, **kwargs)
    )

def read_wind_resource_field(name, data, coords, fields, dims):
    """ 
    Reads wind resource data into fields and dims

    Parameters
    ----------
    name: str
        The windio variable name
    data: object
        The windio data
    coords: dict
        The coordinates dict, filled on the fly
    fields: dict
        The fields dict, filled on the fly    
    dims: dict
        The dimensions dict, filled on the fly

    Returns
    -------
    success: bool
        Flag for successful data extraction

    :group: input.windio

    """
    if name in [
        "weibull_a", 
        "weibull_k", 
        "potential_temperature",
        "friction_velocity",
        "k",
        "epsilon",
    ]:
        print(f"Ignoring variable '{name}'")
        return False
        
    elif (
        name in ["time", "wind_turbine"] and
        _read_multi_dimensional_coordinate(name, data, coords, fields, dims)
    ):
        return True
    
    elif (
        name in [
            "wind_direction", 
            "wind_speed", 
            "x", 
            "y", 
            "height",
        ] and (
            _read_multi_dimensional_coordinate(name, data, coords, fields, dims) or
            _read_multi_dimensional_data(name, data, coords, fields, dims)
        )
    ):
        return True
    
    elif (
        name in [
            "probability", 
            "sector_probability", 
            "turbulence_intensity",
            "LMO",
            "z0",
            "k",
        ] and
        _read_multi_dimensional_data(name, data, coords, fields, dims)
    ):
        return True
        
    else:
        raise NotImplementedError(f"No reading method implemented for field '{name}'")
    