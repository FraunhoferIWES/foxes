import numpy as np

def _read_nondimensional_coordinate(name, data, coords, fields, dims):
    """ read nondimensional coordinate """
    if isinstance(data, (int, float, list, tuple)):
        data = np.array(data)
    if isinstance(data, np.ndarray) and len(data.shape) == 0:
        raise NotImplementedError("nondimensional coordinates are not implemented yet")
    return False

def _read_dimensional_coordinate(name, data, coords, fields, dims):
    """ read dimensional coordinate """
    if isinstance(data, list):
        data = np.array(data)
    if isinstance(data, np.ndarray) and len(data.shape) == 1:
        coords[name] = data
        return True
    return False

def _read_multi_dimensional_coordinate(*args, **kwargs):
    """ Read multi dimensional coordinate """
    return (
        _read_nondimensional_coordinate(*args, **kwargs) or
        _read_dimensional_coordinate(*args, **kwargs)
    )

def _read_nondimensional_data(name, data, coords, fields, dims):
    """ read nondimensional data """
    if isinstance(data, (int, float, list, tuple)):
        data = np.array(data)
    if isinstance(data, np.ndarray) and len(data.shape) == 0:
        fields[name] = data
        dims[name] = []
        return True
    return False

def _read_dimensional_data(name, data, coords, fields, dims):
    """ read dimensional data """
    if isinstance(data, dict) and "data" in data and "dims" in data:
        d = data["data"]
        fields[name] = d if isinstance(d, np.ndarray) else np.array(d)
        dims[name] = data["dims"]
        if len(dims[name]) != len(fields[name].shape):
            raise ValueError(f"Field '{name}': Dimensions {dims[name]} do not match shape {fields[name].shape}")
        return True
    return False

def _read_multi_dimensional_data(*args, **kwargs):
    """ Read multi dimensional data """
    return (
        _read_nondimensional_data(*args, **kwargs) or
        _read_dimensional_data(*args, **kwargs)
    )

def read_wind_resource_field(name, *args, **kwargs):
    """ Reads wind resource data into fields and dims """
    if (
        name in ["time", "wind_turbine"] and
        _read_multi_dimensional_coordinate(name, *args, **kwargs)
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
            _read_multi_dimensional_coordinate(name, *args, **kwargs) or
            _read_multi_dimensional_data(name, *args, **kwargs)
        )
    ):
        return True
    
    elif (
        name in [
            "probability", 
            "weibull_a", 
            "weibull_k", 
            "sector_probability", 
            "turbulence_intensity",
            "LMO",
            "z0",
            "k",
            "epsilon",
            "potential_temperature",
            "friction_velocity",
        ] and
        _read_multi_dimensional_data(name, *args, **kwargs)
    ):
        return True
        
    else:
        raise NotImplementedError(f"No reading method implemented for field '{name}'")
    