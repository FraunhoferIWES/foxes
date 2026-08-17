import numpy as np
from numbers import Number
from typing import Any

import foxes.variables as FV
import foxes.constants as FC


""" Mapping from windio to foxes variables
:group: input.yaml.windio
"""
wio2foxes = {
    "time": FC.TIME,
    "x": FV.X,
    "y": FV.Y,
    "height": FV.H,
    "wind_turbine": FC.TURBINE,
    "wind_direction": FV.WD,
    "wind_speed": FV.WS,
    "density": FV.RHO,
    "probability": FV.WEIGHT,
    "sector_probability": "sector_probability",
    "turbulence_intensity": FV.TI,
    "LMO": FV.MOL,
    "z0": FV.Z0,
    "reference_height": FV.H,
    "weibull_a": FV.WEIBULL_A,
    "weibull_k": FV.WEIBULL_k,
    "operating": FV.OPERATING,
}

""" Mapping from foxes to windio variables
:group: input.yaml.windio
"""
foxes2wio = {d: k for k, d in wio2foxes.items()}


def _read_nondimensional_coordinate(
    name: str, wio_data: Any, coords: dict[str, Any]
) -> bool:
    """read nondimensional coordinate
    :group: input.yaml.windio
    """
    if isinstance(wio_data, Number):
        coords[wio2foxes[name]] = wio_data
        return True
    return False


def _read_dimensional_coordinate(
    name: str, wio_data: Any, coords: dict[str, Any]
) -> bool:
    """read dimensional coordinate
    :group: input.yaml.windio
    """
    if isinstance(wio_data, list):
        wio_data = np.array(wio_data)
    if isinstance(wio_data, np.ndarray) and len(wio_data.shape) == 1:
        coords[wio2foxes[name]] = wio_data
        return True
    return False


def _read_multi_dimensional_coordinate(
    name: str, wio_data: Any, coords: dict[str, Any]
) -> bool:
    """Read multi dimensional coordinate
    :group: input.yaml.windio
    """
    return _read_nondimensional_coordinate(
        name, wio_data, coords
    ) or _read_dimensional_coordinate(name, wio_data, coords)


def _read_nondimensional_data(
    name: str,
    wio_data: Any,
    fields: dict[str, Any],
    dims: dict[str, Any],
) -> bool:
    """read nondimensional data
    :group: input.yaml.windio
    """
    if isinstance(wio_data, Number):
        v = wio2foxes[name]
        fields[v] = wio_data
        dims[v] = []
        return True
    return False


def _read_dimensional_data(
    name: str,
    wio_data: Any,
    fields: dict[str, Any],
    dims: dict[str, Any],
) -> bool:
    """read dimensional data
    :group: input.yaml.windio
    """
    if isinstance(wio_data, dict) and "data" in wio_data and "dims" in wio_data:
        d = wio_data["data"]
        v = wio2foxes[name]
        fields[v] = d if isinstance(d, np.ndarray) else np.array(d)
        dims[v] = tuple([wio2foxes[c] for c in wio_data["dims"]])
        if len(dims[v]) != len(fields[v].shape):
            raise ValueError(
                f"Field '{name}': Dimensions {dims[v]} do not match shape {fields[v].shape}"
            )
        return True
    return False


def _read_multi_dimensional_data(
    name: str,
    wio_data: Any,
    fields: dict[str, Any],
    dims: dict[str, Any],
) -> bool:
    """Read multi dimensional data
    :group: input.yaml.windio
    """
    return _read_nondimensional_data(
        name, wio_data, fields, dims
    ) or _read_dimensional_data(name, wio_data, fields, dims)


def read_wind_resource_field(
    name: str,
    wio_data: Any,
    coords: dict[str, Any],
    fields: dict[str, Any],
    dims: dict[str, Any],
    verbosity: int,
) -> bool:
    """
    Reads wind resource data into fields and dims

    Parameters
    ----------
    name
        The windio variable name
    wio_data
        The windio data
    coords
        The coordinates dict, filled on the fly
    fields
        The fields dict, filled on the fly
    dims
        The dimensions dict, filled on the fly
    verbosity
        The verbosity level, 0=silent

    Returns
    -------
    success
        Flag for successful data extraction

    :group: input.yaml.windio

    """
    if name in [
        "potential_temperature",
        "real_temperature",
        "friction_velocity",
        "k",
        "epsilon",
        "ABL_height",
        "lapse_rate",
        "capping_inversion_thickness",
        "capping_inversion_strength",
        "tau_x",
        "tau_y",
        "fc",
    ]:
        if verbosity > 2:
            print(f"        Ignoring variable '{name}'")
        return False

    if verbosity > 2:
        print(f"        Reading variable '{name}'")
    if name in ["time", "wind_turbine"] and _read_multi_dimensional_coordinate(
        name, wio_data, coords
    ):
        return True

    elif name in [
        "wind_direction",
        "wind_speed",
        "x",
        "y",
        "height",
    ] and (
        _read_multi_dimensional_coordinate(name, wio_data, coords)
        or _read_multi_dimensional_data(name, wio_data, fields, dims)
    ):
        return True

    elif name in [
        "probability",
        "sector_probability",
        "turbulence_intensity",
        "density",
        "LMO",
        "z0",
        "reference_height",
        "weibull_a",
        "weibull_k",
        "operating",
    ] and _read_multi_dimensional_data(name, wio_data, fields, dims):
        return True

    else:
        raise NotImplementedError(f"No reading method implemented for field '{name}'")
