import pandas as pd
from numbers import Number

from foxes.core import States
import foxes.constants as FC
import foxes.variables as FV

ovars = [FV.WS, FV.WD, FV.TI, FV.RHO]
fixval = {FV.TI: 0.05, FV.RHO: 1.225}


def _get_SingleStateStates(coords, fields, dims, states_dict, verbosity):
    """Try to generate single state parameters
    :group: input.windio
    """
    for c in coords:
        if not isinstance(c, Number):
            return False

    if verbosity > 1:
        print("        selecting class 'SingleStateStates'")

    smap = {FV.WS: "ws", FV.WD: "wd", FV.TI: "ti", FV.RHO: "rho"}

    data = {smap[v]: d for v, d in fixval.items()}
    for v, d in coords.items():
        if v in smap:
            data[smap[v]] = d
        elif verbosity > 1:
            print(f"        ignoring coord '{v}'")
    for v, d in fields.items():
        if v in smap and len(dims[v]) == 0:
            data[smap[v]] = d
        elif verbosity > 1:
            print(f"        ignoring field '{v}' with dims {dims[v]}")

    sdata = pd.DataFrame(index=coords[FC.TIME], data=data)
    sdata.index.name = FC.TIME
    states_dict.update(
        dict(
            states_type="SingleStateStates",
            **data,
        )
    )
    return True


def _get_Timeseries(coords, fields, dims, states_dict, verbosity):
    """Try to generate time series parameters
    :group: input.windio
    """
    if len(coords) == 1 and FC.TIME in coords:
        if verbosity > 1:
            print("        selecting class 'Timeseries'")

        data = {}
        fix = {}
        for v, d in fields.items():
            if dims[v] == (FC.TIME,):
                data[v] = d
            elif len(dims[v]) == 0:
                fix[v] = d
            elif verbosity > 1:
                print(f"        ignoring field '{v}' with dims {dims[v]}")
        fix.update({v: d for v, d in fixval.items() if v not in data})

        sdata = pd.DataFrame(index=coords[FC.TIME], data=data)
        sdata.index.name = FC.TIME
        states_dict.update(
            dict(
                states_type="Timeseries",
                data_source=sdata,
                output_vars=ovars,
                fixed_vars=fix,
            )
        )
        return True
    return False


def get_states(coords, fields, dims, verbosity=1):
    """
    Reads states parameters from windio input

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
        print("      Preparing states")

    states_dict = {}
    if _get_SingleStateStates(
        coords, fields, dims, states_dict, verbosity
    ) or _get_Timeseries(coords, fields, dims, states_dict, verbosity):
        return States.new(**states_dict)
    else:
        raise ValueError(
            f"Failed to create states for coords {list(coords.keys())} and fields {list(fields.keys())} with dims {dims}"
        )
