import numpy as np
import pandas as pd
from xarray import Dataset
from numbers import Number

from foxes.core import States
from foxes.utils import Dict
import foxes.variables as FV
import foxes.constants as FC

from .read_fields import read_wind_resource_field


default_values = {
    FV.WS: 8.0,
    FV.WD: 270.0,
    FV.TI: 0.1,
    FV.RHO: 1.225,
}

def _get_profiles(coords, fields, dims, ovars, fixval, verbosity):
    """Read ABL profiles information
    :group: input.yaml.windio
    """
    profiles = {}
    if FV.Z0 in fields:
        if FV.H not in fields:
            if verbosity > 0:
                print(
                    f"Ignoring '{FV.Z0}', since no reference_height found. No ABL profile activated."
                )
            fields.pop(FV.Z0)
            dims.pop(FV.Z0)
        elif FV.MOL in fields:
            ovars.append(FV.MOL)
            fixval[FV.H] = fields[FV.H]
            profiles = {FV.WS: "ABLLogWsProfile"}
        else:
            fixval[FV.H] = fields[FV.H]
            profiles = {FV.WS: "ABLLogNeutralWsProfile"}
    elif FV.H in fields and verbosity > 0:
        print(
            f"Ignoring '{FV.H}', since no '{FV.Z0}' data found. No ABL profile activated."
        )
    if len(profiles) and verbosity > 2:
        print(
            f"        Selecting ABL profile '{profiles[FV.WS]}', {FV.H} = {fields[FV.H]} m"
        )

    return profiles


def _get_SingleStateStates(
    coords, fields, dims, states_dict, ovars, fixval, profiles, verbosity
):
    """Try to generate single state parameters
    :group: input.yaml.windio
    """
    for c in coords:
        if not isinstance(c, Number):
            return False

    if verbosity > 2:
        print("        selecting class 'SingleStateStates'")

    smap = {FV.WS: "ws", FV.WD: "wd", FV.TI: "ti", FV.RHO: "rho"}

    data = {smap[v]: fixval.get(v, default_values[v]) for v in ovars}
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
            profiles=profiles,
            **data,
        )
    )
    return True


def _get_Timeseries(
    coords, fields, dims, states_dict, ovars, fixval, profiles, verbosity
):
    """Try to generate time series parameters
    :group: input.yaml.windio
    """
    if len(coords) == 1 and FC.TIME in coords:
        if verbosity > 2:
            print("        selecting class 'Timeseries'")

        data = {}
        fix = {
            v: fixval.get(v, default_values[v]) 
            for v in ovars if v not in fields
        }
        for v, d in fields.items():
            if dims[v] == (FC.TIME,):
                data[v] = d
            elif len(dims[v]) == 0:
                fix[v] = d
            elif verbosity > 2:
                print(f"        ignoring field '{v}' with dims {dims[v]}")

        sdata = pd.DataFrame(index=coords[FC.TIME], data=data)
        sdata.index.name = FC.TIME
        states_dict.update(
            dict(
                states_type="Timeseries",
                data_source=sdata,
                output_vars=ovars,
                fixed_vars=fix,
                profiles=profiles,
            )
        )
        return True
    return False


def _get_MultiHeightNCTimeseries(
    coords, fields, dims, states_dict, ovars, fixval, profiles, verbosity
):
    """Try to generate time series parameters
    :group: input.yaml.windio
    """
    if len(coords) == 2 and FC.TIME in coords and FV.H in coords:
        if verbosity > 2:
            print("        selecting class 'MultiHeightNCTimeseries'")

        if len(profiles) and verbosity > 0:
            print(
                f"Ignoring profile '{profiles[FV.WS]}' for states class 'MultiHeightNCTimeseries'"
            )

        data = {}
        fix = {
            v: fixval.get(v, default_values[v]) 
            for v in ovars if v not in fields
        }
        for v, d in fields.items():
            if dims[v] == (FC.TIME, FV.H):
                data[v] = ((FC.TIME, FV.H), d)
            elif dims[v] == (FV.H, FC.TIME):
                data[v] = ((FC.TIME, FV.H), np.swapaxes(d, 0, 1))
            elif len(dims[v]) == 0:
                fix[v] = d
            elif verbosity > 2:
                print(f"        ignoring field '{v}' with dims {dims[v]}")

        sdata = Dataset(coords=coords, data_vars=data)
        states_dict.update(
            dict(
                states_type="MultiHeightNCTimeseries",
                h_coord=FV.H,
                format_times_func=None,
                data_source=sdata,
                output_vars=ovars,
                fixed_vars=fix,
                bounds_error=False,
            )
        )
        return True
    return False


def _get_WeibullSectors(
    coords, fields, dims, states_dict, ovars, fixval, profiles, verbosity
):
    """Try to generate Weibull sector parameters
    :group: input.yaml.windio
    """
    if (
        FV.WEIBULL_A in fields
        and FV.WEIBULL_k in fields
        and "sector_probability" in fields
        and len(dims[FV.WEIBULL_A]) == 1
        and len(dims[FV.WEIBULL_k]) == 1
        and len(dims["sector_probability"]) == 1
    ):
        if verbosity > 2:
            print("        selecting class 'WeibullSectors'")

        data = {}
        fix = {
            v: fixval.get(v, default_values[v]) 
            for v in ovars if v not in fields
        }
        c = dims[FV.WEIBULL_A][0]
        for v, d in fields.items():
            if dims[v] == (c,):
                data[v] = d
            elif len(dims[v]) == 0:
                fix[v] = d
            elif verbosity > 2:
                print(f"        ignoring field '{v}' with dims {dims[v]}")

        if FV.WD in coords:
            data[FV.WD] = coords[FV.WD]

        sdata = pd.DataFrame(index=range(len(fields[FV.WEIBULL_A])), data=data)
        sdata.index.name = "sector"
        states_dict.update(
            dict(
                states_type="WeibullSectors",
                data_source=sdata,
                ws_bins=np.arange(60) / 2 if FV.WS not in sdata else None,
                output_vars=ovars,
                var2ncvar={FV.WEIGHT: "sector_probability"},
                fixed_vars=fix,
                profiles=profiles,
            )
        )
        return True
    return False


def _get_WeibullPointCloud(
    coords, fields, dims, states_dict, ovars, fixval, profiles, verbosity
):
    """Try to generate Weibull sector parameters
    :group: input.yaml.windio
    """
    if (
        FV.WD in coords
        and FV.WEIBULL_A in fields
        and FV.WEIBULL_k in fields
        and "sector_probability" in fields
        and FV.X in fields
        and FV.Y in fields
        and len(dims[FV.X]) == 1
        and dims[FV.X] == dims[FV.Y]
        and dims[FV.X][0] != FV.WD
        and dims[FV.X][0] in coords
    ):
        if verbosity > 2:
            print("        selecting class 'WeibullPointCloud'")

        data = {}
        fix = {
            v: fixval.get(v, default_values[v]) 
            for v in ovars if v not in fields
        }
        for v, d in fields.items():
            if len(dims[v]) == 0:
                fix[v] = d
            elif v not in fixval:
                data[v] = (dims[v], d)

        sdata = Dataset(
            coords=coords,
            data_vars=data,
        )

        states_dict.update(
            dict(
                states_type="WeibullPointCloud",
                data_source=sdata,
                output_vars=ovars,
                var2ncvar={},
                fixed_vars=fix,
                point_coord=dims[FV.X][0],
                wd_coord=FV.WD,
                ws_coord=FV.WS if FV.WS in sdata.coords else None,
                ws_bins=np.arange(60) / 2 if FV.WS not in sdata else None,
                x_ncvar=FV.X,
                y_ncvar=FV.Y,
                h_ncvar=FV.H if FV.H in sdata.data_vars else None,
                weight_ncvar="sector_probability",
            )
        )
        return True
    return False


def _get_WeibullField(
    coords, fields, dims, states_dict, ovars, fixval, profiles, verbosity
):
    """Try to generate Weibull sector parameters
    :group: input.yaml.windio
    """
    if (
        FV.WD in coords
        and FV.X in coords
        and FV.Y in coords
        and FV.WEIBULL_A in fields
        and FV.WEIBULL_k in fields
        and "sector_probability" in fields
    ):
        if verbosity > 2:
            print("        selecting class 'WeibullField'")

        data = {}
        fix = {
            v: fixval.get(v, default_values[v]) 
            for v in ovars if v not in fields
        }
        for v, d in fields.items():
            if len(dims[v]) == 0:
                fix[v] = d
            elif v not in fixval:
                data[v] = (dims[v], d)

        sdata = Dataset(
            coords=coords,
            data_vars=data,
        )

        states_dict.update(
            dict(
                states_type="WeibullField",
                data_source=sdata,
                output_vars=ovars,
                wd_coord=FV.WD,
                x_coord=FV.X,
                y_coord=FV.Y,
                h_coord=FV.H if FV.H in sdata.coords else None,
                weight_ncvar="sector_probability",
                var2ncvar={},
                fixed_vars=fix,
                ws_bins=np.arange(60) / 2 if FV.WS not in sdata.coords else None,
                ws_coord=FV.WS if FV.WS in sdata.coords else None,
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

    :group: input.yaml.windio

    """
    if verbosity > 2:
        print("      Creating states")

    ovars = [FV.WS, FV.WD, FV.TI, FV.RHO]
    fixval = {}
    profiles = _get_profiles(coords, fields, dims, ovars, fixval, verbosity)

    states_dict = {}
    if (
        _get_SingleStateStates(
            coords, fields, dims, states_dict, ovars, fixval, profiles, verbosity
        )
        or _get_Timeseries(
            coords, fields, dims, states_dict, ovars, fixval, profiles, verbosity
        )
        or _get_MultiHeightNCTimeseries(
            coords, fields, dims, states_dict, ovars, fixval, profiles, verbosity
        )
        or _get_WeibullPointCloud(
            coords, fields, dims, states_dict, ovars, fixval, profiles, verbosity
        )
        or _get_WeibullSectors(
            coords, fields, dims, states_dict, ovars, fixval, profiles, verbosity
        )
        or _get_WeibullField(
            coords, fields, dims, states_dict, ovars, fixval, profiles, verbosity
        )
    ):
        return States.new(**states_dict)
    else:
        raise ValueError(
            f"Failed to create states for coords {list(coords.keys())} and fields {list(fields.keys())} with dims {dims}"
        )


def read_site(wio_dict, verbosity=1):
    """
    Reads the site information

    Parameters
    ----------
    wio_dict: foxes.utils.Dict
        The windio data
    verbosity: int
        The verbosity level, 0=silent

    Returns
    -------
    states: foxes.core.States
        The states object

    :group: input.yaml.windio

    """

    def _print(*args, level=1, **kwargs):
        if verbosity >= level:
            print(*args, **kwargs)

    wio_site = wio_dict["site"]
    _print("Reading site")
    _print("  Name:", wio_site.pop_item("name", None))
    _print("  Contents:", [k for k in wio_site.keys()])
    _print("  Ignoring boundaries", level=2)

    # read energy_resource:
    energy_resource = wio_site["energy_resource"]
    _print("  Reading energy_resource", level=2)
    _print("    Name:", energy_resource.pop_item("name", None), level=2)
    _print("    Contents:", [k for k in energy_resource.keys()], level=2)

    # read wind_resource:
    wind_resource = energy_resource["wind_resource"]
    _print("    Reading wind_resource", level=3)
    _print("      Name:", wind_resource.pop_item("name", None), level=3)
    _print("      Contents:", [k for k in wind_resource.keys()], level=3)

    # read fields
    coords = Dict(_name="coords")
    fields = Dict(_name="fields")
    dims = Dict(_name="dims")
    for n, d in wind_resource.items():
        read_wind_resource_field(n, d, coords, fields, dims, verbosity)

    # special case: operating field
    if FV.OPERATING in fields:
        wio_dict["wind_farm"][FV.OPERATING] = (dims.pop(FV.OPERATING), fields.pop(FV.OPERATING))
        if FC.TURBINE in coords:
            if not any([FC.TURBINE in dms for dms in dims.values()]):
                if verbosity > 2:
                    print(f"      Removing coordinate '{FC.TURBINE}', since only relevant for operating flag")
                coords.pop(FC.TURBINE)

    if verbosity > 2:
        print("      Coords:")
        for c, d in coords.items():
            print(f"        {c}: Shape {d.shape}")
        print("      Fields:")
        for f, d in dims.items():
            if len(d):
                print(f"        {f}: Dims {d}, shape {fields[f].shape}")
            else:
                print(f"        {f} = {fields[f]}")

    return get_states(coords, fields, dims, verbosity)
