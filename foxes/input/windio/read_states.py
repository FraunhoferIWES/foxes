import numpy as np
import pandas as pd

from foxes.input.states import Timeseries, StatesTable
from foxes.models.vertical_profiles import ABLLogNeutralWsProfile
import foxes.variables as FV
import foxes.constants as FC

def read_Timeseries(res, fixed_vars={}, **states_pars):
    """
    Reads timeseries data.
    
    Parameters
    ----------
    res: dict
        Data from the yaml file
    fixed_vars: dict
        Additional fixes variables that do
        not occur in the yaml
    states_pars: dict, optional
        Additional arguments for Timeseries

    Returns
    -------
    states: foxes.states.Timeseries
        The timeseries states
    
    """
    times = np.array([d["time"] for d in res])
    try:
        times = times.astype(np.float64)
    except ValueError:
        times = times.astype("datetime64[ns]")
    n_times = len(times)

    vars = [v for v in res[0] if v != "time"]
    n_vars = len(vars)
    data = np.zeros((n_times, n_vars), dtype=np.float64)
    for vi, v in enumerate(vars):
        for ti in range(n_times):
            data[ti, vi] = res[ti][v]
    
    kmap = {
        "direction": FV.WD,
        "speed": FV.WS,
        "z0": FV.Z0,
        "TI": FV.TI,
        "ustar": FV.USTAR,
    }

    fvars = [kmap[v] for v in vars if v in kmap]
    ovars = fvars + [v for v in fixed_vars.keys() if v not in fvars]

    sdata = pd.DataFrame(index=times, data=data, columns=fvars)
    sdata.index.name = "Time"

    if FV.TI in fvars:
        sdata[FV.TI] /= 100

    pdict = {}
    if FV.Z0 in ovars and FV.USTAR in ovars:
        pdict = {FV.WS: ABLLogNeutralWsProfile(ustar_input=True)}

    states = Timeseries(
        data_source=sdata,
        output_vars=ovars,
        fixed_vars={v: d for v, d in fixed_vars.items() if v not in fvars},
        profiles=pdict,
        **states_pars,
    )
    
    return states

def read_StatesTable(res, fixed_vars={}, **states_pars):
    """
    Reads a WindIO energy resource

    Parameters
    ----------
    res: dict
        Data from the yaml file
    fixed_vars: dict
        Additional fixes variables that do
        not occur in the yaml
    states_pars: dict, optional
        Additional arguments for the states class

    Returns
    -------
    states: foxes.states.StatesTable
        The states object

    """

    wd = np.array(res["wind_direction"], dtype=FC.DTYPE)
    ws = np.array(res["wind_speed"], dtype=FC.DTYPE)
    n_wd = len(wd)
    n_ws = len(ws)
    n = n_wd * n_ws

    data = np.zeros((n_wd, n_ws, 2), dtype=FC.DTYPE)
    data[:, :, 0] = wd[:, None]
    data[:, :, 1] = ws[None, :]
    names = ["wind_direction", "wind_speed"]
    sec_prob = None

    def _to_data(v, d, dims):
        nonlocal data, names, sec_prob
        hdata = np.zeros((n_wd, n_ws, 1), dtype=FC.DTYPE)
        if len(dims) == 0:
            hdata[:, :, 0] = FC.DTYPE(d)
        elif len(dims) == 1:
            if dims[0] == "wind_direction":
                hdata[:, :, 0] = np.array(d, dtype=FC.DTYPE)[:, None]
            elif dims[0] == "wind_speed":
                hdata[:, :, 0] = np.array(d, dtype=FC.DTYPE)[None, :]
            else:
                raise ValueError(f"Unknown dimension '{dims[0]}' for data '{v}'")
        elif len(dims) == 2:
            if dims[0] == "wind_direction" and dims[1] == "wind_speed":
                hdata[:, :, 0] = np.array(d, dtype=FC.DTYPE)
            elif dims[1] == "wind_direction" and dims[0] == "wind_speed":
                hdata[:, :, 0] = np.swapaxes(np.array(d, dtype=FC.DTYPE), 0, 1)
            else:
                raise ValueError(f"Cannot handle dims = {dims} for data '{v}'")
        else:
            raise ValueError(
                f"Can not accept more than two dimensions, got {dims} for data '{v}'"
            )
        if v == "sector_probability":
            sec_prob = hdata[:, :, 0].copy()
        else:
            data = np.append(data, hdata, axis=2)
            names.append(v)

    vmap = {
        "wind_direction": FV.WD,
        "wind_speed": FV.WS,
        "turbulence_intensity": FV.TI,
        "air_density": FV.RHO,
        "probability": FV.WEIGHT,
    }

    for v, d in res.items():
        if (v == "sector_probability" or v in vmap) and isinstance(d, dict):
            _to_data(v, d["data"], d["dims"])
    if sec_prob is not None and "probability" in names:
        data[:, :, names.index("probability")] *= sec_prob

    n_vars = len(names)
    data = data.reshape(n, n_vars)

    data = pd.DataFrame(index=range(n), data=data, columns=names)
    data.index.name = "state"
    data.rename(columns=vmap, inplace=True)

    ovars = {v: v for v in data.columns if v != FV.WEIGHT}
    ovars.update({k: v for k, v in fixed_vars.items() if k not in data.columns})

    return StatesTable(data, output_vars=ovars, fixed_vars=fixed_vars, **states_pars)
