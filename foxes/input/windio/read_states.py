import numpy as np
import pandas as pd

from foxes.input.states import Timeseries, StatesTable, MultiHeightTimeseries
from foxes.models.vertical_profiles import ABLLogNeutralWsProfile
import foxes.variables as FV
import foxes.constants as FC

def read_Timeseries(res, fixed_vars={}, ignore_vars=[], var_map={}, 
                    rescale={}, fillna=False, **states_pars):
    """
    Reads timeseries data.
    
    Parameters
    ----------
    res: dict
        Data from the yaml file
    fixed_vars: dict
        Additional fixes variables that do
        not occur in the yaml
    ignore_vars: list of str
        windio variables to be ignored
    var_map: dict
        Map from data column names to foxes names
    rescale: dict
        Rescale foxes variables by these factors
    fillna: bool
        Fill NaN values by neighbours
    states_pars: dict, optional
        Additional arguments for Timeseries

    Returns
    -------
    states: foxes.states.Timeseries
        The timeseries states
    
    """
    times = np.array([d["time"] for d in res])
    try:
        times = times.astype(FC.DTYPE)
    except ValueError:
        times = times.astype("datetime64[ns]")
    n_times = len(times)

    kmap = {
        "direction": FV.WD,
        "speed": FV.WS,
        "TI": FV.TI,
        "z0": FV.Z0,
        "ustar": FV.USTAR,
    }
    kmap.update(var_map)
    kmapi = {f: w for w, f in kmap.items()}

    heights = np.array(res[0]["z"], dtype=FC.DTYPE) if "z" in res[0] else None
    if heights is not None:
        ignore_vars += ["z0", "ustar"]

    ign = ["time", "z"] + ignore_vars
    vars = [v for v in res[0] if v not in ign]
    fvars = [kmap[v] for v in vars if v in kmap]
    ovars = fvars + [v for v in fixed_vars.keys() if v not in fvars]
    n_vars = len(fvars)

    # spatially uniform case:
    if heights is None:
        data = np.zeros((n_times, n_vars), dtype=FC.DTYPE)
        for ti in range(n_times):
            for vi, v in enumerate(fvars):
                w = kmapi[v]
                f = rescale.get(v, 1.)
                data[ti, vi] = f*np.array(res[ti][w])

        sdata = pd.DataFrame(index=times, data=data, columns=fvars)
        sdata.index.name = "Time"
        #print(sdata)
        #print(sdata.describe())

        pdict = {}
        if FV.Z0 in ovars:
            if FV.USTAR in ovars:
                pdict = {FV.WS: ABLLogNeutralWsProfile(ustar_input=True)}
            else:
                pdict = {FV.WS: ABLLogNeutralWsProfile(ustar_input=False)}

        if fillna:
            sdata = sdata.ffill().bfill()
            
        states = Timeseries(
            data_source=sdata,
            output_vars=ovars,
            fixed_vars={v: d for v, d in fixed_vars.items() if v not in fvars},
            profiles=pdict,
            **states_pars,
        )
    
    # multi-height case:
    else:
        n_heights = len(heights)
        data = np.zeros((n_times, n_vars, n_heights), dtype=FC.DTYPE)
        for ti in range(n_times):
            if ti > 0 and np.any(res[ti]["z"] != heights):
                raise ValueError(f"Height mismatch between time 0 and time {times[ti]}")
            for vi, v in enumerate(fvars):
                w = kmapi[v]
                f = rescale.get(v, 1.)
                data[ti, vi] = f*np.array(res[ti][w])                      

        hmap = {h: f"h{hi:04d}" for hi, h in enumerate(heights)}
        cfun = lambda v, h: f"{v}-{hmap[h]}"
        ddict = {cfun(v, h): data[:, vi, hi] for vi, v in enumerate(fvars) for hi, h in enumerate(heights)}
        sdata = pd.DataFrame(index=times, data=ddict)

        if fillna:
            sdata = sdata.ffill().bfill()

        states = MultiHeightTimeseries(
            data_source=sdata,
            output_vars=ovars,
            fixed_vars={v: d for v, d in fixed_vars.items() if v not in fvars},
            heights=heights,
            height2col=hmap,
            **states_pars,
        )
    
    return states

def read_StatesTable(res, fixed_vars={}, ignore_vars=[], **states_pars):
    """
    Reads a WindIO energy resource

    Parameters
    ----------
    res: dict
        Data from the yaml file
    fixed_vars: dict
        Additional fixes variables that do
        not occur in the yaml
    ignore_vars: list of str
        windio variables to be ignored
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
        if v not in ignore_vars and (v == "sector_probability" or v in vmap) and isinstance(d, dict):
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
