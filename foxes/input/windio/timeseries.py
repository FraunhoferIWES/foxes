import numpy as np
import pandas as pd

from foxes.input.states import Timeseries
from foxes.models.vertical_profiles import ABLLogNeutralWsProfile
import foxes.variables as FV

def read_timeseries(res, fixed_vars={}, **kwargs):
    """
    Reads timeseries data.
    
    Parameters
    ----------
    res: dict
        Data from the yaml file
    fixed_vars: dict
        Additional fixes variables that do
        not occur in the yaml
    kwargs: dict, optional
        Additional arguments for Timeseries

    Returns
    -------
    states: foxes.states.Timeseries
        The timeseries states
    
    """
    times = [d["time"] for d in res]
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

    pdict = {}
    if FV.Z0 in ovars and FV.USTAR in ovars:
        pdict = {FV.WS: ABLLogNeutralWsProfile(ustar_input=True)}

    states = Timeseries(
        data_source=sdata,
        output_vars=ovars,
        fixed_vars={v: d for v, d in fixed_vars.items() if v not in fvars},
        profiles=pdict,
    )
    
    return states
