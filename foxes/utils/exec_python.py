import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


def exec_python(s, indicator="%", newline=";", globals=globals(), locals={}):
    """
    Executes strings that start with the
    indicator as python commands, returns one value

    Example:
    s = "%p%N=10;p=np.zeros((N,3));p[:,0]=50;p[:,1]=np.linspace(0,7000,N);p[:,2]=119"

    Parameters
    ----------
    s: list, dict or object
        The source to by analyzed
    indicator: str
        The indicator that trigger python evaluation
    newline: str
        The new line indicator
    globals: dict
        The global namespace
    locals: dict
        The local namespace

    Returns
    -------
    out: list, dict or object
        The same structure, but all python
        strings evaluated

    :group: utils

    """
    if isinstance(s, str):
        L = len(indicator)
        if len(s) > L and s[:L] == indicator:
            a = s[L:]
            if not indicator in a:
                exec(a, globals, locals)
            else:
                ilist = a.split(indicator)
                if len(ilist) != 2:
                    raise ValueError(
                        f"Expecting at most 2 occurences of '{indicator}', found {len(ilist)}: {s}"
                    )
                v, b = ilist
                for c in b.split(newline):
                    exec(c, globals, locals)
                return locals[v]
    elif isinstance(s, list):
        return [exec_python(a, indicator) for a in s]
    elif isinstance(s, tuple):
        return tuple(exec_python(list(s), indicator))
    elif isinstance(s, dict):
        return {k: exec_python(a, indicator) for k, a in s.items()}
    return s
