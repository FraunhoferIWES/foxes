import pandas as pd
import numpy as np

import foxes.variables as FV
import foxes.constants as FC


def create_random_abl_states(
    n_states, cols_minmax, var2col={}, mol_abs_range=(50.0, 5000.0), normalize=True
):
    """
    Create random abl states.

    Parameters
    ----------
    n_states: int
        The number of states
    cols_minmax: dict
        For each variable the min and max values,
        keys: variable name str, values: array_like
        with length 2
    var2col: dict, optional
        Mapping from variables to column names
    mol_abs_range: tuple
        Min and max of allowed MOL values, set to
        nan if exceeded (i.e., neutral stratification)
    normalize: bool
        Normalize weights to 1

    Returns
    -------
    data: pandas.DataFrame
        The created states data

    :group: input.states.create

    """

    data = pd.DataFrame(index=range(n_states))
    data.index.name = FC.STATE

    for v, mm in cols_minmax.items():
        data[v] = np.random.uniform(low=mm[0], high=mm[1], size=(n_states,)).astype(
            FC.DTYPE
        )

    cmol = var2col.get(FV.MOL, FV.MOL)
    if cmol in data and mol_abs_range is not None:
        sel = (data[cmol].abs() < mol_abs_range[0]) | (
            data[cmol].abs() > mol_abs_range[1]
        )
        data.loc[sel, cmol] = np.nan

    wcol = var2col.get(FV.WEIGHT, FV.WEIGHT)
    if wcol in data and normalize:
        data[wcol] /= data[wcol].sum()

    return data


def write_random_abl_states(
    file_path,
    n_states,
    cols_minmax,
    var2col={},
    mol_abs_range=(50.0, 5000.0),
    normalize=True,
    verbosity=1,
    digits="auto",
    **kwargs
):
    """
    Writes random abl states to file

    Parameters
    ----------
    file_path: str
        Path to the file
    n_states: int
        The number of states
    cols_minmax: dict
        For each variable the min and max values,
        keys: variable name str, values: array_like
        with length 2
    var2col: dict, optional
        Mapping from variables to column names
    mol_abs_range: tuple
        Min and max of allowed MOL values, set to
        nan if exceeded (i.e., neutral stratification)
    normalize: bool
        Normalize weights to 1
    verbosity: int
        The verbosity level, 0 = silent
    digits: int or auto
        The number of digits to be written
    kwargs: dict, optional
        Parameters for `pandas.DataFrame.to_csv`

    """

    if verbosity:
        print("Writing file", file_path)

    data = create_random_abl_states(
        n_states, cols_minmax, var2col, mol_abs_range, normalize
    )

    if digits is not None:
        hdigits = {c: 4 for c in cols_minmax.keys()}

        wcol = var2col.get(FV.WEIGHT, FV.WEIGHT)
        if wcol in cols_minmax:
            hdigits[wcol] = None

        mcol = var2col.get(FV.MOL, FV.MOL)
        if mcol in cols_minmax:
            hdigits[mcol] = 1

        tcol = var2col.get(FV.TI, FV.TI)
        if tcol in cols_minmax:
            hdigits[tcol] = 6

        if isinstance(digits, str) and digits == "auto":
            digits = hdigits
        else:
            digits = hdigits

    if digits is not None:
        for v, d in digits.items():
            if d is not None:
                data[v] = data[v].round(d)

    data.to_csv(file_path, **kwargs)
