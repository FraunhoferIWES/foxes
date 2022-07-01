import pandas as pd

from foxes.core import Turbine


def add_from_csv(
    farm,
    file_path,
    col_index=None,
    col_name=None,
    col_x="x",
    col_y="y",
    col_H=None,
    col_D=None,
    col_id=None,
    cols_models=None,
    turbine_base_name="T",
    turbine_ids=None,
    turbine_base_name_count_shift=False,
    verbosity=1,
    **turbine_parameters,
):
    """
    Add turbines to wind farm via csv input file.

    Additional turbine_parameters are forwarded to the WindFarm.add_turbine().

    Parameters
    ----------
    farm : foxes.WindFarm
        The wind farm
    file_path : str
        The input csv file
    col_index : str, optional
        The index column, or None
    col_name : str, optional
        The name column, or None
    col_x : str, optional
        The x column
    col_y : str, optional
        The y column
    col_H: str, optional
        The hub height column
    col_D : str, optional
        The rotor diameter column
    col_id : str, optional
        The id column
    cols_models : list of str, optional
        The turbine model columns
    turbine_base_name : str, optional
        The turbine base name, only used
        if col_name is None
    turbine_ids : list, optional
        The turbine ids, or None for
        index
    turbine_base_name_count_shift : bool, optional
        Start turbine names by 1 instead of 0
    verbosity : int
        The verbosity level, 0 = silent

    """

    if verbosity:
        print("Reading file", file_path)
    data = pd.read_csv(file_path, index_col=col_index)

    tmodels = turbine_parameters.pop("turbine_models", [])
    H = turbine_parameters.pop("H", None)
    D = turbine_parameters.pop("D", None)

    for i in data.index:

        s = 1 if turbine_base_name_count_shift else 0
        tname = (
            f"{turbine_base_name}{i+s}" if col_name is None else data.loc[i, col_name]
        )
        txy = data.loc[i, [col_x, col_y]].values

        if turbine_ids is not None:
            tid = turbine_ids[i]
        elif col_id is not None:
            tid = data.loc[i, col_id]
        else:
            tid = None

        hmodels = [] if cols_models is None else data.loc[i, cols_models].tolist()
        hmodels += tmodels

        farm.add_turbine(
            Turbine(
                name=tname,
                index=tid,
                xy=txy,
                H=H if col_H not in data.columns else data.loc[i, col_H],
                D=D if col_D not in data.columns else data.loc[i, col_D],
                turbine_models=tmodels,
                **turbine_parameters,
            ),
            verbosity=verbosity,
        )
