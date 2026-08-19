import pandas as pd
from collections.abc import Sequence
from typing import Any

from foxes.core import Turbine, WindFarm
from foxes.config import get_input_path


def add_from_csv(
    farm: WindFarm,
    data_source: str | pd.DataFrame,
    col_index: str | None = None,
    col_name: str | None = None,
    col_x: str = "x",
    col_y: str = "y",
    col_H: str | None = None,
    col_D: str | None = None,
    col_id: str | None = None,
    cols_models_pre: Sequence[str] | None = None,
    col_turbine_type: str | None = None,
    cols_models_post: Sequence[str] | None = None,
    col_wind_farm: str | None = None,
    col_cluster: str | None = None,
    turbine_base_name: str = "T",
    turbine_ids: Sequence[int] | None = None,
    turbine_base_name_count_shift: bool = False,
    verbosity: int = 1,
    **turbine_parameters: Any,
) -> None:
    """
    Add turbines to wind farm via csv input file.

    Parameters
    ----------
    farm
        The wind farm
    data_source
        The input csv file or data source
    col_index
        The index column, or None
    col_name
        The name column, or None
    col_x
        The x column
    col_y
        The y column
    col_H
        The hub height column
    col_D
        The rotor diameter column
    col_id
        The id column
    cols_models_pre
        The turbine model columns, entered before
        turbine_models
    col_turbine_type
        The turbine type name
    cols_models_post
        The turbine model columns, entered after
        turbine_models
    col_wind_farm
        The wind farm name column
    col_cluster
        The cluster name column
    turbine_base_name
        The turbine base name, only used
        if col_name is None
    turbine_ids
        The turbine ids, or None for
        index
    turbine_base_name_count_shift
        Start turbine names by 1 instead of 0
    verbosity
        The verbosity level, 0 = silent
    turbine_parameters
        Additional parameters are forwarded to the WindFarm.add_turbine().


    """

    if isinstance(data_source, pd.DataFrame):
        data = data_source
    else:
        if verbosity:
            print("Reading file", data_source)
        pth = get_input_path(data_source)
        data = pd.read_csv(pth, index_col=col_index)

    tmodels = turbine_parameters.pop("turbine_models", [])
    H = turbine_parameters.pop("H", None)
    D = turbine_parameters.pop("D", None)

    for i in data.index:
        s = 1 if turbine_base_name_count_shift else 0
        tname = (
            f"{turbine_base_name}{i + s}" if col_name is None else data.loc[i, col_name]
        )
        txy = data.loc[i, [col_x, col_y]].values

        if turbine_ids is not None:
            tid = turbine_ids[i]
        elif col_id is not None:
            tid = data.loc[i, col_id]
        else:
            tid = None

        hmodels = (
            [] if cols_models_pre is None else data.loc[i, cols_models_pre].tolist()
        )
        hmodels += [] if col_turbine_type is None else [data.loc[i, col_turbine_type]]
        hmodels += tmodels
        hmodels += (
            [] if cols_models_post is None else data.loc[i, cols_models_post].tolist()
        )

        farm.add_turbine(
            Turbine(
                name=tname,
                index=tid,
                xy=txy,
                H=H if col_H not in data.columns else data.loc[i, col_H],
                D=D if col_D not in data.columns else data.loc[i, col_D],
                turbine_models=hmodels,
                wind_farm_name=None
                if col_wind_farm is None
                else data.loc[i, col_wind_farm],
                cluster_name=None if col_cluster is None else data.loc[i, col_cluster],
                **turbine_parameters,
            ),
            verbosity=verbosity,
        )
