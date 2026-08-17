from collections.abc import Sequence
from typing import cast

import numpy as np
from numpy.typing import ArrayLike

from foxes.core import Turbine, WindFarm


def add_from_arrays(
    farm: WindFarm,
    x: ArrayLike,
    y: ArrayLike,
    heights: ArrayLike | None = None,
    diameters: ArrayLike | None = None,
    ids: Sequence[int] | np.ndarray | None = None,
    names: Sequence[str] | np.ndarray | None = None,
    turbine_base_name: str = "T",
    turbine_base_name_count_shift: bool = False,
    verbosity: int = 1,
    **turbine_parameters: object,
) -> None:
    """
    Add turbines to wind farm from direct one dimensional data arrays.

    Parameters
    ----------
    farm
        The wind farm
    x
        The x-coordinates of the turbines
    y
        The y-coordinates of the turbines
    heights
        The hub heights of the turbines, or None
    diameters
        The rotor diameters of the turbines, or None
    ids
        The ids of the turbines, or None
    names
        The names of the turbines, or None
    turbine_base_name
        The turbine base name, only used
        if col_name is None
    turbine_base_name_count_shift
        Start turbine names by 1 instead of 0
    verbosity
        The verbosity level, 0 = silent
    turbine_parameters
        Additional parameters are forwarded to the WindFarm.add_turbine().

    :group: input.farm_layout

    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    height_values = None if heights is None else np.asarray(heights, dtype=float)
    diameter_values = None if diameters is None else np.asarray(diameters, dtype=float)
    id_values = None if ids is None else np.asarray(ids)
    name_values = None if names is None else np.asarray(names)

    tmodels = cast(list[str], turbine_parameters.pop("turbine_models", []))
    H = cast(float | None, turbine_parameters.pop("H", None))
    D = cast(float | None, turbine_parameters.pop("D", None))

    for i in range(len(x)):
        s = 1 if turbine_base_name_count_shift else 0
        tname = f"{turbine_base_name}{i + s}" if name_values is None else name_values[i]

        farm.add_turbine(
            Turbine(
                name=tname,
                index=id_values[i] if id_values is not None else i,
                xy=[x[i], y[i]],
                H=height_values[i] if height_values is not None else H,
                D=diameter_values[i] if diameter_values is not None else D,
                turbine_models=tmodels,
                **turbine_parameters,  # type: ignore[arg-type]
            ),
            verbosity=verbosity,
        )
