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
    farm: foxes.core.WindFarm
        The wind farm
    x: numpy.typing.ArrayLike
        The x-coordinates of the turbines
    y: numpy.typing.ArrayLike
        The y-coordinates of the turbines
    heights: numpy.typing.ArrayLike, optional
        The hub heights of the turbines, or None
    diameters: numpy.typing.ArrayLike, optional
        The rotor diameters of the turbines, or None
    ids: collections.abc.Sequence[int] or numpy.ndarray, optional
        The ids of the turbines, or None
    names: collections.abc.Sequence[str] or numpy.ndarray, optional
        The names of the turbines, or None
    turbine_base_name: str, optional
        The turbine base name, only used
        if col_name is None
    turbine_base_name_count_shift: bool, optional
        Start turbine names by 1 instead of 0
    verbosity: int
        The verbosity level, 0 = silent
    turbine_parameters: object, optional
        Additional parameters are forwarded to the WindFarm.add_turbine().

    :group: input.farm_layout

    """
    tmodels = cast(list[str], turbine_parameters.pop("turbine_models", []))
    H = cast(float | None, turbine_parameters.pop("H", None))
    D = cast(float | None, turbine_parameters.pop("D", None))

    for i in range(len(x)):
        s = 1 if turbine_base_name_count_shift else 0
        tname = f"{turbine_base_name}{i + s}" if names is None else names[i]

        farm.add_turbine(
            Turbine(
                name=tname,
                index=ids[i] if ids is not None else i,
                xy=[x[i], y[i]],
                H=heights[i] if heights is not None else H,
                D=diameters[i] if diameters is not None else D,
                turbine_models=tmodels,
                **turbine_parameters,  # type: ignore[arg-type]
            ),
            verbosity=verbosity,
        )
