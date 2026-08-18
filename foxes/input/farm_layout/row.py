from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from foxes.core import Turbine, WindFarm


def add_row(
    farm: WindFarm,
    xy_base: ArrayLike,
    xy_step: ArrayLike,
    n_turbines: int,
    indices: Sequence[int] | np.ndarray | None = None,
    names: Sequence[str] | np.ndarray | None = None,
    verbosity: int = 1,
    **turbine_parameters: Any,
) -> None:
    """
    Add a single row of turbines.

    Parameters
    ----------
    farm
        The wind farm
    xy_base
        The base point, shape: (2,)
    xy_step
        The step vector, shape: (2,)
    n_turbines
        The number of turbines
    indices
        The turbine indices
    names
        The turbine names
    verbosity
        The verbosity level, 0 = silent
    turbine_parameters
        Parameters forwarded to `foxes.core.Turbine`


    """
    p0 = np.array(xy_base)
    delta = np.array(xy_step)

    for i in range(n_turbines):
        farm.add_turbine(
            Turbine(
                xy=p0 + i * delta,
                index=None if indices is None else indices[i],
                name=None if names is None else names[i],
                **turbine_parameters,
            ),
            verbosity=verbosity,
        )
