from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from foxes.core import Turbine, WindFarm
from foxes.config import config


def add_grid(
    farm: WindFarm,
    xy_base: ArrayLike,
    step_vectors: ArrayLike,
    steps: tuple[int, int],
    indices: Sequence[int] | np.ndarray | None = None,
    names: Sequence[str] | np.ndarray | None = None,
    verbosity: int = 1,
    **turbine_parameters: Any,
) -> None:
    """
    Add a regular grid of turbines.

    Parameters
    ----------
    farm
        The wind farm
    xy_base
        The base point, shape: (2,)
    step_vectors
        The two step vectors in x and y,
        respectively, shape: (2, 2)
    steps
        The steps in x, y. Length 2
    indices
        The turbine indices
    names
        The turbine names
    verbosity
        The verbosity level, 0 = silent
    turbine_parameters
        Parameters forwarded to `foxes.core.Turbine`

    :group: input.farm_layout

    """

    inds = list(np.ndindex(*steps))
    n_turbines = len(inds)

    xy_base = np.array(xy_base, dtype=config.dtype_double)
    step_vectors = np.array(step_vectors, dtype=config.dtype_double)

    for i in range(n_turbines):
        xi, yi = inds[i]
        farm.add_turbine(
            Turbine(
                xy=xy_base + xi * step_vectors[0] + yi * step_vectors[1],
                index=None if indices is None else indices[i],
                name=None if names is None else names[i],
                **turbine_parameters,
            ),
            verbosity=verbosity,
        )
