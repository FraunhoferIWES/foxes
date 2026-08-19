import numpy as np
from numpy.typing import ArrayLike
from typing import Any

from foxes.utils import random_xy_square
from foxes.core import Turbine, WindFarm


def add_random(
    farm: WindFarm,
    n_turbines: int,
    min_dist: float,
    centre: ArrayLike = [0, 0],
    seed: int | None = None,
    verbosity: int = 1,
    **turbine_parameters: Any,
) -> None:
    """
    Add turbines that lie randomly within a square

    Parameters
    ----------
    farm
        The wind farm
    n_turbines
        The number of turbines
    min_dist
        The minimal distance between turbines
    centre
        The (x, y) coordinates of the mean
    seed
        The random seed
    verbosity
        The verbosity level, 0 = silent
    turbine_parameters
        Additional parameters are forwarded to the WindFarm.add_turbine().


    """
    xy = random_xy_square(n_turbines, min_dist, seed=seed, verbosity=verbosity)
    xy += np.array(centre)[None, :]

    for i in range(len(xy)):
        farm.add_turbine(
            Turbine(
                xy=xy[i],
                **turbine_parameters,
            ),
            verbosity=verbosity,
        )
