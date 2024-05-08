import numpy as np

from foxes.utils import random_xy_square
from foxes.core import Turbine


def add_random(
    farm,
    n_turbines,
    min_dist,
    centre=[0, 0],
    seed=None,
    verbosity=1,
    **turbine_parameters,
):
    """
    Add turbines that lie randomly within a square

    Parameters
    ----------
    farm: foxes.WindFarm
        The wind farm
    n_turbines: int
        The number of turbines
    min_dist: float
        The minimal distance between turbines
    centre: array-like
        The (x, y) coordinates of the mean
    seed: int, optional
        The random seed
    verbosity: int
        The verbosity level, 0 = silent
    turbine_parameters: dict, optional
        Additional parameters are forwarded to the WindFarm.add_turbine().

    :group: input.farm_layout

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
