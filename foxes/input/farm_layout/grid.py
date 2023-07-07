import numpy as np

from foxes.core import Turbine
import foxes.constants as FC


def add_grid(
    farm,
    xy_base,
    step_vectors,
    steps,
    indices=None,
    names=None,
    verbosity=1,
    **turbine_parameters
):
    """
    Add a regular grid of turbines.

    Parameters
    ----------
    farm: foxes.WindFarm
        The wind farm
    xy_base: numpy.ndarray
        The base point, shape: (2,)
    step_vectors: numpy.ndarray
        The two step vectors in x and y,
        respectively, shape: (2, 2)
    steps: array_like of int
        The steps in x, y. Length 2
    indices: list of int, optional
        The turbine indices
    names: list of str, optional
        The turbine names
    verbosity: int
        The verbosity level, 0 = silent
    turbine_parameters: dict, optional
        Parameters forwarded to `foxes.core.Turbine`

    :group: input.farm_layout

    """

    inds = list(np.ndindex(*steps))
    n_turbines = len(inds)

    xy_base = np.array(xy_base, dtype=FC.DTYPE)
    step_vectors = np.array(step_vectors, dtype=FC.DTYPE)

    for i in range(n_turbines):
        xi, yi = inds[i]
        farm.add_turbine(
            Turbine(
                xy=xy_base + xi * step_vectors[0] + yi * step_vectors[1],
                index=None if indices is None else indices[i],
                name=None if names is None else names[i],
                **turbine_parameters
            ),
            verbosity=verbosity,
        )
