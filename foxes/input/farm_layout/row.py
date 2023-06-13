import numpy as np

from foxes.core import Turbine


def add_row(
    farm,
    xy_base,
    xy_step,
    n_turbines,
    indices=None,
    names=None,
    verbosity=1,
    **turbine_parameters
):
    """
    Add a single row of turbines.

    Parameters
    ----------
    farm: foxes.WindFarm
        The wind farm
    xy_base: numpy.ndarray
        The base point, shape: (2,)
    xy_step: numpy.ndarray
        The step vector, shape: (2,)
    n_turbines: int
        The number of turbines
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
    p0 = np.array(xy_base)
    delta = np.array(xy_step)

    for i in range(n_turbines):
        farm.add_turbine(
            Turbine(
                xy=p0 + i * delta,
                index=None if indices is None else indices[i],
                name=None if names is None else names[i],
                **turbine_parameters
            ),
            verbosity=verbosity,
        )
