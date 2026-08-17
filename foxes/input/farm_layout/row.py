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
    **turbine_parameters,
):
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
                **turbine_parameters,
            ),
            verbosity=verbosity,
        )
