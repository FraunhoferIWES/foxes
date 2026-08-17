import numpy as np

from foxes.core import Turbine
from foxes.utils import wd2wdvec


def add_ring(
    farm,
    xy_base,
    dist,
    n_turbines,
    offset_deg=0,
    indices=None,
    names=None,
    verbosity=1,
    **turbine_parameters,
):
    """
    Add a ring of turbines.

    Parameters
    ----------
    farm
        The wind farm
    xy_base
        The base point, shape: (2,)
    dist
        The distance between turbines
    n_turbines
        The number of turbines
    offset_deg
        The offset from north in degrees,
        following wind direction conventions
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
    R = n_turbines * dist / (2 * np.pi)
    a = np.atleast_1d(offset_deg)
    da = 360 / n_turbines

    for i in range(n_turbines):
        n = wd2wdvec(a)[0]

        farm.add_turbine(
            Turbine(
                xy=p0 + R * n,
                index=None if indices is None else indices[i],
                name=None if names is None else names[i],
                **turbine_parameters,
            ),
            verbosity=verbosity,
        )

        a[0] += da
