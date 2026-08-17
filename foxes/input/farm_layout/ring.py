from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from foxes.core import Turbine, WindFarm
from foxes.utils import wd2wdvec


def add_ring(
    farm: WindFarm,
    xy_base: ArrayLike,
    dist: float,
    n_turbines: int,
    offset_deg: float = 0,
    indices: Sequence[int] | np.ndarray | None = None,
    names: Sequence[str] | np.ndarray | None = None,
    verbosity: int = 1,
    **turbine_parameters: Any,
) -> None:
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
