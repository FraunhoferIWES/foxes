import numpy as np

from foxes.core import Turbine

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

    inds = list(np.ndindex(*steps))
    n_turbines = len(inds)

    for i in range(n_turbines):
        xi, yi  = inds[i]
        farm.add_turbine(
            Turbine(
                xy    = xy_base + xi * step_vectors[0] + yi * step_vectors[1],
                index = None if indices is None else indices[i],
                name  = None if names is None else names[i],
                **turbine_parameters
            ),
            verbosity=verbosity
        )
        