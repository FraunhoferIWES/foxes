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
    for i in range(n_turbines):
        farm.add_turbine(
            Turbine(
                xy    = xy_base + i * xy_step,
                index = None if indices is None else indices[i],
                name  = None if names is None else names[i],
                **turbine_parameters
            ),
            verbosity=verbosity
        )