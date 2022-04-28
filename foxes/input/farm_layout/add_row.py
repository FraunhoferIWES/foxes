from foxes.core import Turbine

def add_row(
    farm,
    xy_base,
    xy_step,
    n_turbines,
    indices=None,
    labels=None,
    verbosity=1,
    **turbine_parameters
):
    for i in range(n_turbines):
        farm.add_turbine(
            Turbine(
                xy    = xy_base + i * xy_step,
                index = None if indices is None else indices[i],
                label = None if labels is None else labels[i],
                **turbine_parameters
            ),
            verbosity=verbosity
        )