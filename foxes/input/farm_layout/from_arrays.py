from foxes.core import Turbine


def add_from_arrays(
    farm,
    x,
    y,
    heights=None,
    diameters=None,
    ids=None,
    names=None,
    turbine_base_name="T",
    turbine_base_name_count_shift=False,
    verbosity=1,
    **turbine_parameters,
):
    """
    Add turbines to wind farm from direct one dimensional data arrays.

    Parameters
    ----------
    farm: foxes.WindFarm
        The wind farm
    x: list or numpy.ndarray
        The x-coordinates of the turbines
    y: list or numpy.ndarray
        The y-coordinates of the turbines
    heights: list or numpy.ndarray, optional
        The hub heights of the turbines, or None
    diameters: list or numpy.ndarray, optional
        The rotor diameters of the turbines, or None
    ids: list or numpy.ndarray, optional
        The ids of the turbines, or None
    names: list or numpy.ndarray, optional
        The names of the turbines, or None
    turbine_base_name: str, optional
        The turbine base name, only used
        if col_name is None
    turbine_base_name_count_shift: bool, optional
        Start turbine names by 1 instead of 0
    verbosity: int
        The verbosity level, 0 = silent
    turbine_parameters: dict, optional
        Additional parameters are forwarded to the WindFarm.add_turbine().

    :group: input.farm_layout

    """
    tmodels = turbine_parameters.pop("turbine_models", [])
    H = turbine_parameters.pop("H", None)
    D = turbine_parameters.pop("D", None)

    for i in range(len(x)):
        s = 1 if turbine_base_name_count_shift else 0
        tname = (
            f"{turbine_base_name}{i + s}" if names is None else names[i]
        )

        farm.add_turbine(
            Turbine(
                name=tname,
                index=ids[i] if ids is not None else i,
                xy=[x[i], y[i]],
                H=heights[i] if heights is not None else H,
                D=diameters[i] if diameters is not None else D,
                turbine_models=tmodels,
                **turbine_parameters,
            ),
            verbosity=verbosity,
        )
