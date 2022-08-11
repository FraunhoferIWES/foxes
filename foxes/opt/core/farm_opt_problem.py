from iwopy import Problem


class FarmOptProblem(Problem):
    """
    Abstract base class of wind farm optimization problems.

    Parameters
    ----------
    name : str
        The name of the problem
    farm : foxes.WindFarm
        The wind farm
    sel_turbines : list of int, optional
        The turbines selected for optimization,
        or None for all
    kwargs : dict, optional
        Additional parameters for `iwopy.Problem`

    Attributes
    ----------
    farm : foxes.WindFarm
        The wind farm
    sel_turbines : list of int
        The turbines selected for optimization

    """

    def __init__(self, name, farm, sel_turbines=None, **kwargs):
        super().__init__(name, **kwargs)

        self.farm = farm
        self.sel_turbines = (
            sel_turbines if sel_turbines is not None else list(range(farm.n_turbines))
        )
