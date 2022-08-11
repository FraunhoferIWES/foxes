from iwopy import Problem


class FarmProblem(Problem):
    """
    Abstract base class of wind farm optimization problems.

    Parameters
    ----------
    name : str
        The name of the problem
    mbook : foxes.ModelBook
        The model book
    farm : foxes.WindFarm
        The wind farm
    sel_turbines : list of int, optional
        The turbines selected for optimization,
        or None for all
    kwargs : dict, optional
        Additional parameters for `iwopy.Problem`

    Attributes
    ----------
    mbook : foxes.ModelBook
        The model book
    farm : foxes.WindFarm
        The wind farm
    sel_turbines : list of int
        The turbines selected for optimization

    """

    def __init__(self, name, mbook, farm, sel_turbines=None, **kwargs):
        super().__init__(name, **kwargs)

        self.mbook = mbook
        self.farm = farm
        self.sel_turbines = (
            sel_turbines if sel_turbines is not None else list(range(farm.n_turbines))
        )

    def initialize(self, verbosity=0):
        """
        Initialize the object.

        Parameters
        ----------
        verbosity : int
            The verbosity level, 0 = silent

        """



        super().initialize(verbosity)

    def add_to_layout_figure(self, ax, **kwargs):
        """
        Add to a layout figure

        Parameters
        ----------
        ax : matplotlib.pyplot.Axis
            The figure axis
        
        """
        for c in self.cons.functions:
            ax = c.add_to_layout_figure(ax, **kwargs)
        for f in self.objs.functions:
            ax = f.add_to_layout_figure(ax, **kwargs)
        
        return ax
