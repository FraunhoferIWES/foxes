from iwopy import Constraint

from foxes.utils import all_subclasses


class FarmConstraint(Constraint):
    """
    Abstract base class for foxes wind farm
    optimization constraints.

    :group: opt.core

    """

    def __init__(self, problem, name, sel_turbines=None, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        problem: foxes.opt.FarmOptProblem
            The underlying optimization problem
        name: str
            The name of the constraint
        sel_turbines: list of int, optional
            The selected turbines
        kwargs: dict, optional
            Additional parameters for `iwopy.Constraint`

        """
        super().__init__(problem, name, **kwargs)
        self._sel_turbines = sel_turbines

    @property
    def farm(self):
        """
        The wind farm

        Returns
        -------
        foxes.core.WindFarm :
            The wind farm

        """
        return self.problem.farm

    @property
    def sel_turbines(self):
        """
        The list of selected turbines

        Returns
        -------
        list of int :
            The list of selected turbines

        """
        return (
            self.problem.sel_turbines
            if self._sel_turbines is None
            else self._sel_turbines
        )

    @property
    def n_sel_turbines(self):
        """
        The numer of selected turbines

        Returns
        -------
        int :
            The numer of selected turbines

        """
        return len(self.sel_turbines)

    def add_to_layout_figure(self, ax, **kwargs):
        """
        Add to a layout figure

        Parameters
        ----------
        ax: matplotlib.pyplot.Axis
            The figure axis

        """
        return ax

    @classmethod
    def print_models(cls):
        """
        Prints all model names.
        """
        names = sorted([scls.__name__ for scls in all_subclasses(cls)])
        for n in names:
            print(n)
