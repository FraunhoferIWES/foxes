from abc import abstractmethod
from iwopy import Constraint

from foxes.utils import all_subclasses


class FarmConstraint(Constraint):
    """
    Abstract base class for foxes wind farm
    optimization constraints.

    Parameters
    ----------
    problem : foxes.opt.FarmOptProblem
        The underlying optimization problem
    name : str
        The name of the constraint
    sel_turbines : list of int, optional
        The selected turbines
    kwargs : dict, optional
        Additional parameters for `iwopy.Constraint`

    Attributes
    ----------
    farm: foxes.WindFarm
        The wind farm
    sel_turbines: list of int
        The selected turbines

    """

    def __init__(self, problem, name, sel_turbines=None, **kwargs):
        super().__init__(problem, name, **kwargs)
        self.farm = problem.farm
        self.sel_turbines = (
            problem.sel_turbines if sel_turbines is None else sel_turbines
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
        ax : matplotlib.pyplot.Axis
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
