from abc import abstractmethod
from iwopy import Constraint

from foxes.utils import all_subclasses

class FarmConstraint(Constraint):
    """
    Abstract base class for foxes wind farm 
    optimization constraints.

    Parameters
    ----------
    problem : foxes.opt.FarmProblem
        The underlying optimization problem
    name : str
        The name of the constraint
    kwargs : dict, optional
        Additional parameters for `iwopy.Constraint`

    Attributes
    ----------
    farm: foxes.WindFarm
        The wind farm
    sel_turbines: list
        The selected turbines

    """

    def __init__(self, problem, name, **kwargs):
        super().__init__(problem, name, **kwargs)
        self.farm = problem.farm
        self.sel_turbines = problem.sel_turbines

    def set_sel_turbines(self, sel_turbines):
        """
        Set the turbine selection, in case it deviates
        from the problem's turbine selection.

        Parameters
        ----------
        sel_turbines : list of int
            The turbine indices
            
        """
        self.sel_turbines = sel_turbines

    @abstractmethod
    def required_variables(self):
        """
        Returns the foxes variables that
        are required for the calculation.

        Returns
        -------
        vnames : list of str
            The required foxes variable names

        """
        pass

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
        names = sorted([ scls.__name__ for scls in all_subclasses(cls)])
        for n in names:
            print(n)