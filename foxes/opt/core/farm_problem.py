import numpy as np
from abc import abstractmethod
from iwopy import Problem

from foxes.models.turbine_models import SetFarmVars
import foxes.constants as FC

class FarmProblem(Problem):
    """
    Abstract base class of wind farm optimization problems.

    Parameters
    ----------
    algo : foxes.core.Algorithm
        The algorithm
    runner : foxes.core.Runner
        The runner for running the algorithm
    sel_turbines : list of int, optional
        The turbines selected for optimization,
        or None for all
    pre_rotor : bool
        Flag for running before rotor model
    calc_farm_args : dict
        Additional parameters for algo.calc_farm()
    kwargs : dict, optional
        Additional parameters for `iwopy.Problem`

    Attributes
    ----------
    algo : foxes.core.Algorithm
        The algorithm
    runner : foxes.core.Runner
        The runner for running the algorithm
    sel_turbines : list of int
        The turbines selected for optimization
    pre_rotor : bool
        Flag for running before rotor model
    calc_farm_args : dict
        Additional parameters for algo.calc_farm()

    """

    def __init__(
            self, 
            name, 
            algo, 
            runner, 
            sel_turbines=None, 
            pre_rotor=False, 
            calc_farm_args={},
            **kwargs,
        ):
        super().__init__(name, **kwargs)

        self.algo = algo
        self.runner = runner
        self.pre_rotor = pre_rotor
        self.calc_farm_args = calc_farm_args
        self.sel_turbines = (
            sel_turbines if sel_turbines is not None else list(range(algo.n_turbines))
        )

    def initialize(self, verbosity=0):
        """
        Initialize the object.

        Parameters
        ----------
        verbosity : int
            The verbosity level, 0 = silent

        """
        found = False
        for t in self.algo.farm.turbines:
            if self.name in t.models:
                found = True
                break
        if not found:
            raise ValueError(f"FarmProblem '{self.name}': Missing entry '{self.name}' among any of the turbine models")

        if self.name in self.algo.mbook.turbine_models:
            raise KeyError(f"FarmProblem '{self.name}': Turbine model entry '{self.name}' already exists in model book")

        super().initialize(verbosity)

    @abstractmethod
    def opt2farm_vars_individual(self, vars_int, vars_float):
        """
        Translates optimization variables to farm variables

        Parameters
        ----------
        vars_int : numpy.ndarray
            The integer optimization variable values,
            shape: (n_vars_int,)
        vars_float : numpy.ndarray
            The float optimization variable values,
            shape: (n_vars_float,)

        Returns
        -------
        farm_vars : dict
            The foxes farm variables. Key: var name,
            value: numpy.ndarray with values, shape: 
            (n_states, n_sel_turbines)

        """
        pass

    @abstractmethod
    def opt2farm_vars_population(self, vars_int, vars_float):
        """
        Translates optimization variables to farm variables

        Parameters
        ----------
        vars_int : numpy.ndarray
            The integer optimization variable values,
            shape: (n_pop, n_vars_int)
        vars_float : numpy.ndarray
            The float optimization variable values,
            shape: (n_pop, n_vars_float)

        Returns
        -------
        farm_vars : dict
            The foxes farm variables. Key: var name,
            value: numpy.ndarray with values, shape: 
            (n_pop, n_states, n_sel_turbines)

        """
        pass

    @property
    def all_turbines(self):
        """
        Flag for all turbines optimization

        Returns
        -------
        bool :
            True if all turbines are subject to optimization

        """
        return len(self.sel_turbines) == self.algo.n_turbines

    def apply_individual(self, vars_int, vars_float):
        """
        Apply new variables to the problem.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float : np.array
            The float variable values, shape: (n_vars_float,)

        Returns
        -------
        problem_results : Any
            The results of the variable application
            to the problem

        """
        
        # create/overwrite turbine model that sets variables to opt values:
        self.algo.mbook.turbine_models[self.name] = SetFarmVars(pre_rotor=self.pre_rotor)
        model = self.algo.mbook.turbine_models[self.name]
        for v, vals in self.opt2farm_vars_individual(vars_int, vars_float).items:
            if self.all_turbines:
                model.add_var(v, vals)
            else:
                data = np.zeros((self.algo.n_states, self.algo.n_turbines), dtype=FC.DTYPE)
                data[:, self.sel_turbines] = vals
                model.add_var(v, data)
        
        # run the farm calculation:
        return self.runner.run(self.algo.calc_farm, kwargs=self.calc_farm_args)

    def apply_population(self, vars_int, vars_float):
        """
        Apply new variables to the problem,
        for a whole population.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float : np.array
            The float variable values, shape: (n_pop, n_vars_float)

        Returns
        -------
        problem_results : Any
            The results of the variable application
            to the problem

        """
        return None

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
