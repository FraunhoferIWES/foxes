import numpy as np
from abc import abstractmethod
from iwopy import Problem

from foxes.models.turbine_models import SetFarmVars
from foxes.utils.runners import DefaultRunner
import foxes.constants as FC
from .pop_states import PopStates


class FarmOptProblem(Problem):
    """
    Abstract base class of wind farm optimization problems.

    Parameters
    ----------
    algo : foxes.core.Algorithm
        The algorithm
    runner : foxes.core.Runner, optional
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
        runner=None,
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

    def tvar(self, var, turbine_i):
        """
        Gets turbine variable name

        Parameters
        ----------
        var : str
            The variable name
        turbine_i : int
            The turbine index

        Returns
        -------
        str :
            The turbine variable name

        """
        return f"{var}_{turbine_i:04d}"

    def parse_tvar(self, tvr):
        """
        Parse foxes variable name and turbine index
        from turbine variable

        Parameters
        ----------
        tvr : str
            The turbine variable name

        Returns
        -------
        var : str
            The foxes variable name
        turbine_i : int
            The turbine index

        """
        t = tvr.split("_")
        return t[0], int(t[1])

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

    @property
    def farm(self):
        """
        The wind farm

        Returns
        -------
        foxes.core.WindFarm :
            The wind farm

        """
        return self.algo.farm

    def initialize(self, verbosity=1):
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
            raise ValueError(
                f"FarmOptProblem '{self.name}': Missing entry '{self.name}' among any of the turbine models"
            )

        if self.name in self.algo.mbook.turbine_models:
            raise KeyError(
                f"FarmOptProblem '{self.name}': Turbine model entry '{self.name}' already exists in model book"
            )

        if self.runner is None:
            self.runner = DefaultRunner()
            self.runner.initialize()
        elif not self.runner.initialized:
            raise ValueError(f"FarmOptProblem '{self.name}': Runner not initialized.")

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
    def opt2farm_vars_population(self, vars_int, vars_float, n_states):
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
        n_states : int
            The number of original (non-pop) states

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

        # initialize algorithm:
        if not self.algo.initialized:
            self.algo.initialize()  # TODO: add optional parameters
        if isinstance(self.algo.states, PopStates):
            self.algo.reset_states(self.algo.states.states)

        # create/overwrite turbine model that sets variables to opt values:
        self.algo.mbook.turbine_models[self.name] = SetFarmVars(
            pre_rotor=self.pre_rotor
        )
        model = self.algo.mbook.turbine_models[self.name]
        for v, vals in self.opt2farm_vars_individual(vars_int, vars_float).items():
            if self.all_turbines:
                model.add_var(v, vals)
            else:
                data = np.zeros(
                    (self.algo.n_states, self.algo.n_turbines), dtype=FC.DTYPE
                )
                data[:, self.sel_turbines] = vals
                model.add_var(v, data)

        # run the farm calculation:
        pars = dict(verbosity=0)
        pars.update(self.calc_farm_args)
        return self.runner.run(self.algo.calc_farm, kwargs=pars)

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

        # initialize algorithm:
        n_pop = len(vars_float)
        if not isinstance(self.algo.states, PopStates):
            self.algo.reset_states(PopStates(self.algo.states, n_pop))
        elif self.algo.states.n_pop != n_pop:
            ostates = self.algo.states.states
            self.algo.reset_states(PopStates(ostates, n_pop))
            del ostates
        n_states = int(self.algo.n_states / n_pop)

        # create/overwrite turbine model that sets variables to opt values:
        self.algo.mbook.turbine_models[self.name] = SetFarmVars(
            pre_rotor=self.pre_rotor
        )
        model = self.algo.mbook.turbine_models[self.name]
        for v, vals in self.opt2farm_vars_population(
            vars_int, vars_float, n_states
        ).items():
            shp0 = list(vals.shape)
            shp1 = [self.algo.n_states] + shp0[2:]
            if self.all_turbines:
                model.add_var(v, vals.reshape(shp1))
            else:
                data = np.zeros(
                    (self.algo.n_states, self.algo.n_turbines), dtype=FC.DTYPE
                )
                data[:, self.sel_turbines] = vals.reshape(shp1)
                model.add_var(v, data)
                del data

        # run the farm calculation:
        pars = dict(verbosity=0)
        pars.update(self.calc_farm_args)
        results = self.runner.run(self.algo.calc_farm, kwargs=pars)
        results["n_pop"] = n_pop
        results["n_org_states"] = n_states

        return results

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

    def finalize_population(self, vars_int, vars_float, verbosity=0):
        """
        Finalization, given the final population data.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values of the final
            generation, shape: (n_pop, n_vars_int)
        vars_float : np.array
            The float variable values of the final
            generation, shape: (n_pop, n_vars_float)
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        problem_results : Any
            The results of the variable application
            to the problem
        objs : np.array
            The final objective function values, shape: (n_pop, n_components)
        cons : np.array
            The final constraint values, shape: (n_pop, n_constraints)

        """
        results = super().finalize_population(vars_int, vars_float, verbosity)
        self.algo.reset_states(self.algo.states.states)
        return results
