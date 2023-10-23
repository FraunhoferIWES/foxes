import numpy as np
from iwopy import Problem

from foxes.utils.runners import DefaultRunner
import foxes.constants as FC
from .pop_states import PopStates


class FarmOptProblem(Problem):
    """
    Abstract base class of wind farm optimization problems.

    Attributes
    ----------
    algo: foxes.core.Algorithm
        The algorithm
    runner: foxes.core.Runner
        The runner for running the algorithm
    calc_farm_args: dict
        Additional parameters for algo.calc_farm()
    points : numpy.ndarray
        The probe points, shape: (n_states, n_points, 3)

    :group: opt.core

    """

    def __init__(
        self,
        name,
        algo,
        runner=None,
        sel_turbines=None,
        calc_farm_args={},
        points=None,
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        name: str
            The problem's name
        algo: foxes.core.Algorithm
            The algorithm
        runner: foxes.core.Runner, optional
            The runner for running the algorithm
        sel_turbines: list of int, optional
            The turbines selected for optimization,
            or None for all
        calc_farm_args: dict
            Additional parameters for algo.calc_farm()
        points : numpy.ndarray, optional
            The probe points, shape: (n_states, n_points, 3)
        kwargs: dict, optional
            Additional parameters for `iwopy.Problem`

        """
        super().__init__(name, **kwargs)

        self.algo = algo
        self.runner = runner
        self.calc_farm_args = calc_farm_args
        self.points = points

        self._sel_turbines = sel_turbines
        self._count = None

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

    @property
    def sel_turbines(self):
        """
        The selected turbines

        Returns
        -------
        list of int :
            Indices of the selected turbines

        """
        return (
            self._sel_turbines
            if self._sel_turbines is not None
            else list(range(self.farm.n_turbines))
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

    @property
    def counter(self):
        """
        The current value of the application counter

        Returns
        -------
        int :
            The current value of the application counter

        """
        return self._count

    @classmethod
    def tvar(cls, var, turbine_i):
        """
        Gets turbine variable name

        Parameters
        ----------
        var: str
            The variable name
        turbine_i: int
            The turbine index

        Returns
        -------
        str :
            The turbine variable name

        """
        return f"{var}_{turbine_i:04d}"

    @classmethod
    def parse_tvar(cls, tvr):
        """
        Parse foxes variable name and turbine index
        from turbine variable

        Parameters
        ----------
        tvr: str
            The turbine variable name

        Returns
        -------
        var: str
            The foxes variable name
        turbine_i: int
            The turbine index

        """
        t = tvr.split("_")
        return t[0], int(t[1])

    def initialize(self, verbosity=1):
        """
        Initialize the object.

        Parameters
        ----------
        verbosity: int
            The verbosity level, 0 = silent

        """
        if self.runner is None:
            self.runner = DefaultRunner()
            self.runner.initialize()
        elif not self.runner.initialized:
            raise ValueError(f"FarmOptProblem '{self.name}': Runner not initialized.")

        if not self.algo.initialized:
            self.algo.initialize()
        self._org_states_name = self.algo.states.name
        self._org_n_states = self.algo.n_states

        self.algo.finalize()

        self._count = 0

        super().initialize(verbosity)

    def _reset_states(self, states):
        """
        Reset the states in the algorithm
        """
        if states is not self.algo.states:
            if self.algo.initialized:
                self.algo.finalize()
            self.algo.states = states

    def update_problem_individual(self, vars_int, vars_float):
        """
        Update the algo and other data using
        the latest optimization variables.

        This function is called before running the farm
        calculation.

        Parameters
        ----------
        vars_int: np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float: np.array
            The float variable values, shape: (n_vars_float,)

        """
        # reset states, if needed:
        if isinstance(self.algo.states, PopStates):
            self._reset_states(self.algo.states.states)
            self.algo.n_states = self._org_n_states

    def update_problem_population(self, vars_int, vars_float):
        """
        Update the algo and other data using
        the latest optimization variables.

        This function is called before running the farm
        calculation.

        Parameters
        ----------
        vars_int: np.array
            The integer variable values, shape: (n_pop, n_vars_int,)
        vars_float: np.array
            The float variable values, shape: (n_pop, n_vars_float,)

        """
        # set/reset pop states, if needed:
        n_pop = len(vars_float)
        if not isinstance(self.algo.states, PopStates):
            self._reset_states(PopStates(self.algo.states, n_pop))
        elif self.algo.states.n_pop != n_pop:
            ostates = self.algo.states.states
            self._reset_states(PopStates(ostates, n_pop))

    def apply_individual(self, vars_int, vars_float):
        """
        Apply new variables to the problem.

        Parameters
        ----------
        vars_int: np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float: np.array
            The float variable values, shape: (n_vars_float,)

        Returns
        -------
        problem_results: Any
            The results of the variable application
            to the problem

        """
        self._count += 1
        self.update_problem_individual(vars_int, vars_float)
        farm_results = self.runner.run(self.algo.calc_farm, kwargs=self.calc_farm_args)

        if self.points is None:
            return farm_results
        else:
            point_results = self.runner.run(
                self.algo.calc_points, args=(farm_results, self.points)
            )
            return farm_results, point_results

    def apply_population(self, vars_int, vars_float):
        """
        Apply new variables to the problem,
        for a whole population.

        Parameters
        ----------
        vars_int: np.array
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float: np.array
            The float variable values, shape: (n_pop, n_vars_float)

        Returns
        -------
        problem_results: Any
            The results of the variable application
            to the problem

        """
        self._count += 1

        self.update_problem_population(vars_int, vars_float)
        farm_results = self.runner.run(self.algo.calc_farm, kwargs=self.calc_farm_args)
        farm_results["n_pop"] = len(vars_float)
        farm_results["n_org_states"] = self._org_n_states

        if self.points is None:
            return farm_results
        else:
            n_pop = farm_results["n_pop"].values
            n_states, n_points = self.points.shape[:2]
            pop_points = np.zeros((n_pop, n_states, n_points, 3), dtype=FC.DTYPE)
            pop_points[:] = self.points[None, :, :, :]
            pop_points = pop_points.reshape(n_pop * n_states, n_points, 3)
            point_results = self.runner.run(
                self.algo.calc_points, args=(farm_results, pop_points)
            )
            return farm_results, point_results

    def add_to_layout_figure(self, ax, **kwargs):
        """
        Add to a layout figure

        Parameters
        ----------
        ax: matplotlib.pyplot.Axis
            The figure axis

        """
        for c in self.cons.functions:
            ax = c.add_to_layout_figure(ax, **kwargs)
        for f in self.objs.functions:
            ax = f.add_to_layout_figure(ax, **kwargs)

        return ax
