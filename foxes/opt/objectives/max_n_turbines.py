import numpy as np

from foxes.opt.core.farm_objective import FarmObjective
import foxes.constants as FC


class MaxNTurbines(FarmObjective):
    """
    Maximizes the number of turrbines.

    Attributes
    ----------
    check_valid: bool
        Check FC.VALID variable before counting

    :group: opt.objectives

    """

    def __init__(
        self,
        problem,
        name="max_n_turbines",
        check_valid=True,
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        problem: foxes.opt.FarmOptProblem
            The underlying optimization problem
        name: str
            The name of the objective function
        check_valid: bool
            Check FC.VALID variable before counting
        kwargs: dict, optional
            Additional parameters for `FarmObjective`

        """
        super().__init__(problem, name, **kwargs)
        self.check_valid = check_valid

    def n_components(self):
        """
        Returns the number of components of the
        function.

        Returns
        -------
        int:
            The number of components.

        """
        return 1

    def maximize(self):
        """
        Returns flag for maximization of each component.

        Returns
        -------
        flags: np.array
            Bool array for component maximization,
            shape: (n_components,)

        """
        return [True]

    def calc_individual(self, vars_int, vars_float, problem_results, components=None):
        """
        Calculate values for a single individual of the
        underlying problem.

        Parameters
        ----------
        vars_int: np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float: np.array
            The float variable values, shape: (n_vars_float,)
        problem_results: Any
            The results of the variable application
            to the problem
        components: list of int, optional
            The selected components or None for all

        Returns
        -------
        values: np.array
            The component values, shape: (n_sel_components,)

        """
        if FC.VALID in problem_results and self.check_valid:
            vld = np.sum(problem_results[FC.VALID].to_numpy(), axis=1)
            if np.min(vld) != np.max(vld):
                raise ValueError(
                    f"Objective '{self.name}': Number of valid turbines is state dependend, counting impossible"
                )
            return np.array([vld[0]], dtype=np.float64)
        else:
            return np.array([self.farm.n_turbines], dtype=np.float64)

    def calc_population(self, vars_int, vars_float, problem_results, components=None):
        """
        Calculate values for all individuals of a population.

        Parameters
        ----------
        vars_int: np.array
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float: np.array
            The float variable values, shape: (n_pop, n_vars_float)
        problem_results: Any
            The results of the variable application
            to the problem
        components: list of int, optional
            The selected components or None for all

        Returns
        -------
        values: np.array
            The component values, shape: (n_pop, n_sel_components)

        """
        n_pop = problem_results["n_pop"].to_numpy()
        if self.check_valid:
            n_states = problem_results["n_org_states"].to_numpy()
            n_turbines = self.farm.n_turbines
            vld = (
                problem_results[FC.VALID]
                .to_numpy()
                .reshape(n_pop, n_states, n_turbines)
            )
            vld = np.sum(vld, axis=2)
            if np.any(np.min(vld, axis=1) != np.max(vld, axis=1)):
                raise ValueError(
                    f"Objective '{self.name}': Number of valid turbines is state dependend, counting impossible"
                )
            return vld[:, 0, None]
        else:
            return np.full((n_pop, 1), self.farm.n_turbines, dtype=vars_float.dtype)
