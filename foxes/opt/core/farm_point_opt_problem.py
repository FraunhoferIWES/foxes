import numpy as np

from .farm_opt_problem import FarmOptProblem
import foxes.constants as FC

class FarmPointOptProblem(FarmOptProblem):
    """
    Base class for farm problems that evaluate
    data at probe points

    Parameters
    ----------
    name : str
        The problem's name
    algo : foxes.core.Algorithm
        The algorithm
    points : numpy.ndarray
        The probe points, shape: (n_states, n_points, 3)
    
    Attributes
    ----------
    points : numpy.ndarray
        The probe points, shape: (n_states, n_points, 3)

    """

    def __init__(self, name, algo, points, *args, **kwargs):
        super().__init__(name, algo, *args, **kwargs)
        self.points = points

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
        farm_results = super().apply_individual(vars_int, vars_float)
        point_results = self.problem.algo.calc_points(farm_results, self.points)
        return farm_results, point_results

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
        farm_results = super().apply_population(vars_int, vars_float)

        n_pop = farm_results["n_pop"]
        n_states, n_points = self.points.shape[:2]
        pop_points = np.zeros((n_pop, n_states, n_points, 3), dtype=FC.DTYPE)
        pop_points[:] = self.points[None, :, : , :]
        pop_points = pop_points.reshape(n_pop*n_states, n_points, 3)

        point_results = self.problem.algo.calc_points(farm_results, pop_points)

        return farm_results, point_results
