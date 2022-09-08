import numpy as np

from foxes.opt.core import FarmOptProblem
import foxes.variables as FV
import foxes.constants as FC

class FarmLayoutOptProblem(FarmOptProblem):
    """
    The turbine positioning optimization problem

    Parameters
    ----------
    algo : foxes.core.Algorithm
        The algorithm
    runner : foxes.core.Runner, optional
        The runner for running the algorithm
    sel_turbines : list of int, optional
        The turbines selected for optimization,
        or None for all
    calc_farm_args : dict
        Additional parameters for algo.calc_farm()
    kwargs : dict, optional
        Additional parameters for `FarmOptProblem`

    """


    def __init__(
        self,
        name,
        algo,
        runner=None,
        sel_turbines=None,
        calc_farm_args={},
        **kwargs,
    ):
        super().__init__(name, algo, runner, pre_rotor=True, 
            sel_turbines=sel_turbines, calc_farm_args=calc_farm_args, 
            **kwargs)
        
    def var_names_float(self):
        """
        The names of float variables.

        Returns
        -------
        names : list of str
            The names of the float variables

        """
        vrs = []
        for ti in self.sel_turbines:
            vrs += [self.tvar(FV.X, ti), self.tvar(FV.Y, ti)]
        return vrs

    def initial_values_float(self):
        """
        The initial values of the float variables.

        Returns
        -------
        values : numpy.ndarray
            Initial float values, shape: (n_vars_float,)

        """
        out = np.zeros((self.n_sel_turbines, 2), dtype=FC.DTYPE)
        for ti in self.all_turbines:
            out[ti] = self.farm.turbines[ti].xy
        return out.reshape(self.n_sel_turbines*2)

    def min_values_float(self):
        """
        The minimal values of the float variables.

        Use -numpy.inf for unbounded.

        Returns
        -------
        values : numpy.ndarray
            Minimal float values, shape: (n_vars_float,)

        """
        b = self.farm.boundary
        assert b is not None, f"Problem '{self.name}': Missing wind farm boundary."
        out = np.zeros((self.n_sel_turbines, 2), dtype=FC.DTYPE)
        out[:] = b.p_min()[None, :]
        return out.reshape(self.n_sel_turbines * 2)

    def max_values_float(self):
        """
        The maximal values of the float variables.

        Use numpy.inf for unbounded.

        Returns
        -------
        values : numpy.ndarray
            Maximal float values, shape: (n_vars_float,)

        """
        b = self.farm.boundary
        assert b is not None, f"Problem '{self.name}': Missing wind farm boundary."
        out = np.zeros((self.n_sel_turbines, 2), dtype=FC.DTYPE)
        out[:] = b.p_max()[None, :]
        return out.reshape(self.n_sel_turbines * 2)

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
        farm_vars = {
            FV.X: np.zeros((self.algo.n_states, self.n_sel_turbines), dtype=FC.DTYPE),
            FV.Y: np.zeros((self.algo.n_states, self.n_sel_turbines), dtype=FC.DTYPE),
        }
        xy = vars_float.reshape(self.n_sel_turbines, 2)
        farm_vars[FV.X][:] = xy[None, :, 0]
        farm_vars[FV.Y][:] = xy[None, :, 1]
        
        return farm_vars

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
        raise NotImplementedError

    def finalize_individual(self, vars_int, vars_float, verbosity=1):
        """
        Finalization, given the champion data.

        Parameters
        ----------
        vars_int : np.array
            The optimal integer variable values, shape: (n_vars_int,)
        vars_float : np.array
            The optimal float variable values, shape: (n_vars_float,)
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        problem_results : Any
            The results of the variable application
            to the problem
        objs : np.array
            The objective function values, shape: (n_objectives,)
        cons : np.array
            The constraints values, shape: (n_constraints,)

        """
        res, objs, cons = super().finalize_individual(vars_int, vars_float, verbosity)

        xy = vars_float.reshape(self.n_sel_turbines, 2)
        for ti in self.sel_turbines:
            self.farm.turbines[ti].xy = xy[ti]

        return res, objs, cons
