import numpy as np

from foxes.opt.core import FarmVarsProblem
import foxes.variables as FV
import foxes.constants as FC


class FarmLayoutOptProblem(FarmVarsProblem):
    """
    The turbine positioning optimization problem

    :group: opt.problems.layout

    """

    def var_names_float(self):
        """
        The names of float variables.

        Returns
        -------
        names: list of str
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
        values: numpy.ndarray
            Initial float values, shape: (n_vars_float,)

        """
        out = np.zeros((self.n_sel_turbines, 2), dtype=FC.DTYPE)
        for i, ti in enumerate(self.sel_turbines):
            out[i] = self.farm.turbines[ti].xy
        return out.reshape(self.n_sel_turbines * 2)

    def min_values_float(self):
        """
        The minimal values of the float variables.

        Use -numpy.inf for unbounded.

        Returns
        -------
        values: numpy.ndarray
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
        values: numpy.ndarray
            Maximal float values, shape: (n_vars_float,)

        """
        b = self.farm.boundary
        assert b is not None, f"Problem '{self.name}': Missing wind farm boundary."
        out = np.zeros((self.n_sel_turbines, 2), dtype=FC.DTYPE)
        out[:] = b.p_max()[None, :]
        return out.reshape(self.n_sel_turbines * 2)

    def initialize(self, verbosity=1, **kwargs):
        """
        Initialize the object.

        Parameters
        ----------
        verbosity: int
            The verbosity level, 0 = silent
        kwargs: dict, optional
            Additional parameters for super class init

        """
        super().initialize(
            pre_rotor_vars=[FV.X, FV.Y],
            post_rotor_vars=[],
            verbosity=verbosity,
            **kwargs,
        )

    def opt2farm_vars_individual(self, vars_int, vars_float):
        """
        Translates optimization variables to farm variables

        Parameters
        ----------
        vars_int: numpy.ndarray
            The integer optimization variable values,
            shape: (n_vars_int,)
        vars_float: numpy.ndarray
            The float optimization variable values,
            shape: (n_vars_float,)

        Returns
        -------
        farm_vars: dict
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

    def opt2farm_vars_population(self, vars_int, vars_float, n_states):
        """
        Translates optimization variables to farm variables

        Parameters
        ----------
        vars_int: numpy.ndarray
            The integer optimization variable values,
            shape: (n_pop, n_vars_int)
        vars_float: numpy.ndarray
            The float optimization variable values,
            shape: (n_pop, n_vars_float)
        n_states: int
            The number of original (non-pop) states

        Returns
        -------
        farm_vars: dict
            The foxes farm variables. Key: var name,
            value: numpy.ndarray with values, shape:
            (n_pop, n_states, n_sel_turbines)

        """
        n_pop = len(vars_float)
        farm_vars = {
            FV.X: np.zeros((n_pop, n_states, self.n_sel_turbines), dtype=FC.DTYPE),
            FV.Y: np.zeros((n_pop, n_states, self.n_sel_turbines), dtype=FC.DTYPE),
        }
        xy = vars_float.reshape(n_pop, self.n_sel_turbines, 2)
        farm_vars[FV.X][:] = xy[:, None, :, 0]
        farm_vars[FV.Y][:] = xy[:, None, :, 1]

        return farm_vars

    def finalize_individual(self, vars_int, vars_float, verbosity=1):
        """
        Finalization, given the champion data.

        Parameters
        ----------
        vars_int: np.array
            The optimal integer variable values, shape: (n_vars_int,)
        vars_float: np.array
            The optimal float variable values, shape: (n_vars_float,)
        verbosity: int
            The verbosity level, 0 = silent

        Returns
        -------
        problem_results: Any
            The results of the variable application
            to the problem
        objs: np.array
            The objective function values, shape: (n_objectives,)
        cons: np.array
            The constraints values, shape: (n_constraints,)

        """
        res, objs, cons = super().finalize_individual(vars_int, vars_float, verbosity)

        xy = vars_float.reshape(self.n_sel_turbines, 2)
        for i, ti in enumerate(self.sel_turbines):
            self.farm.turbines[ti].xy = xy[i]

        return res, objs, cons
