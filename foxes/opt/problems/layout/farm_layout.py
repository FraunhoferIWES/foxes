import numpy as np

from foxes.opt.core.farm_opt_problem import FarmOptProblem
import foxes.variables as FV
import foxes.constants as FC


class FarmLayoutOptProblem(FarmOptProblem):
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
        super().update_problem_individual(vars_int, vars_float)

        xy = vars_float.reshape(self.n_sel_turbines, 2)
        for i, ti in enumerate(self.sel_turbines):
            t = self.algo.farm.turbines[ti]
            t.xy = xy[i]

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
        super().update_problem_population(vars_int, vars_float)

        n_pop = len(vars_float)
        n_ostates = self._org_n_states
        n_states = n_pop * n_ostates

        xy = vars_float.reshape(n_pop, self.n_sel_turbines, 2)
        sxy = np.zeros(
            (n_pop, n_ostates, self.n_sel_turbines, 2), dtype=vars_float.dtype
        )
        sxy[:] = xy[:, None, :, :]
        sxy = sxy.reshape(n_states, self.n_sel_turbines, 2)
        del xy

        for i, ti in enumerate(self.sel_turbines):
            t = self.algo.farm.turbines[ti]
            t.xy = sxy[:, i]
