import numpy as np
from copy import deepcopy

from foxes.opt.core import FarmOptProblem, FarmVarsProblem
from foxes.models.turbine_models import Calculator
import foxes.variables as FV
import foxes.constants as FC
from .geom_layouts.geom_reggrids import GeomRegGrids


class RegGridsLayoutOptProblem(FarmVarsProblem):
    """
    Places turbines on several regular grids and optimizes
    their parameters.

    Note that this problem has both int and float variables
    (mixed problem).

    Attributes
    ----------
    min_spacing: float
        The minimal turbine spacing
    n_grids: int
        The number of grids
    max_n_row: int
        The maximal number of turbines per
        grid and row

    :group: opt.problems.layout

    """

    def __init__(
        self,
        name,
        algo,
        min_dist,
        n_grids=1,
        n_row_max=None,
        max_dist=None,
        runner=None,
        **kwargs,
    ):
        """
        Constraints.

        Parameters
        ----------
        name: str
            The problem's name
        algo: foxes.core.Algorithm
            The algorithm
        min_dist: float
            The minimal distance between points
        n_grids: int
            The number of grids
        n_row_max: int, optional
            The maximal number of points in a row
        max_dist: float, optional
            The maximal distance between points
        runner: foxes.core.Runner, optional
            The runner for running the algorithm
        kwargs: dict, optional
            Additional parameters for `FarmVarsProblem`

        """
        super().__init__(name, algo, runner, **kwargs)

        b = algo.farm.boundary
        assert b is not None, f"Problem '{self.name}': Missing wind farm boundary."

        self._geomp = GeomRegGrids(
            b,
            min_dist=min_dist,
            n_grids=n_grids,
            n_row_max=n_row_max,
            max_dist=max_dist,
        )

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
        self._geomp.objs = self.objs
        self._geomp.cons = self.cons
        self._geomp.initialize(verbosity)

        self._mname = self.name + "_calc"
        for t in self.algo.farm.turbines:
            if self._mname not in t.models:
                t.models.append(self._mname)
        self._turbine = deepcopy(self.farm.turbines[-1])

        self.algo.mbook.turbine_models[self._mname] = Calculator(
            in_vars=[FC.VALID, FV.P, FV.CT],
            out_vars=[FC.VALID, FV.P, FV.CT],
            func=lambda valid, P, ct, st_sel: (valid, P * valid, ct * valid),
            pre_rotor=False,
        )

        super().initialize(
            pre_rotor_vars=[FV.X, FV.Y, FC.VALID],
            post_rotor_vars=[],
            verbosity=verbosity,
            **kwargs,
        )

    def var_names_int(self):
        """
        The names of int variables.

        Returns
        -------
        names: list of str
            The names of the int variables

        """
        return self._geomp.var_names_int()

    def initial_values_int(self):
        """
        The initial values of the int variables.

        Returns
        -------
        values: numpy.ndarray
            Initial int values, shape: (n_vars_int,)

        """
        return self._geomp.initial_values_int()

    def min_values_int(self):
        """
        The minimal values of the integer variables.

        Use -self.INT_INF for unbounded.

        Returns
        -------
        values: numpy.ndarray
            Minimal int values, shape: (n_vars_int,)

        """
        return self._geomp.min_values_int()

    def max_values_int(self):
        """
        The maximal values of the integer variables.

        Use self.INT_INF for unbounded.

        Returns
        -------
        values: numpy.ndarray
            Maximal int values, shape: (n_vars_int,)

        """
        return self._geomp.max_values_int()

    def var_names_float(self):
        """
        The names of float variables.

        Returns
        -------
        names: list of str
            The names of the float variables

        """
        return self._geomp.var_names_float()

    def initial_values_float(self):
        """
        The initial values of the float variables.

        Returns
        -------
        values: numpy.ndarray
            Initial float values, shape: (n_vars_float,)

        """
        return self._geomp.initial_values_float()

    def min_values_float(self):
        """
        The minimal values of the float variables.

        Use -numpy.inf for unbounded.

        Returns
        -------
        values: numpy.ndarray
            Minimal float values, shape: (n_vars_float,)

        """
        return self._geomp.min_values_float()

    def max_values_float(self):
        """
        The maximal values of the float variables.

        Use numpy.inf for unbounded.

        Returns
        -------
        values: numpy.ndarray
            Maximal float values, shape: (n_vars_float,)

        """
        return self._geomp.max_values_float()

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
        n0 = self.farm.n_turbines
        nxny = vars_int.reshape(self._geomp.n_grids, 2)
        n = np.sum(np.product(nxny, axis=1))
        if n0 > n:
            self.farm.turbines = self.farm.turbines[:n]
        elif n0 < n:
            for i in range(n0, n):
                self.farm.turbines.append(deepcopy(self._turbine))
                self.farm.turbines[-1].index = n0 + i
                self.farm.turbines[-1].name = f"T{n0 + i}"
        if n != n0:
            self.algo.update_n_turbines()

        super().update_problem_individual(vars_int, vars_float)

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
        n0 = self.farm.n_turbines
        n_pop = vars_int.shape[0]
        nxny = vars_int.reshape(n_pop, self._geomp.n_grids, 2)
        n = np.max(np.sum(np.product(nxny, axis=2), axis=1))
        if n0 > n:
            self.farm.turbines = self.farm.turbines[:n]
        elif n0 < n:
            for i in range(n0, n):
                self.farm.turbines.append(deepcopy(self._turbine))
                self.farm.turbines[-1].index = n0 + i
                self.farm.turbines[-1].name = f"T{n0 + i}"
        if n != n0:
            self.algo.update_n_turbines()

        super().update_problem_population(vars_int, vars_float)

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
        pts, vld = self._geomp.apply_individual(vars_int, vars_float)

        n_pts = pts.shape[0]
        n_states = self.algo.n_states
        n_turbines = self.farm.n_turbines

        pmi = np.min(self._geomp._pmin)
        points = np.full((n_states, n_turbines, 2), pmi, dtype=FC.DTYPE)
        if n_pts <= n_turbines:
            points[:, :n_pts] = pts[None, :, :]
        else:
            points[:] = pts[None, :n_turbines, :]

        valid = np.zeros((n_states, n_turbines), dtype=FC.DTYPE)
        if n_pts <= n_turbines:
            valid[:, :n_pts] = vld[None, :]
        else:
            valid[:] = vld[None, :n_turbines]

        farm_vars = {FV.X: points[:, :, 0], FV.Y: points[:, :, 1], FC.VALID: valid}

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
        pts, vld = self._geomp.apply_population(vars_int, vars_float)

        n_pop, n_pts = vld.shape
        n_states = self._org_n_states
        n_turbines = self.farm.n_turbines

        pmi = np.min(self._geomp._pmin)
        points = np.full((n_pop, n_states, n_turbines, 2), pmi, dtype=FC.DTYPE)
        if n_pts <= n_turbines:
            points[:, :, :n_pts] = pts[:, None, :, :]
        else:
            points[:] = pts[:, None, :n_turbines, :]

        valid = np.zeros((n_pop, n_states, n_turbines), dtype=FC.DTYPE)
        if n_pts <= n_turbines:
            valid[:, :, :n_pts] = vld[:, None, :]
        else:
            valid[:] = vld[:, None, :n_turbines]

        farm_vars = {
            FV.X: points[:, :, :, 0],
            FV.Y: points[:, :, :, 1],
            FC.VALID: valid,
        }

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
        pts, vld = self._geomp.apply_individual(vars_int, vars_float)
        xy = pts[vld]
        n_xy = xy.shape[0]

        self.farm.turbines = self.farm.turbines[:n_xy]
        for ti, t in enumerate(self.farm.turbines):
            t.xy = xy[ti]
            t.index = ti
            t.name = f"T{ti}"
            t.models = [
                mname for mname in t.models if mname not in [self.name, self._mname]
            ]
        self.algo.update_n_turbines()

        return FarmOptProblem.finalize_individual(
            self, vars_int, vars_float, verbosity=1
        )
