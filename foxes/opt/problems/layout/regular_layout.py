import numpy as np
from copy import deepcopy

from foxes.opt.core import FarmOptProblem
from foxes.models.turbine_models import Calculator
import foxes.variables as FV
import foxes.constants as FC
from foxes.utils import wd2uv


class RegularLayoutOptProblem(FarmOptProblem):
    """
    Places turbines on a regular grid and optimizes
    its parameters.

    Parameters
    ----------
    name : str
        The problem's name
    algo : foxes.core.Algorithm
        The algorithm
    min_spacing : float
        The minimal turbine spacing
    max_spacing : float
        The maximal turbine spacing
    n_row_max : int
        The maximal number of turbines in a row/column
    runner : foxes.core.Runner, optional
        The runner for running the algorithm
    calc_farm_args : dict
        Additional parameters for algo.calc_farm()
    kwargs : dict, optional
        Additional parameters for `FarmOptProblem`

    Attributes
    ----------
    min_spacing : float
        The minimal turbine spacing
    max_spacing : float
        The maximal turbine spacing
    n_row_max : int
        The maximal number of turbines in a row/column

    """

    N_X = "n_x"
    N_Y = "n_y"
    SPACING_X = "spacing_x"
    SPACING_Y = "spacing_y"
    OFFSET_X = "offset_X"
    OFFSET_Y = "offset_Y"
    ANGLE = "angle"
    VALID = "valid"

    def __init__(
        self,
        name,
        algo,
        min_spacing,
        max_spacing=1e6,
        n_row_max=None,
        runner=None,
        calc_farm_args={},
        **kwargs,
    ):
        super().__init__(
            name,
            algo,
            runner,
            pre_rotor=True,
            calc_farm_args=calc_farm_args,
            **kwargs,
        )

        self.min_spacing = min_spacing
        self.max_spacing = max_spacing
        self.n_row_max = self.farm.n_turbines if n_row_max is None else n_row_max

        b = self.farm.boundary
        assert b is not None, f"Problem '{name}': Missing wind farm boundary."
        pmax = b.p_max()
        pmin = b.p_min()
        self._xy0 = pmin
        self._xy_span = pmax - pmin

        self._turbine = deepcopy(self.farm.turbines[0])
        self._mname = self.name + "_calc"

    def _init_mbook(self, verbosity=1):
        """
        Initialize the model book

        Parameters
        ----------
        verbosity : int
            The verbosity level, 0 = silent

        """
        super()._init_mbook(verbosity)

        self.algo.mbook.turbine_models[self._mname] = Calculator(
            in_vars=[self.VALID, FV.P, FV.CT],
            out_vars=[self.VALID, FV.P, FV.CT],
            func=lambda valid, P, ct, st_sel: (valid, P*valid, ct*valid),
            pre_rotor=False)
        
        for t in self.algo.farm.turbines:
            if self._mname not in t.models:
                t.models.append(self._mname)

    def _update_keep_models(self, drop_vars, verbosity=1):
        """
        Updates algo.keep_models during initialization

        Parameters
        ----------
        drop_vars : list of str
            Variables that decided about dropping model
            from algo.keep_models
        verbosity : int
            The verbosity level, 0 = silent

        """
        # the turbine number changes during optimization,
        # hence drop all models that store turbine related data:
        if FV.TURBINE not in drop_vars:
            drop_vars += [FV.TURBINE]
        super()._update_keep_models(drop_vars, verbosity)

    def var_names_int(self):
        """
        The names of integer variables.

        Returns
        -------
        names : list of str
            The names of the integer variables

        """
        return [self.N_X, self.N_Y]

    def var_names_float(self):
        """
        The names of float variables.

        Returns
        -------
        names : list of str
            The names of the float variables

        """
        return [
            self.SPACING_X, 
            self.SPACING_Y, 
            self.OFFSET_X,
            self.OFFSET_Y,
            self.ANGLE]

    def initial_values_int(self):
        """
        The initial values of the integer variables.

        Returns
        -------
        values : numpy.ndarray
            Initial int values, shape: (n_vars_int,)

        """
        return [1, 1]

    def initial_values_float(self):
        """
        The initial values of the float variables.

        Returns
        -------
        values : numpy.ndarray
            Initial float values, shape: (n_vars_float,)

        """
        return [
            self.min_spacing, 
            self.min_spacing, 
            0., 0., -90.]

    def min_values_int(self):
        """
        The minimal values of the integer variables.

        Use -self.INT_INF for unbounded.

        Returns
        -------
        values : numpy.ndarray
            Minimal int values, shape: (n_vars_int,)

        """
        return [1, 1]

    def min_values_float(self):
        """
        The minimal values of the float variables.

        Use -numpy.inf for unbounded.

        Returns
        -------
        values : numpy.ndarray
            Minimal float values, shape: (n_vars_float,)

        """
        return [
            self.min_spacing, 
            self.min_spacing, 
            -self.min_spacing, 
            -self.min_spacing, 
            0.]

    def max_values_int(self):
        """
        The maximal values of the integer variables.

        Use self.INT_INF for unbounded.

        Returns
        -------
        values : numpy.ndarray
            Maximal int values, shape: (n_vars_int,)

        """
        return [self.n_row_max, self.n_row_max]

    def max_values_float(self):
        """
        The maximal values of the float variables.

        Use numpy.inf for unbounded.

        Returns
        -------
        values : numpy.ndarray
            Maximal float values, shape: (n_vars_float,)

        """
        return [
                self.max_spacing, 
                self.max_spacing, 
                self._xy_span[0] + self.min_spacing,
                self._xy_span[1] + self.min_spacing,
                360.]

    def _update_farm_individual(self, vars_int, vars_float):
        """
        Update basic wind farm data during optimization,
        for example the number of turbines

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float : np.array
            The float variable values, shape: (n_vars_float,)

        """
        n = np.product(vars_int)
        if self.farm.n_turbines < n:
            self.farm.turbines += (n - self.farm.n_turbines) * self._turbine
        elif self.farm.n_turbines > n:
            self.farm.turbines = self.farm.turbines[:n]

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

        nx, ny = vars_int
        dx, dy, ox, oy, a = vars_float
        n_states = self.algo.n_states

        a = np.deg2rad(a)
        nax = np.array([np.cos(a), np.sin(a), 0.], dtype=FC.DTYPE)
        nay = np.cross(np.array([0., 0., 1.], dtype=FC.DTYPE), nax)
        x0 = self._xy0 + np.array([ox, oy], dtype=FC.DTYPE)

        pts = np.zeros((n_states, nx, ny, 2), dtype=FC.DTYPE)
        pts[:] = (
            x0[None, None, None, :] 
            + np.arange(nx)[None, :, None, None] * dx * nax[None, None, None, :2] 
            + np.arange(ny)[None, None, :, None] * dy * nay[None, None, None, :2]
        )

        pts = pts.reshape(n_states, nx*ny, 2)
        valid = self.farm.boundary.points_inside(pts.reshape(n_states*nx*ny))

        farm_vars = {
            FV.X: pts[:, :, 0],
            FV.Y: pts[:, :, 1],
            self.VALID: np.astype(valid.reshape(n_states, nx*ny), FC.DTYPE)
        }

        return farm_vars

    def _update_farm_population(self, vars_int, vars_float):
        """
        Update basic wind farm data during optimization,
        for example the number of turbines

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float : np.array
            The float variable values, shape: (n_pop, n_vars_float)

        """
        n = np.max(vars_int[:, 0]) * np.max(vars_int[:, 1])
        if self.farm.n_turbines < n:
            self.farm.turbines += (n - self.farm.n_turbines) * self._turbine
        elif self.farm.n_turbines > n:
            self.farm.turbines = self.farm.turbines[:n]

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
        n_pop = len(vars_float)
        n_turbines = self.farm.n_turbines
        nx = vars_int[:, 0]
        ny = vars_int[:, 1]
        dx = vars_float[:, 0]
        dy = vars_float[:, 1] 
        ox = vars_float[:, 2]
        oy = vars_float[:, 3]
        a = vars_float[:, 4]
        N = nx*ny

        a = np.deg2rad(a)
        nax = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], axis=-1)
        naz = np.zeros_like(nax)
        naz[..., 2] = 1
        nay = np.cross(naz, nax)

        pts = np.zeros((n_pop, n_states, nx, ny, 2), dtype=FC.DTYPE)
        pts[:] = self._xy0[None, None, None, None, :]
        pts[..., 0] += ox[:, None, None, None]
        pts[..., 1] += oy[:, None, None, None]
        pts[:] += (
            np.arange(nx)[None, None, :, None, None] * dx * nax[None, None, None, :2] 
            + np.arange(ny)[None, None, None, :, None] * dy * nay[None, None, None, :2]
            )

        qts = np.zeros((n_pop, n_states, n_turbines, 2)) 
        qts[:, :N] = pts.reshape(n_pop, n_states, N, 2)
        qts[:, N:] = self._xy0[None, None, None, :] - 1000
        del pts

        valid = self.farm.boundary.points_inside(qts.reshape(n_states*n_turbines, 2))

        farm_vars = {
            FV.X: pts[:, :, 0],
            FV.Y: pts[:, :, 1],
            self.VALID: np.astype(valid.reshape(n_states, n_turbines), FC.DTYPE)
        }

        return farm_vars

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

        for ti in range(self.farm.n_turbines):
            self.farm.turbines[ti].xy = np.array(
                [res[FV.X][0, ti], res[FV.Y][0, ti]], dtype=FC.DTYPE)

        return res, objs, cons
