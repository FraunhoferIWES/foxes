import numpy as np
from copy import deepcopy

from foxes.opt.core import FarmVarsProblem, FarmOptProblem
from foxes.models.turbine_models import Calculator
import foxes.variables as FV
import foxes.constants as FC


class RegularLayoutOptProblem(FarmVarsProblem):
    """
    Places turbines on a regular grid and optimizes
    its parameters.

    Attributes
    ----------
    min_spacing: float
        The minimal turbine spacing

    :group: opt.problems.layout

    """

    SPACING_X = "spacing_x"
    SPACING_Y = "spacing_y"
    OFFSET_X = "offset_X"
    OFFSET_Y = "offset_Y"
    ANGLE = "angle"

    def __init__(
        self,
        name,
        algo,
        min_spacing,
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
        min_spacing: float
            The minimal turbine spacing
        kwargs: dict, optional
            Additional parameters for `FarmVarsProblem`

        """
        super().__init__(name, algo, **kwargs)
        self.min_spacing = min_spacing

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

        b = self.farm.boundary
        assert b is not None, f"Problem '{self.name}': Missing wind farm boundary."
        pmax = b.p_max()
        pmin = b.p_min()
        self._pmin = pmin
        self._xy0 = 0.5 * (pmin + pmax)
        self._halfspan = (pmax - pmin) / 2
        self._halflen = np.linalg.norm(self._halfspan)
        self.max_spacing = 2 * (self._halflen + self.min_spacing)
        self._halfn = int(self._halflen / self.min_spacing)
        if self._halfn * self.min_spacing < self._halflen:
            self._halfn += 1
        self._nrow = 2 * self._halfn + 1
        self._nturb = self._nrow**2

        if verbosity > 0:
            print(f"Problem '{self.name}':")
            print(f"  xy0          = {self._xy0}")
            print(f"  span         = {np.linalg.norm(self._halfspan*2):.2f}")
            print(f"  min spacing  = {self.min_spacing:.2f}")
            print(f"  max spacing  = {self.max_spacing:.2f}")
            print(f"  n row turbns = {self._nrow}")
            print(f"  n turbines   = {self._nturb}")
            print(f"  turbine mdls = {self._turbine.models}")

        if self.farm.n_turbines < self._nturb:
            for i in range(self._nturb - self.farm.n_turbines):
                ti = len(self.farm.turbines)
                self.farm.turbines.append(deepcopy(self._turbine))
                self.farm.turbines[-1].index = ti
                self.farm.turbines[-1].name = f"T{ti}"
        elif self.farm.n_turbines > self._nturb:
            self.farm.turbines = self.farm.turbines[: self._nturb]
        self.algo.n_turbines = self._nturb

        super().initialize(
            pre_rotor_vars=[FV.X, FV.Y, FC.VALID],
            post_rotor_vars=[],
            verbosity=verbosity,
            **kwargs,
        )

    def var_names_float(self):
        """
        The names of float variables.

        Returns
        -------
        names: list of str
            The names of the float variables

        """
        return [
            self.SPACING_X,
            self.SPACING_Y,
            self.OFFSET_X,
            self.OFFSET_Y,
            self.ANGLE,
        ]

    def initial_values_float(self):
        """
        The initial values of the float variables.

        Returns
        -------
        values: numpy.ndarray
            Initial float values, shape: (n_vars_float,)

        """
        return [self.min_spacing, self.min_spacing, 0.0, 0.0, 0.0]

    def min_values_float(self):
        """
        The minimal values of the float variables.

        Use -numpy.inf for unbounded.

        Returns
        -------
        values: numpy.ndarray
            Minimal float values, shape: (n_vars_float,)

        """
        return [
            self.min_spacing,
            self.min_spacing,
            -self._halfspan[0] - self.min_spacing,
            -self._halfspan[1] - self.min_spacing,
            0.0,
        ]

    def max_values_float(self):
        """
        The maximal values of the float variables.

        Use numpy.inf for unbounded.

        Returns
        -------
        values: numpy.ndarray
            Maximal float values, shape: (n_vars_float,)

        """
        return [
            self.max_spacing,
            self.max_spacing,
            self._halfspan[0] + self.min_spacing,
            self._halfspan[1] + self.min_spacing,
            90.0,
        ]

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

        dx, dy, ox, oy, a = vars_float
        n_states = self.algo.n_states
        nx = self._nrow
        ny = self._nrow

        a = np.deg2rad(a)
        nax = np.array([np.cos(a), np.sin(a), 0.0], dtype=FC.DTYPE)
        nay = np.cross(np.array([0.0, 0.0, 1.0], dtype=FC.DTYPE), nax)
        x0 = self._xy0 + np.array([ox, oy], dtype=FC.DTYPE)

        pts = np.zeros((n_states, nx, ny, 2), dtype=FC.DTYPE)
        pts[:] = (
            x0[None, None, None, :]
            + np.arange(nx)[None, :, None, None] * dx * nax[None, None, None, :2]
            + np.arange(ny)[None, None, :, None] * dy * nay[None, None, None, :2]
        )

        pts = pts.reshape(n_states, nx * ny, 2)
        valid = self.farm.boundary.points_inside(pts.reshape(n_states * nx * ny, 2))

        farm_vars = {
            FV.X: pts[:, :, 0],
            FV.Y: pts[:, :, 1],
            FC.VALID: valid.reshape(n_states, nx * ny).astype(FC.DTYPE),
        }

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
        n_turbines = self.farm.n_turbines
        dx = vars_float[:, 0]
        dy = vars_float[:, 1]
        ox = vars_float[:, 2]
        oy = vars_float[:, 3]
        nx = self._nrow
        ny = self._nrow
        a = vars_float[:, 4]
        N = self._nturb

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
            np.arange(nx)[None, None, :, None, None]
            * dx[:, None, None, None, None]
            * nax[:, None, None, None, :2]
            + np.arange(ny)[None, None, None, :, None]
            * dy[:, None, None, None, None]
            * nay[:, None, None, None, :2]
        )

        qts = np.zeros((n_pop, n_states, n_turbines, 2))
        qts[:, :N] = pts.reshape(n_pop, n_states, N, 2)
        del pts

        valid = self.farm.boundary.points_inside(
            qts.reshape(n_pop * n_states * n_turbines, 2)
        )

        farm_vars = {
            FV.X: qts[:, :, :, 0],
            FV.Y: qts[:, :, :, 1],
            FC.VALID: valid.reshape(n_pop, n_states, n_turbines).astype(FC.DTYPE),
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
        farm_vars = self.opt2farm_vars_individual(vars_int, vars_float)
        sel = np.where(farm_vars[FC.VALID][0])[0]
        x = farm_vars[FV.X][0, sel]
        y = farm_vars[FV.Y][0, sel]

        self.farm.turbines = [t for i, t in enumerate(self.farm.turbines) if i in sel]
        for i, t in enumerate(self.farm.turbines):
            t.xy = np.array([x[i], y[i]], dtype=FC.DTYPE)
            t.models = [m for m in t.models if m not in [self.name, self._mname]]
            t.index = i
            t.name = f"T{i}"
        self.algo.update_n_turbines()

        return FarmOptProblem.finalize_individual(
            self, vars_int, vars_float, verbosity=1
        )
