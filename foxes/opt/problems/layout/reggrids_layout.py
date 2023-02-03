import numpy as np
from copy import deepcopy

from foxes.opt.core import FarmOptProblem, PopStates
from foxes.models.turbine_models import Calculator
import foxes.variables as FV
import foxes.constants as FC


class RegGridsLayoutOptProblem(FarmOptProblem):
    """
    Places turbines on several regular grids and optimizes
    their parameters.

    Note that this problem has both int and float variables
    (mixed problem).

    Parameters
    ----------
    name : str
        The problem's name
    algo : foxes.core.Algorithm
        The algorithm
    min_spacing : float
        The minimal turbine spacing
    n_grids : int
        The number of grids
    max_n_row : int, optional
        The maximal number of turbines per 
        grid and row
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
    n_grids : int
        The number of grids
    max_n_row : int
        The maximal number of turbines per 
        grid and row

    """

    def __init__(
        self,
        name,
        algo,
        min_spacing,
        n_grids,
        max_n_row=None,
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
        self.n_grids = n_grids
        self.max_n_row = max_n_row

        self.NX = [f"nx{i}" for i in range(self.n_grids)]
        self.NY = [f"ny{i}" for i in range(self.n_grids)]
        self.OX = [f"ox{i}" for i in range(self.n_grids)]
        self.OY = [f"oy{i}" for i in range(self.n_grids)]
        self.DX = [f"dx{i}" for i in range(self.n_grids)]
        self.DY = [f"dy{i}" for i in range(self.n_grids)]
        self.ALPHA = [f"alpha{i}" for i in range(self.n_grids)]

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
        dvrs = drop_vars + [FV.TURBINE] if FV.TURBINE not in drop_vars else drop_vars
        super()._update_keep_models(dvrs, verbosity)

    def initialize(self, verbosity=1):
        """
        Initialize the object.

        Parameters
        ----------
        verbosity : int
            The verbosity level, 0 = silent

        """
        self._mname = self.name + "_calc"
        for t in self.algo.farm.turbines:
            if self._mname not in t.models:
                t.models.append(self._mname)
        self._turbine = deepcopy(self.farm.turbines[-1])

        super().initialize(verbosity)

        b = self.farm.boundary
        assert b is not None, f"Problem '{self.name}': Missing wind farm boundary."
        pmax = b.p_max()
        pmin = b.p_min()
        self._pmin = pmin
        self._xy0 = pmin
        self._span = pmax - pmin
        self._len = np.linalg.norm(self._span)
        self.max_spacing = self._len
        if self.max_n_row is None:
            n_steps = int(self._len/self.min_spacing)
            if n_steps * self.min_spacing < self._len:
                n_steps += 1
            self._nrow = n_steps + 1
        else:
            self._nrow = self.max_n_row
        self._gpts = self._nrow**2

        if verbosity > 0:
            print(f"Problem '{self.name}':")
            print(f"  xy0          = {self._xy0}")
            print(f"  span         = {self._len:.2f}")
            print(f"  min spacing  = {self.min_spacing:.2f}")
            print(f"  max spacing  = {self.max_spacing:.2f}")
            print(f"  n row pts    = {self._nrow}")
            print(f"  n grid pts   = {self._gpts}")
            print(f"  n grids      = {self.n_grids}")
            print(f"  n max turbns = {self.n_grids*self._gpts}")
            print(f"  turbine mdls = {self._turbine.models}")

        if self.farm.n_turbines < self._gpts:
            for i in range(self._gpts - self.farm.n_turbines):
                ti = len(self.farm.turbines)
                self.farm.turbines.append(deepcopy(self._turbine))
                self.farm.turbines[-1].index = ti
                self.farm.turbines[-1].name = f"T{ti}"
        elif self.farm.n_turbines > self._gpts:
            self.farm.turbines = self.farm.turbines[:self._gpts]
        self.algo.n_turbines = self._gpts
        self.sel_turbines = list(range(self._gpts))

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
            in_vars=[FV.VALID, FV.P, FV.CT],
            out_vars=[FV.VALID, FV.P, FV.CT],
            func=lambda valid, P, ct, st_sel: (valid, P*valid, ct*valid),
            pre_rotor=False)
        
        for t in self.farm.turbines:
            if not self._mname in t.models:
                t.models.append(self._mname)

    def var_names_int(self):
        """
        The names of int variables.

        Returns
        -------
        names : list of str
            The names of the int variables

        """
        return np.array(np.array([self.NX, self.NY]).T.flat)

    def initial_values_int(self):
        """
        The initial values of the int variables.

        Returns
        -------
        values : numpy.ndarray
            Initial int values, shape: (n_vars_int,)

        """
        return np.full(self.n_grids*2, 2, dtype=FC.ITYPE)

    def min_values_int(self):
        """
        The minimal values of the integer variables.

        Use -self.INT_INF for unbounded.

        Returns
        -------
        values : numpy.ndarray
            Minimal int values, shape: (n_vars_int,)

        """
        return np.ones(self.n_grids*2, dtype=FC.ITYPE)

    def max_values_int(self):
        """
        The maximal values of the integer variables.

        Use self.INT_INF for unbounded.

        Returns
        -------
        values : numpy.ndarray
            Maximal int values, shape: (n_vars_int,)

        """
        return np.full(self.n_grids*2, self._nrow, dtype=FC.ITYPE)

    def var_names_float(self):
        """
        The names of float variables.

        Returns
        -------
        names : list of str
            The names of the float variables

        """
        return np.array(np.array([
            self.OX, self.OY, self.DX, self.DY, self.ALPHA]).T.flat)

    def initial_values_float(self):
        """
        The initial values of the float variables.

        Returns
        -------
        values : numpy.ndarray
            Initial float values, shape: (n_vars_float,)

        """
        vals = np.zeros((self.n_grids, 5), dtype=FC.DTYPE)
        vals[:, :2] = self._xy0 + self._span/2
        vals[:, 2:4] = self.min_spacing
        return vals.reshape(self.n_grids*5)

    def min_values_float(self):
        """
        The minimal values of the float variables.

        Use -numpy.inf for unbounded.

        Returns
        -------
        values : numpy.ndarray
            Minimal float values, shape: (n_vars_float,)

        """
        vals = np.zeros((self.n_grids, 5), dtype=FC.DTYPE)
        vals[:, :2] = self._xy0 - self._len - self.min_spacing
        vals[:, 2:4] = self.min_spacing
        return vals.reshape(self.n_grids*5)

    def max_values_float(self):
        """
        The maximal values of the float variables.

        Use numpy.inf for unbounded.

        Returns
        -------
        values : numpy.ndarray
            Maximal float values, shape: (n_vars_float,)

        """
        vals = np.zeros((self.n_grids, 5), dtype=FC.DTYPE)
        vals[:, :2] = self._xy0 + self._len + self.min_spacing
        vals[:, 2:4] = self._len
        vals[:, 4] = 90.
        return vals.reshape(self.n_grids*5)

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
        n0 = self.farm.n_turbines
        nxny = vars_int.reshape(self.n_grids, 2)
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
            self.sel_turbines = range(n)

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
        n0 = self.farm.n_turbines
        n_pop = vars_int.shape[0]
        nxny = vars_int.reshape(n_pop, self.n_grids, 2)
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
            self.sel_turbines = range(n)

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
        vint = vars_int.reshape(self.n_grids, 2)
        vflt = vars_float.reshape(self.n_grids, 5)
        nx = vint[:, 0]
        ny = vint[:, 1]
        ox = vflt[:, 0]
        oy = vflt[:, 1]
        dx = vflt[:, 2]
        dy = vflt[:, 3]
        a = np.deg2rad(vflt[:, 4])
        n_states = self.algo.n_states
        n_turbines = self.farm.n_turbines
        
        nax = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], axis=-1)
        naz = np.zeros_like(nax)
        naz[:, 2] = 1
        nay = np.cross(naz, nax)

        valid = np.zeros((n_states, n_turbines), dtype=bool)
        pts = np.zeros((n_states, n_turbines, 2), dtype=FC.DTYPE)
        n0 = 0
        for gi in range(self.n_grids):

            n = nx[gi] * ny[gi]
            n1 = n0 + n

            qts = pts[:, n0:n1].reshape(n_states, nx[gi], ny[gi], 2)
            qts[:, :, :, 0] = ox[gi]
            qts[:, :, :, 1] = oy[gi]
            qts[:] += np.arange(nx[gi])[None, :, None, None] * dx[gi] * nax[gi, None, None, None, :2]
            qts[:] += np.arange(ny[gi])[None, None, :, None] * dy[gi] * nay[gi, None, None, None, :2]

            valid[:, n0:n1] = self.farm.boundary.points_inside(qts.reshape(n_states*n, 2)).reshape(n_states, n)

            # set points invalid which are too close to other grids:
            if n0 > 0:
                for i in range(n):
                    dists = np.linalg.norm(pts[0, n0+i, None, :] - pts[0, :n0, :], axis=-1)
                    if np.min(dists) < self.min_spacing:
                        valid[:, n0+i] = False

            n0 = n1
        pts[:, n1:] = self._xy0[None, None, :] - self._len

        farm_vars = {
            FV.X: pts[:, :, 0],
            FV.Y: pts[:, :, 1],
            FV.VALID: valid.astype(FC.DTYPE)
        }

        return farm_vars

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
        vint = vars_int.reshape(n_pop, self.n_grids, 2)
        vflt = vars_float.reshape(n_pop, self.n_grids, 5)
        nx = vint[:, :, 0]
        ny = vint[:, :, 1]
        ox = vflt[:, :, 0]
        oy = vflt[:, :, 1]
        dx = vflt[:, :, 2]
        dy = vflt[:, :, 3]
        a = np.deg2rad(vflt[:, :, 4])
        n_states = self._org_n_states
        n_turbines = self.farm.n_turbines

        a = np.deg2rad(a)
        nax = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], axis=-1)
        naz = np.zeros_like(nax)
        naz[..., 2] = 1
        nay = np.cross(naz, nax, axis=-1)

        valid = np.zeros((n_pop, n_states, n_turbines), dtype=bool)
        pts = np.zeros((n_pop, n_states, n_turbines, 2), dtype=FC.DTYPE)
        for pi in range(n_pop):
            n0 = 0
            for gi in range(self.n_grids):

                n = nx[pi, gi] * ny[pi, gi]
                n1 = n0 + n

                qts = pts[pi, :, n0:n1].reshape(n_states, nx[pi, gi], ny[pi, gi], 2)
                qts[:, :, :, 0] = ox[pi, gi]
                qts[:, :, :, 1] = oy[pi, gi]
                qts[:] += np.arange(nx[pi, gi])[None, :, None, None] * dx[pi, gi] * nax[pi, gi, None, None, None, :2]
                qts[:] += np.arange(ny[pi, gi])[None, None, :, None] * dy[pi, gi] * nay[pi, gi, None, None, None, :2]

                valid[pi, :, n0:n1] = self.farm.boundary.points_inside(qts.reshape(n_states*n, 2)).reshape(n_states, n)

                # set points invalid which are too close to other grids:
                if n0 > 0:
                    for i in range(n):
                        dists = np.linalg.norm(pts[pi, 0, n0+i, None, :] - pts[pi, 0, :n0, :], axis=-1)
                        if np.min(dists) < self.min_spacing:
                            valid[pi, :, n0+i] = False

                n0 = n1
            pts[pi, :, n1:] = self._xy0[None, None, :] - self._len

        farm_vars = {
            FV.X: pts[:, :, :, 0],
            FV.Y: pts[:, :, :, 1],
            FV.VALID: valid.astype(FC.DTYPE)
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
        print("FINALIZE",vars_int)
        farm_vars = self.apply_individual(vars_int, vars_float)
        print("VALID", self.algo.n_states, self.algo.n_turbines,np.sum(farm_vars[FV.VALID].to_numpy()))

        """
        if isinstance(self.algo.states, PopStates):
            self._reset_states(self.algo.states.states)
            self.algo.n_states = self._org_n_states

        self._update_farm_individual(vars_int, vars_float)
        self._update_models_individual(vars_int, vars_float)
        farm_vars = self.opt2farm_vars_individual(vars_int, vars_float)
        """

        sel = np.where(farm_vars[FV.VALID][0])[0]
        x = farm_vars[FV.X][0, sel]
        y = farm_vars[FV.Y][0, sel]
        
        self.farm.turbines = [t for i, t in enumerate(self.farm.turbines) if i in sel]
        for i, t in enumerate(self.farm.turbines):
            t.xy = np.array([x[i], y[i]], dtype=FC.DTYPE)
            t.models = [m for m in t.models if m not in [self.name, self._mname]]
            t.index = i
            t.name = f"T{i}"
        self.algo.update_n_turbines()

        self.algo.mbook.turbine_models[self.name].reset()
        pars = dict(finalize=True)
        pars.update(self.calc_farm_args)
        self._count += 1
        results = self.runner.run(self.algo.calc_farm, kwargs=pars)

        varsi, varsf = self._find_vars(vars_int, vars_float, self.objs)
        objs = self.objs.finalize_individual(varsi, varsf, results, verbosity)

        varsi, varsf = self._find_vars(vars_int, vars_float, self.cons)
        cons = self.cons.finalize_individual(varsi, varsf, results, verbosity)

        return results, objs, cons
