import numpy as np
from scipy.spatial.distance import cdist
from iwopy import Constraint

import foxes.constants as FC


class Valid(Constraint):
    """
    Validity constraint for purely geometrical layouts problems.

    :group: opt.problems.layout.geom_layouts.constraints

    """

    def __init__(self, problem, name="valid", **kwargs):
        """
        Constructor.

        Parameters
        ----------
        problem: foxes.opt.FarmOptProblem
            The underlying geometrical layout
            optimization problem
        name: str
            The constraint name
        kwargs: dict, optional
            Additioal parameters for the base class

        """
        super().__init__(
            problem,
            name,
            vnames_int=problem.var_names_int(),
            vnames_float=problem.var_names_float(),
            **kwargs,
        )

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

    def calc_individual(self, vars_int, vars_float, problem_results, cmpnts=None):
        """
        Calculate values for a single individual of the
        underlying problem.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float : np.array
            The float variable values, shape: (n_vars_float,)
        problem_results : Any
            The results of the variable application
            to the problem
        components : list of int, optional
            The selected components or None for all

        Returns
        -------
        values : np.array
            The component values, shape: (n_sel_components,)

        """
        __, valid = problem_results
        return np.sum(~valid)

    def calc_population(self, vars_int, vars_float, problem_results, cmpnts=None):
        """
        Calculate values for all individuals of a population.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float : np.array
            The float variable values, shape: (n_pop, n_vars_float)
        problem_results : Any
            The results of the variable application
            to the problem
        components : list of int, optional
            The selected components or None for all

        Returns
        -------
        values : np.array
            The component values, shape: (n_pop, n_sel_components)

        """
        __, valid = problem_results
        return np.sum(~valid, axis=1)[:, None]


class Boundary(Constraint):
    """
    Boundary constraint for purely geometrical layouts problems.

    :group: opt.problems.layout.geom_layouts.constraints

    """

    def __init__(self, problem, n_turbines=None, D=None, name="boundary", **kwargs):
        """
        Constructor.

        Parameters
        ----------
        problem: foxes.opt.FarmOptProblem
            The underlying geometrical layout
            optimization problem
        n_turbines: int, optional
            The number of turbines
        D: float, optional
            The rotor diameter
        name: str
            The constraint name
        kwargs: dict, optional
            Additioal parameters for the base class

        """
        super().__init__(
            problem,
            name,
            vnames_int=problem.var_names_int(),
            vnames_float=problem.var_names_float(),
            **kwargs,
        )
        self.n_turbines = problem.n_turbines if n_turbines is None else n_turbines
        self.D = problem.D if D is None else D

    def n_components(self):
        """
        Returns the number of components of the
        function.

        Returns
        -------
        int:
            The number of components.

        """
        return self.n_turbines

    def calc_individual(self, vars_int, vars_float, problem_results, cmpnts=None):
        """
        Calculate values for a single individual of the
        underlying problem.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float : np.array
            The float variable values, shape: (n_vars_float,)
        problem_results : Any
            The results of the variable application
            to the problem
        components : list of int, optional
            The selected components or None for all

        Returns
        -------
        values : np.array
            The component values, shape: (n_sel_components,)

        """
        xy, __ = problem_results

        dists = self.problem.boundary.points_distance(xy)
        dists[self.problem.boundary.points_inside(xy)] *= -1

        if self.D is not None:
            dists += self.D / 2

        return dists

    def calc_population(self, vars_int, vars_float, problem_results, cmpnts=None):
        """
        Calculate values for all individuals of a population.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float : np.array
            The float variable values, shape: (n_pop, n_vars_float)
        problem_results : Any
            The results of the variable application
            to the problem
        components : list of int, optional
            The selected components or None for all

        Returns
        -------
        values : np.array
            The component values, shape: (n_pop, n_sel_components)

        """
        xy, __ = problem_results
        n_pop, n_xy = xy.shape[:2]

        xy = xy.reshape(n_pop * n_xy, 2)
        dists = self.problem.boundary.points_distance(xy)
        dists[self.problem.boundary.points_inside(xy)] *= -1
        dists = dists.reshape(n_pop, n_xy)

        if self.D is not None:
            dists += self.D / 2

        return dists


class MinDist(Constraint):
    """
    Minimal distance constraint for purely geometrical layouts problems.

    :group: opt.problems.layout.geom_layouts.constraints

    """

    def __init__(
        self, problem, min_dist=None, n_turbines=None, name="min_dist", **kwargs
    ):
        """
        Constructor.

        Parameters
        ----------
        problem: foxes.opt.FarmOptProblem
            The underlying geometrical layout
            optimization problem
        min_dist: float, optional
            The minimal distance between turbines
        n_turbines: int, optional
            The number of turbines
        name: str
            The constraint name
        kwargs: dict, optional
            Additioal parameters for the base class

        """
        super().__init__(
            problem,
            name,
            vnames_int=problem.var_names_int(),
            vnames_float=problem.var_names_float(),
            **kwargs,
        )
        self.min_dist = problem.min_dist if min_dist is None else min_dist
        self.n_turbines = problem.n_turbines if n_turbines is None else n_turbines

    def initialize(self, verbosity=0):
        """
        Initialize the constaint.

        Parameters
        ----------
        verbosity: int
            The verbosity level, 0 = silent

        """
        N = self.n_turbines
        self._i2t = []  # i --> (ti, tj)
        self._t2i = np.full([N, N], -1)  # (ti, tj) --> i
        i = 0
        for ti in range(N):
            for tj in range(N):
                if ti != tj and self._t2i[ti, tj] < 0:
                    self._i2t.append([ti, tj])
                    self._t2i[ti, tj] = i
                    self._t2i[tj, ti] = i
                    i += 1
        self._i2t = np.array(self._i2t)
        self._cnames = [f"{self.name}_{ti}_{tj}" for ti, tj in self._i2t]
        super().initialize(verbosity)

    def n_components(self):
        """
        Returns the number of components of the
        function.

        Returns
        -------
        int:
            The number of components.

        """
        return len(self._i2t)

    def calc_individual(self, vars_int, vars_float, problem_results, cmpnts=None):
        """
        Calculate values for a single individual of the
        underlying problem.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float : np.array
            The float variable values, shape: (n_vars_float,)
        problem_results : Any
            The results of the variable application
            to the problem
        components : list of int, optional
            The selected components or None for all

        Returns
        -------
        values : np.array
            The component values, shape: (n_sel_components,)

        """
        xy, __ = problem_results

        a = np.take_along_axis(xy, self._i2t[:, 0, None], axis=0)
        b = np.take_along_axis(xy, self._i2t[:, 1, None], axis=0)
        d = np.linalg.norm(a - b, axis=-1)

        return self.min_dist - d

    def calc_population(self, vars_int, vars_float, problem_results, cmpnts=None):
        """
        Calculate values for all individuals of a population.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float : np.array
            The float variable values, shape: (n_pop, n_vars_float)
        problem_results : Any
            The results of the variable application
            to the problem
        components : list of int, optional
            The selected components or None for all

        Returns
        -------
        values : np.array
            The component values, shape: (n_pop, n_sel_components)

        """
        xy, __ = problem_results

        a = np.take_along_axis(xy, self._i2t[None, :, 0, None], axis=1)
        b = np.take_along_axis(xy, self._i2t[None, :, 1, None], axis=1)
        d = np.linalg.norm(a - b, axis=-1)

        return self.min_dist - d


class CMinN(Constraint):
    """
    Minimal number of turbines constraint for purely geometrical layouts problems.

    :group: opt.problems.layout.geom_layouts.constraints

    """

    def __init__(self, problem, N, name="cminN", **kwargs):
        super().__init__(
            problem,
            name,
            vnames_int=problem.var_names_int(),
            vnames_float=problem.var_names_float(),
            **kwargs,
        )
        """
        Constructor.
        
        Parameters
        ----------
        problem: foxes.opt.FarmOptProblem
            The underlying geometrical layout 
            optimization problem
        N: int
            The minimal number of turbines
        name: str
            The constraint name
        kwargs: dict, optional
            Additioal parameters for the base class

        """
        self.N = N

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

    def calc_individual(self, vars_int, vars_float, problem_results, cmpnts=None):
        """
        Calculate values for a single individual of the
        underlying problem.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float : np.array
            The float variable values, shape: (n_vars_float,)
        problem_results : Any
            The results of the variable application
            to the problem
        components : list of int, optional
            The selected components or None for all

        Returns
        -------
        values : np.array
            The component values, shape: (n_sel_components,)

        """
        __, valid = problem_results
        return self.N - np.sum(valid)

    def calc_population(self, vars_int, vars_float, problem_results, cmpnts=None):
        """
        Calculate values for all individuals of a population.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float : np.array
            The float variable values, shape: (n_pop, n_vars_float)
        problem_results : Any
            The results of the variable application
            to the problem
        components : list of int, optional
            The selected components or None for all

        Returns
        -------
        values : np.array
            The component values, shape: (n_pop, n_sel_components)

        """
        __, valid = problem_results
        return self.N - np.sum(valid, axis=1)[:, None]


class CMaxN(Constraint):
    """
    Maximal number of turbines constraint for purely geometrical layouts problems.

    :group: opt.problems.layout.geom_layouts.constraints

    """

    def __init__(self, problem, N, name="cmaxN", **kwargs):
        """
        Constructor.

        Parameters
        ----------
        problem: foxes.opt.FarmOptProblem
            The underlying geometrical layout
            optimization problem
        N: int
            The maximal number of turbines
        name: str
            The constraint name
        kwargs: dict, optional
            Additioal parameters for the base class

        """
        super().__init__(
            problem,
            name,
            vnames_int=problem.var_names_int(),
            vnames_float=problem.var_names_float(),
            **kwargs,
        )
        self.N = N

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

    def calc_individual(self, vars_int, vars_float, problem_results, cmpnts=None):
        """
        Calculate values for a single individual of the
        underlying problem.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float : np.array
            The float variable values, shape: (n_vars_float,)
        problem_results : Any
            The results of the variable application
            to the problem
        components : list of int, optional
            The selected components or None for all

        Returns
        -------
        values : np.array
            The component values, shape: (n_sel_components,)

        """
        __, valid = problem_results
        return np.sum(valid) - self.N

    def calc_population(self, vars_int, vars_float, problem_results, cmpnts=None):
        """
        Calculate values for all individuals of a population.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float : np.array
            The float variable values, shape: (n_pop, n_vars_float)
        problem_results : Any
            The results of the variable application
            to the problem
        components : list of int, optional
            The selected components or None for all

        Returns
        -------
        values : np.array
            The component values, shape: (n_pop, n_sel_components)

        """
        __, valid = problem_results
        return np.sum(valid, axis=1)[:, None] - self.N


class CFixN(Constraint):
    """
    Fixed number of turbines constraint for purely geometrical layouts problems.

    :group: opt.problems.layout.geom_layouts.constraints

    """

    def __init__(self, problem, N, name="cfixN", **kwargs):
        """
        Constructor.

        Parameters
        ----------
        problem: foxes.opt.FarmOptProblem
            The underlying geometrical layout
            optimization problem
        N: int
            The number of turbines
        name: str
            The constraint name
        kwargs: dict, optional
            Additioal parameters for the base class

        """
        super().__init__(
            problem,
            name,
            vnames_int=problem.var_names_int(),
            vnames_float=problem.var_names_float(),
            tol=0.1,
            **kwargs,
        )
        self.N = N

    def n_components(self):
        """
        Returns the number of components of the
        function.

        Returns
        -------
        int:
            The number of components.

        """
        return 2

    def calc_individual(self, vars_int, vars_float, problem_results, cmpnts=None):
        """
        Calculate values for a single individual of the
        underlying problem.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float : np.array
            The float variable values, shape: (n_vars_float,)
        problem_results : Any
            The results of the variable application
            to the problem
        components : list of int, optional
            The selected components or None for all

        Returns
        -------
        values : np.array
            The component values, shape: (n_sel_components,)

        """
        __, valid = problem_results
        vld = np.sum(valid)
        return np.array([self.N - vld, vld - self.N])

    def calc_population(self, vars_int, vars_float, problem_results, cmpnts=None):
        """
        Calculate values for all individuals of a population.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float : np.array
            The float variable values, shape: (n_pop, n_vars_float)
        problem_results : Any
            The results of the variable application
            to the problem
        components : list of int, optional
            The selected components or None for all

        Returns
        -------
        values : np.array
            The component values, shape: (n_pop, n_sel_components)

        """
        __, valid = problem_results
        vld = np.sum(valid, axis=1)
        return np.stack([self.N - vld, vld - self.N], axis=-1)


class CMinDensity(Constraint):
    """
    Minimal turbine density constraint for purely geometrical layouts problems.

    :group: opt.problems.layout.geom_layouts.constraints

    """

    def __init__(self, problem, min_value, dfactor=1, name="min_density"):
        """
        Constructor.

        Parameters
        ----------
        problem: foxes.opt.FarmOptProblem
            The underlying geometrical layout
            optimization problem
        min_value: float
            The minimal turbine density
        dfactor: float
            Delta factor for grid spacing
        name: str
            The constraint name
        kwargs: dict, optional
            Additioal parameters for the base class

        """
        super().__init__(
            problem,
            name,
            vnames_int=problem.var_names_int(),
            vnames_float=problem.var_names_float(),
        )
        self.min_value = min_value
        self.dfactor = dfactor

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

    def initialize(self, verbosity):
        """
        Initialize the object.

        Parameters
        ----------
        verbosity: int
            The verbosity level, 0 = silent

        """
        super().initialize(verbosity)

        # define regular grid of probe points:
        geom = self.problem.boundary
        pmin = geom.p_min()
        pmax = geom.p_max()
        detlta = self.problem.min_dist / self.dfactor
        self._probes = np.stack(
            np.meshgrid(
                np.arange(pmin[0] - detlta, pmax[0] + 2 * detlta, detlta),
                np.arange(pmin[1] - detlta, pmax[1] + 2 * detlta, detlta),
                indexing="ij",
            ),
            axis=-1,
        )
        nx, ny = self._probes.shape[:2]
        n = nx * ny
        self._probes = self._probes.reshape(n, 2)

        # reduce to points within geometry:
        valid = geom.points_inside(self._probes)
        self._probes = self._probes[valid]

    def calc_individual(self, vars_int, vars_float, problem_results, cmpnts=None):
        """
        Calculate values for a single individual of the
        underlying problem.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float : np.array
            The float variable values, shape: (n_vars_float,)
        problem_results : Any
            The results of the variable application
            to the problem
        components : list of int, optional
            The selected components or None for all

        Returns
        -------
        values : np.array
            The component values, shape: (n_sel_components,)

        """
        xy, valid = problem_results
        xy = xy[valid]
        dists = cdist(self._probes, xy)
        return np.nanmax(np.nanmin(dists, axis=1)) - self.min_value

    def calc_population(self, vars_int, vars_float, problem_results, cmpnts=None):
        """
        Calculate values for all individuals of a population.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float : np.array
            The float variable values, shape: (n_pop, n_vars_float)
        problem_results : Any
            The results of the variable application
            to the problem
        components : list of int, optional
            The selected components or None for all

        Returns
        -------
        values : np.array
            The component values, shape: (n_pop, n_sel_components)

        """
        n_pop = vars_float.shape[0]
        xy, valid = problem_results
        out = np.full(n_pop, 1e20, dtype=FC.DTYPE)
        for pi in range(n_pop):
            if np.any(valid[pi]):
                hxy = xy[pi][valid[pi]]
                dists = cdist(self._probes, hxy)
                out[pi] = np.nanmax(np.nanmin(dists, axis=1)) - self.min_value
        return out[:, None]
