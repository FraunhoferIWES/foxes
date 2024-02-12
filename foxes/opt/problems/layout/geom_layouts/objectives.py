import numpy as np
from iwopy import Objective
from scipy.spatial.distance import cdist

import foxes.constants as FC


class OMaxN(Objective):
    """
    Maximal number of turbines objective
    for purely geometrical layouts problems.

    :group: opt.problems.layout.geom_layouts.objectives

    """

    def __init__(self, problem, name="maxN"):
        """
        Constructor.

        Parameters
        ----------
        problem: foxes.opt.FarmOptProblem
            The underlying geometrical layout
            optimization problem
        name: str
            The constraint name

        """
        super().__init__(
            problem,
            name,
            vnames_int=problem.var_names_int(),
            vnames_float=problem.var_names_float(),
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
        return np.sum(valid)

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
        return np.sum(valid, axis=1)[:, None]


class OMinN(OMaxN):
    """
    Minimal number of turbines objective
    for purely geometrical layouts problems.

    :group: opt.problems.layout.geom_layouts.objectives

    """

    def __init__(self, problem, name="ominN"):
        """
        Constructor.

        Parameters
        ----------
        problem: foxes.opt.FarmOptProblem
            The underlying geometrical layout
            optimization problem
        name: str
            The constraint name

        """
        super().__init__(problem, name)

    def maximize(self):
        return [False]


class OFixN(Objective):
    """
    Fixed number of turbines objective
    for purely geometrical layouts problems.

    :group: opt.problems.layout.geom_layouts.objectives

    """

    def __init__(self, problem, N, name="ofixN"):
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

        """
        super().__init__(
            problem,
            name,
            vnames_int=problem.var_names_int(),
            vnames_float=problem.var_names_float(),
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

    def maximize(self):
        """
        Returns flag for maximization of each component.

        Returns
        -------
        flags: np.array
            Bool array for component maximization,
            shape: (n_components,)

        """
        return [False]

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
        N = np.sum(valid, dtype=np.float64)
        return np.maximum(N - self.N, self.N - N)

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
        N = np.sum(valid, axis=1, dtype=np.float64)[:, None]
        return np.maximum(N - self.N, self.N - N)


class MaxGridSpacing(Objective):
    """
    Maximal grid spacing objective
    for purely geometrical layouts problems.

    :group: opt.problems.layout.geom_layouts.objectives

    """

    def __init__(self, problem, name="max_dxdy"):
        """
        Constructor.

        Parameters
        ----------
        problem: foxes.opt.FarmOptProblem
            The underlying geometrical layout
            optimization problem
        name: str
            The constraint name

        """
        super().__init__(
            problem,
            name,
            vnames_int=problem.var_names_int(),
            vnames_float=problem.var_names_float(),
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
        vflt = vars_float.reshape(self.problem.n_grids, 5)
        delta = np.minimum(vflt[:, 2], vflt[:, 3])
        return np.nanmin(delta)

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
        vflt = vars_float.reshape(n_pop, self.problem.n_grids, 5)
        delta = np.minimum(vflt[:, :, 2], vflt[:, :, 3])
        return np.nanmin(delta, axis=1)[:, None]


class MaxDensity(Objective):
    """
    Maximal turbine density objective
    for purely geometrical layouts problems.

    :group: opt.problems.layout.geom_layouts.objectives

    """

    def __init__(self, problem, dfactor=1, min_dist=None, name="max_density"):
        """
        Constructor.

        Parameters
        ----------
        problem: foxes.opt.FarmOptProblem
            The underlying geometrical layout
            optimization problem
        dfactor: float
            Delta factor for grid spacing
        min_dist: float, optional
            The minimal distance
        name: str
            The constraint name

        """
        super().__init__(
            problem,
            name,
            vnames_int=problem.var_names_int(),
            vnames_float=problem.var_names_float(),
        )
        self.dfactor = dfactor
        self.min_dist = problem.min_dist if min_dist is None else min_dist

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
        return [False]

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
        detlta = self.min_dist / self.dfactor
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
        return np.nanmax(np.nanmin(dists, axis=1))

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
                out[pi] = np.nanmax(np.nanmin(dists, axis=1))
        return out[:, None]


class MeMiMaDist(Objective):
    """
    Mean-min-max distance objective
    for purely geometrical layouts problems.

    :group: opt.problems.layout.geom_layouts.objectives

    """

    def __init__(self, problem, scale=500.0, c1=1, c2=1, c3=1, name="MiMaMean"):
        """
        Constructor.

        Parameters
        ----------
        problem: foxes.opt.FarmOptProblem
            The underlying geometrical layout
            optimization problem
        scale: float
            The distance scale
        c1: float
            Parameter for mean weighting
        c2: float
            Parameter for max diff weighting
        c3: float
            Parameter for min diff weighting
        name: str
            The constraint name

        """
        super().__init__(
            problem,
            name,
            vnames_int=problem.var_names_int(),
            vnames_float=problem.var_names_float(),
        )
        self.scale = scale
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3

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
        # xy = xy[valid]

        dists = cdist(xy, xy)
        np.fill_diagonal(dists, np.inf)
        dists = np.min(dists, axis=1) / self.scale / len(xy)

        mean = np.average(dists)
        mi = np.min(dists)
        ma = np.max(dists)
        return np.atleast_1d(
            self.c1 * mean**2 - self.c2 * (mean - mi) ** 2 - self.c3 * (mean - ma) ** 2
        )

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
        xy, valid = problem_results
        n_pop, n_xy = xy.shape[:2]

        out = np.zeros((n_pop, 1), dtype=FC.DTYPE)
        for pi in range(n_pop):
            hxy = xy[pi]  # , valid[pi]]

            dists = cdist(hxy, hxy)
            np.fill_diagonal(dists, np.inf)
            dists = np.min(dists, axis=1) / self.scale / n_xy

            mean = np.average(dists)
            mi = np.min(dists)
            ma = np.max(dists)
            out[pi, 0] = (
                self.c1 * mean**2
                - self.c2 * (mean - mi) ** 2
                - self.c3 * (mean - ma) ** 2
            )

        return out
