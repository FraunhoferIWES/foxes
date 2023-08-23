import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from iwopy import Problem

import foxes.constants as FC


class GeomLayout(Problem):
    """
    A layout within a boundary geometry, purely
    defined by geometrical optimization (no wakes).

    This optimization problem does not involve
    wind farms.

    Attributes
    ----------
    boundary: foxes.utils.geom2d.AreaGeometry
        The boundary geometry
    n_turbines: int
        The number of turbines in the layout
    min_dist: float
        The minimal distance between points
    D: float
        The diameter of circle fully within boundary
    calc_valid: bool
        Evaluate validity

    :group: opt.problems.layout.geom_layouts

    """

    def __init__(
        self,
        boundary,
        n_turbines,
        min_dist=None,
        D=None,
        calc_valid=None,
    ):
        """
        Constructor.

        Parameters
        ----------
        boundary: foxes.utils.geom2d.AreaGeometry
            The boundary geometry
        n_turbines: int
            The number of turbines in the layout
        min_dist: float, optional
            The minimal distance between points
        D: float, optional
            The diameter of circle fully within boundary
        calc_valid: bool, optional
            Evaluate validity

        """
        super().__init__(name="geom_reg_grids")

        self.boundary = boundary
        self.n_turbines = n_turbines
        self.D = D
        self.min_dist = min_dist
        self.calc_valid = calc_valid
        if calc_valid is None:
            self.calc_valid = min_dist is not None or D is not None

        self._X = [f"x{i}" for i in range(self.n_turbines)]
        self._Y = [f"y{i}" for i in range(self.n_turbines)]

    def initialize(self, verbosity=1):
        """
        Initialize the object.

        Parameters
        ----------
        verbosity: int
            The verbosity level, 0 = silent

        """
        super().initialize(verbosity)
        self.apply_individual(self.initial_values_int(), self.initial_values_float())

    def var_names_float(self):
        """
        The names of float variables.

        Returns
        -------
        names: list of str
            The names of the float variables

        """
        return list(np.array([self._X, self._Y]).T.flat)

    def initial_values_float(self):
        """
        The initial values of the float variables.

        Returns
        -------
        values: numpy.ndarray
            Initial float values, shape: (n_vars_float,)

        """
        pmin = self.boundary.p_min()
        pmax = self.boundary.p_max()
        pc = 0.5 * (pmin + pmax)
        delta = 0.8 * (pmax - pmin)

        vals = np.zeros((self.n_turbines, 2), dtype=FC.DTYPE)
        vals[:] = pc[None, :] - 0.5 * delta[None, :]
        vals[:] += (
            np.arange(self.n_turbines)[:, None] * delta[None, :] / (self.n_turbines - 1)
        )

        return vals.reshape(self.n_turbines * 2)

    def min_values_float(self):
        """
        The minimal values of the float variables.

        Use -numpy.inf for unbounded.

        Returns
        -------
        values: numpy.ndarray
            Minimal float values, shape: (n_vars_float,)

        """
        vals = np.zeros((self.n_turbines, 2), dtype=FC.DTYPE)
        vals[:] = self.boundary.p_min()[None, :]
        return vals.reshape(self.n_turbines * 2)

    def max_values_float(self):
        """
        The maximal values of the float variables.

        Use numpy.inf for unbounded.

        Returns
        -------
        values: numpy.ndarray
            Maximal float values, shape: (n_vars_float,)

        """
        vals = np.zeros((self.n_turbines, 2), dtype=FC.DTYPE)
        vals[:] = self.boundary.p_max()[None, :]
        return vals.reshape(self.n_turbines * 2)

    def apply_individual(self, vars_int, vars_float):
        """
        Apply new variables to the problem.

        Parameters
        ----------
        vars_int: np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float: np.array
            The float variable values, shape: (n_vars_float,)

        Returns
        -------
        problem_results: Any
            The results of the variable application
            to the problem

        """
        xy = vars_float.reshape(self.n_turbines, 2)

        valid = None
        if self.calc_valid:
            if self.D is None:
                valid = self.boundary.points_inside(xy)
            else:
                valid = self.boundary.points_inside(xy) & (
                    self.boundary.points_distance(xy) >= self.D / 2
                )

            if self.min_dist is not None:
                dists = cdist(xy, xy)
                np.fill_diagonal(dists, 1e20)
                dists = np.min(dists, axis=1)
                valid[dists < self.min_dist] = False

        return xy, valid

    def apply_population(self, vars_int, vars_float):
        """
        Apply new variables to the problem,
        for a whole population.

        Parameters
        ----------
        vars_int: np.array
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float: np.array
            The float variable values, shape: (n_pop, n_vars_float)

        Returns
        -------
        problem_results: Any
            The results of the variable application
            to the problem

        """
        n_pop = vars_float.shape[0]
        xy = vars_float.reshape(n_pop, self.n_turbines, 2)

        valid = None
        if self.calc_valid:
            qts = xy.reshape(n_pop * self.n_turbines, 2)
            if self.D is None:
                valid = self.boundary.points_inside(qts)
            else:
                valid = self.boundary.points_inside(qts) & (
                    self.boundary.points_distance(qts) >= self.D / 2
                )
            valid = valid.reshape(n_pop, self.n_turbines)

            if self.min_dist is not None:
                for pi in range(n_pop):
                    dists = cdist(xy[pi], xy[pi])
                    np.fill_diagonal(dists, 1e20)
                    dists = np.min(dists, axis=1)
                    valid[pi, dists < self.min_dist] = False

        return xy, valid

    def get_fig(
        self, xy=None, valid=None, ax=None, title=None, true_circle=True, **bargs
    ):
        """
        Return plotly figure axis.

        Parameters
        ----------
        xy: numpy.ndarary, optional
            The xy coordinate array, shape: (n_points, 2)
        valid: numpy.ndarray, optional
            Boolean array of validity, shape: (n_points,)
        ax: pyplot.Axis, optional
            The figure axis
        title: str, optional
            The figure title
        true_circle: bool
            Draw points as circles with diameter self.D
        bars: dict, optional
            The boundary plot arguments

        Returns
        -------
        ax: pyplot.Axis
            The figure axis

        """
        if ax is None:
            __, ax = plt.subplots()

        hbargs = {"fill_mode": "inside_lightgray"}
        hbargs.update(bargs)
        self.boundary.add_to_figure(ax, **hbargs)

        if xy is not None:
            if valid is not None:
                xy = xy[valid]
            if not true_circle or self.D is None:
                ax.scatter(xy[:, 0], xy[:, 1], color="orange")
            else:
                for x, y in xy:
                    ax.add_patch(
                        plt.Circle((x, y), self.D / 2, color="blue", fill=True)
                    )

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        if title is None:
            if xy is None:
                title = f"Optimization area"
            else:
                l = len(xy) if xy is not None else 0
                dists = cdist(xy, xy)
                np.fill_diagonal(dists, 1e20)
                title = f"N = {l}, min_dist = {np.min(dists):.1f} m"
        ax.set_title(title)

        return ax
