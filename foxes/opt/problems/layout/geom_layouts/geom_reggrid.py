import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from iwopy import Problem

import foxes.constants as FC


class GeomRegGrid(Problem):
    """
    A regular grid within a boundary geometry.

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
    max_dist: float
        The maximal distance between points
    D: float
        The diameter of circle fully within boundary

    :group: opt.problems.layout.geom_layouts

    """

    def __init__(
        self,
        boundary,
        n_turbines,
        min_dist,
        max_dist=None,
        D=None,
    ):
        """
        Constructor.

        Parameters
        ----------
        boundary: foxes.utils.geom2d.AreaGeometry
            The boundary geometry
        n_turbines: int
            The number of turbines in the layout
        min_dist: float
            The minimal distance between points
        max_dist: float, optional
            The maximal distance between points
        D: float, optional
            The diameter of circle fully within boundary

        """
        super().__init__(name="geom_reg_grid")

        self.boundary = boundary
        self.n_turbines = n_turbines
        self.min_dist = float(min_dist)
        self.max_dist = float(max_dist) if max_dist is not None else max_dist
        self.D = D

        self._SX = "sx"
        self._SY = "sy"
        self._DX = "dx"
        self._DY = "dy"
        self._ALPHA = "alpha"

    def initialize(self, verbosity=1):
        """
        Initialize the object.

        Parameters
        ----------
        verbosity: int
            The verbosity level, 0 = silent

        """
        super().initialize(verbosity)

        pmin = self.boundary.p_min()
        pmax = self.boundary.p_max()
        self._pc = 0.5 * (pmin + pmax)
        self._diag = np.linalg.norm(pmax - pmin)
        self.max_dist = self._diag if self.max_dist is None else self.max_dist
        self._nrow = (
            int(np.maximum(self._diag / self.min_dist, np.sqrt(self.n_turbines) + 0.5))
            + 3
        )

        if verbosity > 0:
            print(f"Grid data:")
            print(f"  pmin        = {pmin}")
            print(f"  pmax        = {pmax}")
            print(f"  min dist    = {self.min_dist}")
            print(f"  max dist    = {self.max_dist}")
            print(f"  n row max   = {self._nrow}")
            print(f"  n max       = {self._nrow**2}")

        self.apply_individual(self.initial_values_int(), self.initial_values_float())

    def var_names_float(self):
        """
        The names of float variables.

        Returns
        -------
        names: list of str
            The names of the float variables

        """
        return list(np.array([self._SX, self._SY, self._DX, self._DY, self._ALPHA]))

    def initial_values_float(self):
        """
        The initial values of the float variables.

        Returns
        -------
        values: numpy.ndarray
            Initial float values, shape: (n_vars_float,)

        """
        vals = np.zeros(5, dtype=FC.DTYPE)
        vals[2:4] = self.min_dist
        return vals

    def min_values_float(self):
        """
        The minimal values of the float variables.

        Use -numpy.inf for unbounded.

        Returns
        -------
        values: numpy.ndarray
            Minimal float values, shape: (n_vars_float,)

        """
        vals = np.zeros(5, dtype=FC.DTYPE)
        vals[:2] = -0.5
        vals[2:4] = self.min_dist
        return vals

    def max_values_float(self):
        """
        The maximal values of the float variables.

        Use numpy.inf for unbounded.

        Returns
        -------
        values: numpy.ndarray
            Maximal float values, shape: (n_vars_float,)

        """
        vals = np.zeros(5, dtype=FC.DTYPE)
        vals[:2] = 0.5
        vals[2:4] = self.max_dist
        vals[4] = 90.0
        return vals

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
        sx, sy, dx, dy, alpha = vars_float

        a = np.deg2rad(alpha)
        nax = np.stack([np.cos(a), np.sin(a)], axis=-1)
        nay = np.stack([-np.sin(a), np.cos(a)], axis=-1)

        pts = (
            self._pc[None, None, :]
            + (np.arange(self._nrow)[:, None, None] - (self._nrow - 1) / 2 + sx)
            * dx
            * nax[None, None, :]
            + (np.arange(self._nrow)[None, :, None] - (self._nrow - 1) / 2 + sy)
            * dy
            * nay[None, None, :]
        )
        pts = pts.reshape(self._nrow**2, 2)

        if self.D is None:
            valid = self.boundary.points_inside(pts)
        else:
            valid = self.boundary.points_inside(pts) & (
                self.boundary.points_distance(pts) >= self.D / 2
            )

        nvl = np.sum(valid)
        if nvl >= self.n_turbines:
            return pts[valid][: self.n_turbines], np.ones(self.n_turbines, dtype=bool)
        else:
            qts = np.append(pts[valid], pts[~valid][: (self.n_turbines - nvl)], axis=0)
            vld = np.zeros(self.n_turbines, dtype=bool)
            vld[:nvl] = True
            return qts, vld

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
        sx = vars_float[:, 0]
        sy = vars_float[:, 1]
        dx = vars_float[:, 2]
        dy = vars_float[:, 3]
        alpha = vars_float[:, 4]

        a = np.deg2rad(alpha)
        nax = np.stack([np.cos(a), np.sin(a)], axis=-1)
        nay = np.stack([-np.sin(a), np.cos(a)], axis=-1)

        pts = (
            self._pc[None, None, None, :]
            + (
                np.arange(self._nrow)[None, :, None, None]
                - (self._nrow - 1) / 2
                + sx[:, None, None, None]
            )
            * dx[:, None, None, None]
            * nax[:, None, None, :]
            + (
                np.arange(self._nrow)[None, None, :, None]
                - (self._nrow - 1) / 2
                + sy[:, None, None, None]
            )
            * dy[:, None, None, None]
            * nay[:, None, None, :]
        )
        pts = pts.reshape(n_pop * self._nrow**2, 2)

        if self.D is None:
            valid = self.boundary.points_inside(pts)
        else:
            valid = self.boundary.points_inside(pts) & (
                self.boundary.points_distance(pts) >= self.D / 2
            )
        valid = valid.reshape(n_pop, self._nrow**2)
        pts = pts.reshape(n_pop, self._nrow**2, 2)

        nvl = np.sum(valid, axis=1)
        qts = np.zeros((n_pop, self.n_turbines, 2), dtype=FC.DTYPE)
        vld = np.zeros((n_pop, self.n_turbines), dtype=bool)
        for pi in range(n_pop):
            if nvl[pi] >= self.n_turbines:
                qts[pi] = pts[pi, valid[pi]][: self.n_turbines]
                vld[pi] = np.ones(self.n_turbines, dtype=bool)
            else:
                qts[pi] = np.append(
                    pts[pi, valid[pi]],
                    pts[pi, ~valid[pi]][: (self.n_turbines - nvl[pi])],
                    axis=0,
                )
                vld[pi, : nvl[pi]] = True

        return qts, vld

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
