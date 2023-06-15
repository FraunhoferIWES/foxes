import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from iwopy import Problem

import foxes.constants as FC


class GeomRegGrids(Problem):
    """
    A regular grid within a boundary geometry.

    This optimization problem does not involve
    wind farms.

    Attributes
    ----------
    boundary: foxes.utils.geom2d.AreaGeometry
        The boundary geometry
    min_dist: float
        The minimal distance between points
    n_grids: int
        The number of grids
    n_max: int
        The maximal number of points
    n_row_max: int
        The maximal number of points in a row
    max_dist: float
        The maximal distance between points
    D: float
        The diameter of circle fully within boundary

    :group: opt.problems.layout.geom_layouts

    """

    def __init__(
        self,
        boundary,
        min_dist,
        n_grids,
        n_max=None,
        n_row_max=None,
        max_dist=None,
        D=None,
    ):
        """
        Constructor.

        Parameters
        ----------
        boundary: foxes.utils.geom2d.AreaGeometry
            The boundary geometry
        min_dist: float
            The minimal distance between points
        n_grids: int
            The number of grids
        n_max: int, optional
            The maximal number of points
        n_row_max: int, optional
            The maximal number of points in a row
        max_dist: float, optional
            The maximal distance between points
        D: float, optional
            The diameter of circle fully within boundary

        """
        super().__init__(name="geom_reg_grids")

        self.boundary = boundary
        self.n_grids = n_grids
        self.n_max = n_max
        self.n_row_max = n_row_max
        self.min_dist = float(min_dist)
        self.max_dist = float(max_dist) if max_dist is not None else max_dist
        self.D = D

        self._NX = [f"nx{i}" for i in range(self.n_grids)]
        self._NY = [f"ny{i}" for i in range(self.n_grids)]
        self._OX = [f"ox{i}" for i in range(self.n_grids)]
        self._OY = [f"oy{i}" for i in range(self.n_grids)]
        self._DX = [f"dx{i}" for i in range(self.n_grids)]
        self._DY = [f"dy{i}" for i in range(self.n_grids)]
        self._ALPHA = [f"alpha{i}" for i in range(self.n_grids)]

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
        self._span = pmax - pmin
        self._diag = np.linalg.norm(self._span)
        self.max_dist = self._diag if self.max_dist is None else self.max_dist
        self._nrow = self.n_row_max
        if self.n_row_max is None:
            if self.n_max is None:
                self._nrow = int(self._diag / self.min_dist)
                if self._nrow * self.min_dist < self._diag:
                    self._nrow += 1
                self._nrow += 1
            else:
                self._nrow = self.n_max
        if self.n_max is None:
            self.n_max = self.n_grids * self._nrow**2
        elif self.n_max <= self._nrow:
            self._nrow = self.n_max
        self._pmin = pmin - self._diag - self.min_dist
        self._pmax = pmax + self.min_dist

        if verbosity > 0:
            print(f"Grid data:")
            print(f"  pmin        = {self._pmin}")
            print(f"  pmax        = {self._pmax}")
            print(f"  min dist    = {self.min_dist}")
            print(f"  max dist    = {self.max_dist}")
            print(f"  n row max   = {self._nrow}")
            print(f"  n max       = {self.n_max}")
            print(f"  n grids     = {self.n_grids}")

        self.apply_individual(self.initial_values_int(), self.initial_values_float())

    def var_names_int(self):
        """
        The names of int variables.

        Returns
        -------
        names: list of str
            The names of the int variables

        """
        return list(np.array([self._NX, self._NY]).T.flat)

    def initial_values_int(self):
        """
        The initial values of the int variables.

        Returns
        -------
        values: numpy.ndarray
            Initial int values, shape: (n_vars_int,)

        """
        return np.full(self.n_grids * 2, 2, dtype=FC.ITYPE)

    def min_values_int(self):
        """
        The minimal values of the integer variables.

        Use -self.INT_INF for unbounded.

        Returns
        -------
        values: numpy.ndarray
            Minimal int values, shape: (n_vars_int,)

        """
        return np.ones(self.n_grids * 2, dtype=FC.ITYPE)

    def max_values_int(self):
        """
        The maximal values of the integer variables.

        Use self.INT_INF for unbounded.

        Returns
        -------
        values: numpy.ndarray
            Maximal int values, shape: (n_vars_int,)

        """
        return np.full(self.n_grids * 2, self._nrow, dtype=FC.ITYPE)

    def var_names_float(self):
        """
        The names of float variables.

        Returns
        -------
        names: list of str
            The names of the float variables

        """
        return list(
            np.array([self._OX, self._OY, self._DX, self._DY, self._ALPHA]).T.flat
        )

    def initial_values_float(self):
        """
        The initial values of the float variables.

        Returns
        -------
        values: numpy.ndarray
            Initial float values, shape: (n_vars_float,)

        """
        n = 5
        vals = np.zeros((self.n_grids, n), dtype=FC.DTYPE)
        vals[:, :2] = self._pmin + self._diag + self.min_dist + self._span / 2
        vals[:, 2:4] = 2 * self.min_dist
        vals[:, 5:] = 0
        return vals.reshape(self.n_grids * n)

    def min_values_float(self):
        """
        The minimal values of the float variables.

        Use -numpy.inf for unbounded.

        Returns
        -------
        values: numpy.ndarray
            Minimal float values, shape: (n_vars_float,)

        """
        n = 5
        vals = np.zeros((self.n_grids, n), dtype=FC.DTYPE)
        vals[:, :2] = self._pmin
        vals[:, 2:4] = self.min_dist
        vals[:, 5:] = -self._diag / 3
        return vals.reshape(self.n_grids * n)

    def max_values_float(self):
        """
        The maximal values of the float variables.

        Use numpy.inf for unbounded.

        Returns
        -------
        values: numpy.ndarray
            Maximal float values, shape: (n_vars_float,)

        """
        n = 5
        vals = np.zeros((self.n_grids, n), dtype=FC.DTYPE)
        vals[:, :2] = self._pmax
        vals[:, 2:4] = self.max_dist
        vals[:, 4] = 90.0
        vals[:, 5:] = self._diag / 3
        return vals.reshape(self.n_grids * n)

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
        vint = vars_int.reshape(self.n_grids, 2)
        vflt = vars_float.reshape(self.n_grids, 5)
        nx = vint[:, 0]
        ny = vint[:, 1]
        ox = vflt[:, 0]
        oy = vflt[:, 1]
        dx = vflt[:, 2]
        dy = vflt[:, 3]
        a = np.deg2rad(vflt[:, 4])
        s = vflt[:, 5:]
        n_points = self.n_max

        nax = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], axis=-1)
        naz = np.zeros_like(nax)
        naz[:, 2] = 1
        nay = np.cross(naz, nax)

        valid = np.zeros(n_points, dtype=bool)
        pts = np.full((n_points, 2), np.nan, dtype=FC.DTYPE)
        n0 = 0
        for gi in range(self.n_grids):
            n = nx[gi] * ny[gi]
            n1 = n0 + n

            if n1 <= n_points:
                qts = pts[n0:n1].reshape(nx[gi], ny[gi], 2)
            else:
                qts = np.zeros((nx[gi], ny[gi], 2), dtype=FC.DTYPE)

            qts[:, :, 0] = ox[gi]
            qts[:, :, 1] = oy[gi]
            qts[:] += (
                np.arange(nx[gi])[:, None, None] * dx[gi] * nax[gi, None, None, :2]
            )
            qts[:] += (
                np.arange(ny[gi])[None, :, None] * dy[gi] * nay[gi, None, None, :2]
            )
            qts = qts.reshape(n, 2)

            if n1 > n_points:
                n1 = n_points
                qts = qts[: (n1 - n0)]
                pts[n0:] = qts

            # set out of boundary points invalid:
            if self.D is None:
                valid[n0:n1] = self.boundary.points_inside(qts)
            else:
                valid[n0:n1] = self.boundary.points_inside(qts) & (
                    self.boundary.points_distance(qts) >= self.D / 2
                )

            # set points invalid which are too close to other grids:
            if n0 > 0:
                dists = cdist(qts, pts[:n0])
                valid[n0:n1][np.any(dists < self.min_dist, axis=1)] = False

            n0 = n1
            if n0 >= n_points:
                break

        return pts, valid

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
        n_pop = vars_int.shape[0]
        vint = vars_int.reshape(n_pop, self.n_grids, 2)
        vflt = vars_float.reshape(n_pop, self.n_grids, 5)
        nx = vint[:, :, 0]
        ny = vint[:, :, 1]
        ox = vflt[:, :, 0]
        oy = vflt[:, :, 1]
        dx = vflt[:, :, 2]
        dy = vflt[:, :, 3]
        a = np.deg2rad(vflt[:, :, 4])
        s = vflt[:, :, 5:]
        n_points = self.n_max

        nax = np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], axis=-1)
        naz = np.zeros_like(nax)
        naz[:, :, 2] = 1
        nay = np.cross(naz, nax)

        valid = np.zeros((n_pop, n_points), dtype=bool)
        pts = np.full((n_pop, n_points, 2), np.nan, dtype=FC.DTYPE)
        for pi in range(n_pop):
            n0 = 0
            for gi in range(self.n_grids):
                n = nx[pi, gi] * ny[pi, gi]
                n1 = n0 + n

                if n1 <= n_points:
                    qts = pts[pi, n0:n1].reshape(nx[pi, gi], ny[pi, gi], 2)
                else:
                    qts = np.zeros((nx[pi, gi], ny[pi, gi], 2), dtype=FC.DTYPE)

                qts[:, :, 0] = ox[pi, gi]
                qts[:, :, 1] = oy[pi, gi]
                qts[:] += (
                    np.arange(nx[pi, gi])[:, None, None]
                    * dx[pi, gi]
                    * nax[pi, gi, None, None, :2]
                )
                qts[:] += (
                    np.arange(ny[pi, gi])[None, :, None]
                    * dy[pi, gi]
                    * nay[pi, gi, None, None, :2]
                )
                qts = qts.reshape(n, 2)

                if n1 > n_points:
                    n1 = n_points
                    qts = qts[: (n1 - n0)]
                    pts[pi, n0:] = qts

                # set out of boundary points invalid:
                if self.D is None:
                    valid[pi, n0:n1] = self.boundary.points_inside(qts)
                else:
                    valid[pi, n0:n1] = self.boundary.points_inside(qts) & (
                        self.boundary.points_distance(qts) >= self.D / 2
                    )

                # set points invalid which are too close to other grids:
                if n0 > 0:
                    dists = cdist(qts, pts[pi, :n0])
                    valid[pi, n0:n1][np.any(dists < self.min_dist, axis=1)] = False

                n0 = n1

                if n0 >= n_points:
                    break

        return pts, valid

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
