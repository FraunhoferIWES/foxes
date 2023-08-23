import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from iwopy import Problem

import foxes.constants as FC


class GeomLayoutGridded(Problem):
    """
    A layout within a boundary geometry, purely
    defined by geometrical optimization (no wakes),
    on a fixes background point grid.

    This optimization problem does not involve
    wind farms.

    Attributes
    ----------
    boundary: foxes.utils.geom2d.AreaGeometry
        The boundary geometry
    n_turbines: int
        The number of turbines in the layout
    grid_spacing: float
        The background grid spacing
    min_dist: float
        The minimal distance between points
    D: float
        The diameter of circle fully within boundary

    :group: opt.problems.layout.geom_layouts

    """

    def __init__(
        self,
        boundary,
        n_turbines,
        grid_spacing,
        min_dist=None,
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
        grid_spacing: float
            The background grid spacing
        min_dist: float, optional
            The minimal distance between points
        D: float, optional
            The diameter of circle fully within boundary

        """
        super().__init__(name="geom_reg_grids")

        self.boundary = boundary
        self.n_turbines = n_turbines
        self.grid_spacing = grid_spacing
        self.D = D
        self.min_dist = min_dist

        self._I = [f"i{i}" for i in range(self.n_turbines)]

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
        pmax = self.boundary.p_max() + self.grid_spacing
        self._pts = np.stack(
            np.meshgrid(
                np.arange(pmin[0], pmax[0], self.grid_spacing),
                np.arange(pmin[1], pmax[1], self.grid_spacing),
                indexing="ij",
            ),
            axis=-1,
        )
        nx, ny = self._pts.shape[:2]
        self._pts = self._pts.reshape(nx * ny, 2)

        if self.D is None:
            valid = self.boundary.points_inside(self._pts)
        else:
            valid = self.boundary.points_inside(self._pts) & (
                self.boundary.points_distance(self._pts) >= self.D / 2
            )
        self._pts = self._pts[valid]
        self._N = len(self._pts)

        if verbosity > 0:
            print(f"Problem '{self.name}': n_bgd_pts = {self._N}")

        if self._N < self.n_turbines:
            raise ValueError(
                f"Problem '{self.name}': Background grid only provides {self._N} points for {self.n_turbines} turbines"
            )

        self.apply_individual(self.initial_values_int(), self.initial_values_float())

    def var_names_int(self):
        """
        The names of int variables.

        Returns
        -------
        names: list of str
            The names of the int variables

        """
        return self._I

    def initial_values_int(self):
        """
        The initial values of the int variables.

        Returns
        -------
        values: numpy.ndarray
            Initial int values, shape: (n_vars_int,)

        """
        return np.arange(self.n_turbines, dtype=FC.ITYPE)

    def min_values_int(self):
        """
        The minimal values of the int variables.

        Returns
        -------
        values: numpy.ndarray
            Minimal int values, shape: (n_vars_int,)

        """
        return np.zeros(self.n_turbines, dtype=FC.ITYPE)

    def max_values_int(self):
        """
        The maximal values of the int variables.

        Returns
        -------
        values: numpy.ndarray
            Maximal int values, shape: (n_vars_int,)

        """
        return np.full(self.n_turbines, self._N - 1, dtype=FC.ITYPE)

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
        xy = self._pts[vars_int.astype(FC.ITYPE)]
        __, ui = np.unique(vars_int, return_index=True)
        valid = np.zeros(self.n_turbines, dtype=bool)
        valid[ui] = True
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
        n_pop = vars_int.shape[0]

        vint = vars_int.reshape(n_pop * self.n_turbines).astype(FC.ITYPE)
        xy = self._pts[vint, :].reshape(n_pop, self.n_turbines, 2)

        valid = np.zeros((n_pop, self.n_turbines), dtype=bool)
        for pi in range(n_pop):
            __, ui = np.unique(vars_int[pi], return_index=True)
            valid[pi, ui] = True

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
