import numpy as np

from foxes.opt.core.farm_constraint import FarmConstraint
import foxes.variables as FV
import foxes.constants as FC


class MinDistConstraint(FarmConstraint):
    """
    Turbines must keep at least a minimal
    spatial distance.

    Attributes
    ----------
    farm: foxes.WindFarm
        The wind farm
    sel_turbines: list
        The selected turbines
    min_dist: float
        The minimal distance
    min_dist_unit: str
        The minimal distance unit, either m or D

    :group: opt.constraints

    """

    def __init__(
        self,
        problem,
        min_dist,
        min_dist_unit="m",
        name="dist",
        sel_turbines=None,
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        problem: foxes.opt.FarmOptProblem
            The underlying optimization problem
        min_dist: float
            The minimal distance
        min_dist_unit: str
            The minimal distance unit, either m or D
        name: str
            The name of the constraint
        sel_turbines: list of int, optional
            The selected turbines
        kwargs: dict, optional
            Additional parameters for `iwopy.Constraint`

        """
        self.min_dist = min_dist
        self.min_dist_unit = min_dist_unit

        selt = problem.sel_turbines if sel_turbines is None else sel_turbines
        vrs = []
        for ti in selt:
            vrs += [problem.tvar(FV.X, ti), problem.tvar(FV.Y, ti)]

        super().__init__(problem, name, sel_turbines, vnames_float=vrs, **kwargs)

    def initialize(self, verbosity=0):
        """
        Initialize the constaint.

        Parameters
        ----------
        verbosity: int
            The verbosity level, 0 = silent

        """
        N = self.farm.n_turbines
        self._i2t = []  # i --> (ti, tj)
        self._t2i = np.full([N, N], -1)  # (ti, tj) --> i
        i = 0
        for ti in self.sel_turbines:
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

    def vardeps_float(self):
        """
        Gets the dependencies of all components
        on the function float variables

        Returns
        -------
        deps: numpy.ndarray of bool
            The dependencies of components on function
            variables, shape: (n_components, n_vars_float)

        """
        turbs = list(self.problem.sel_turbines)
        deps = np.zeros((self.n_components(), len(turbs), 2), dtype=bool)
        for i, titj in enumerate(self._i2t):
            for t in titj:
                if t in turbs:
                    j = turbs.index(t)
                    deps[i, j] = True
        return deps.reshape(self.n_components(), 2 * len(turbs))

    def calc_individual(self, vars_int, vars_float, problem_results, components=None):
        """
        Calculate values for a single individual of the
        underlying problem.

        Parameters
        ----------
        vars_int: np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float: np.array
            The float variable values, shape: (n_vars_float,)
        problem_results: Any
            The results of the variable application
            to the problem
        components: list of int, optional
            The selected components or None for all

        Returns
        -------
        values: np.array
            The component values, shape: (n_sel_components,)

        """
        xy = np.stack(
            [problem_results[FV.X].to_numpy(), problem_results[FV.Y].to_numpy()],
            axis=-1,
        )
        if not np.all(np.abs(np.min(xy, axis=0) - np.max(xy, axis=0)) < 1e-13):
            raise ValueError(f"Constraint '{self.name}': Require state independet XY")
        xy = xy[0]

        s = np.s_[:]
        if components is not None and len(components) < self.n_components():
            s = components

        a = np.take_along_axis(xy, self._i2t[s][:, 0, None], axis=0)
        b = np.take_along_axis(xy, self._i2t[s][:, 1, None], axis=0)
        d = np.linalg.norm(a - b, axis=-1)

        if self.min_dist_unit == "m":
            mind = self.min_dist

        elif self.min_dist_unit == "D":
            D = problem_results[FV.D].to_numpy()
            if not np.all(np.abs(np.min(D, axis=0) - np.max(D, axis=0)) < 1e-13):
                raise ValueError(
                    f"Constraint '{self.name}': Require state independet D"
                )
            D = D[0]

            Da = np.take_along_axis(D, self._i2t[s][:, 0], axis=0)
            Db = np.take_along_axis(D, self._i2t[s][:, 1], axis=0)
            mind = self.min_dist * np.maximum(Da, Db)

        return mind - d

    def calc_population(self, vars_int, vars_float, problem_results, components=None):
        """
        Calculate values for all individuals of a population.

        Parameters
        ----------
        vars_int: np.array
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float: np.array
            The float variable values, shape: (n_pop, n_vars_float)
        problem_results: Any
            The results of the variable application
            to the problem
        components: list of int, optional
            The selected components or None for all

        Returns
        -------
        values: np.array
            The component values, shape: (n_pop, n_sel_components)

        """
        n_pop = problem_results["n_pop"].values
        n_states = problem_results["n_org_states"].values
        n_turbines = problem_results.sizes[FC.TURBINE]

        xy = np.stack(
            [problem_results[FV.X].to_numpy(), problem_results[FV.Y].to_numpy()],
            axis=-1,
        )
        xy = xy.reshape(n_pop, n_states, n_turbines, 2)
        if not np.all(np.abs(np.min(xy, axis=1) - np.max(xy, axis=1)) < 1e-13):
            raise ValueError(f"Constraint '{self.name}': Require state independet XY")
        xy = xy[:, 0]

        s = np.s_[:]
        if components is not None and len(components) < self.n_components():
            s = components

        a = np.take_along_axis(xy, self._i2t[s][None, :, 0, None], axis=1)
        b = np.take_along_axis(xy, self._i2t[s][None, :, 1, None], axis=1)
        d = np.linalg.norm(a - b, axis=-1)

        if self.min_dist_unit == "m":
            mind = self.min_dist

        elif self.min_dist_unit == "D":
            D = problem_results[FV.D].to_numpy().reshape(n_pop, n_states, n_turbines)
            if not np.all(np.abs(np.min(D, axis=1) - np.max(D, axis=1)) < 1e-13):
                raise ValueError(
                    f"Constraint '{self.name}': Require state independet D"
                )
            D = D[:, 0]

            Da = np.take_along_axis(D, self._i2t[s][None, :, 0], axis=1)
            Db = np.take_along_axis(D, self._i2t[s][None, :, 1], axis=1)
            mind = self.min_dist * np.maximum(Da, Db)

        return mind - d
