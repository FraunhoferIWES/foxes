import numpy as np

from foxes.opt.core.farm_constraint import FarmConstraint
import foxes.variables as FV


class AreaGeometryConstraint(FarmConstraint):
    """
    Constrains turbine positions to the inside
    of a given area geometry.

    Attributes
    ----------
    farm: foxes.WindFarm
        The wind farm
    sel_turbines: list
        The selected turbines
    geometry: foxes.utils.geom2d.AreaGeometry
        The area geometry
    disc_inside: bool
        Ensure full rotor disc inside boundary
    D: float
        Use this radius for rotor disc inside condition

    :group: opt.constraints

    """

    def __init__(
        self,
        problem,
        name,
        geometry,
        sel_turbines=None,
        disc_inside=False,
        D=None,
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        problem : foxes.opt.FarmOptProblem
            The underlying optimization problem
        name : str
            The name of the constraint
        geometry : foxes.utils.geom2d.AreaGeometry
            The area geometry
        sel_turbines : list of int, optional
            The selected turbines
        disc_inside : bool
            Ensure full rotor disc inside boundary
        D : float, optional
            Use this radius for rotor disc inside condition
        kwargs : dict, optional
            Additional parameters for `iwopy.Constraint`

        """
        self.geometry = geometry
        self.disc_inside = disc_inside
        self.D = D

        selt = problem.sel_turbines if sel_turbines is None else sel_turbines
        vrs = []
        cns = []
        for ti in selt:
            vrs += [problem.tvar(FV.X, ti), problem.tvar(FV.Y, ti)]
            cns.append(f"{name}_{ti:04d}")

        super().__init__(
            problem, name, sel_turbines, vnames_float=vrs, cnames=cns, **kwargs
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
        return self.n_sel_turbines

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
        deps = np.zeros((self.n_components(), self.n_components(), 2), dtype=bool)
        np.fill_diagonal(deps[:, :, 0], True)
        np.fill_diagonal(deps[:, :, 1], True)
        return deps.reshape(self.n_components(), self.n_components() * 2)

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
        s = np.s_[:]
        if components is not None and len(components) < self.n_components():
            s = components
        xy = vars_float.reshape(self.n_components(), 2)[s]

        dists = self.geometry.points_distance(xy)
        dists[self.geometry.points_inside(xy)] *= -1

        if self.disc_inside:
            if self.D is None:
                dists += problem_results[FV.D].to_numpy()[0, self.sel_turbines][s] / 2
            else:
                dists += self.D / 2

        return dists

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
        n_pop = len(vars_float)
        n_cmpnts = self.n_components()
        s = np.s_[:]
        if components is not None and len(components) < self.n_components():
            n_cmpnts = len(components)
            s = components
        xy = vars_float[:, s].reshape(n_pop * n_cmpnts, 2)

        dists = self.geometry.points_distance(xy)
        dists[self.geometry.points_inside(xy)] *= -1
        dists = dists.reshape(n_pop, n_cmpnts)

        if self.disc_inside:
            if self.D is None:
                dists += (
                    problem_results[FV.D].to_numpy()[None, 0, self.sel_turbines][s] / 2
                )
            else:
                dists += self.D / 2

        return dists


class FarmBoundaryConstraint(AreaGeometryConstraint):
    """
    Constrains turbine positions to the inside of
    the wind farm boundary

    :group: opt.constraints

    """

    def __init__(self, problem, name="boundary", **kwargs):
        """
        Constructor.

        Parameters
        ----------
        problem: foxes.opt.FarmOptProblem
            The underlying optimization problem
        name: str
            The name of the constraint
        kwargs: dict, optional
            Additional parameters for `AreaGeometryConstraint`

        """
        b = problem.farm.boundary
        assert b is not None, f"Constraint '{name}': Missing wind farm boundary."
        super().__init__(problem, name, geometry=b, **kwargs)
