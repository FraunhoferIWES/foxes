from .area_geometry import AreaGeometryConstraint

class FarmBoundaryConstraint(AreaGeometryConstraint):
    """
    Constrains turbine positions to the inside of
    the wind farm boundary

    Parameters
    ----------
    problem : foxes.opt.FarmOptProblem
        The underlying optimization problem
    name : str
        The name of the constraint
    kwargs : dict, optional
        Additional parameters for `AreaGeometryConstraint`

    """

    def __init__(
            self,
            problem, 
            name,
            **kwargs,
        ):
        b = problem.farm.boundary
        assert b is not None, f"Constraint '{name}': Missing boundary in wind farm."
        super().__init__(problem, name, geometry=b, **kwargs)
