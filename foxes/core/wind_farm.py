class WindFarm:
    """
    The wind farm.

    Parameters
    ----------
    name : str
        The wind farm name

    Attributes
    ----------
    name : str
        The wind farm name
    turbines : list of foxes.core.Turbine
        The wind turbines
    boundary : foxes.utils.geom2d.AreaGeometry, optional
        The wind farm boundary

    """

    def __init__(self, name="wind_farm", boundary=None):
        self.name = name
        self.turbines = []
        self.boundary = boundary

    def add_turbine(self, turbine, verbosity=1):
        """
        Add a wind turbine to the list.

        Parameters
        ----------
        turbine : foxes.core.Turbine
            The wind turbine
        verbosity : int
            The output verbosity, 0 = silent

        """
        if turbine.index is None:
            turbine.index = len(self.turbines)
        if turbine.name is None:
            turbine.name = f"T{turbine.index}"
        self.turbines.append(turbine)
        if verbosity > 0:
            print(
                f"Turbine {turbine.index}, {turbine.name}: {', '.join(turbine.models)}"
            )

    @property
    def n_turbines(self):
        """
        The number of turbines in the wind farm

        Returns
        -------
        n_turbines : int
            The total number of turbines

        """
        return len(self.turbines)
