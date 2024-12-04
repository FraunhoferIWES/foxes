import numpy as np

from foxes.config import config


class WindFarm:
    """
    The wind farm.

    Attributes
    ----------
    name: str
        The wind farm name
    turbines: list of foxes.core.Turbine
        The wind turbines
    boundary: foxes.utils.geom2d.AreaGeometry, optional
        The wind farm boundary

    :group: core

    """

    def __init__(self, name="wind_farm", boundary=None):
        """
        Constructor.

        Parameters
        ----------
        name: str
            The wind farm name

        """
        self.name = name
        self.turbines = []
        self.boundary = boundary

    def add_turbine(self, turbine, verbosity=1):
        """
        Add a wind turbine to the list.

        Parameters
        ----------
        turbine: foxes.core.Turbine
            The wind turbine
        verbosity: int
            The output verbosity, 0 = silent

        """
        if turbine.index is None:
            turbine.index = len(self.turbines)
        if turbine.name is None:
            turbine.name = f"T{turbine.index}"
        self.turbines.append(turbine)
        if verbosity > 0:
            print(
                f"Turbine {turbine.index}, {turbine.name}: xy=({turbine.xy[0]:.2f}, {turbine.xy[1]:.2f}), {', '.join(turbine.models)}"
            )

    @property
    def n_turbines(self):
        """
        The number of turbines in the wind farm

        Returns
        -------
        n_turbines: int
            The total number of turbines

        """
        return len(self.turbines)

    @property
    def turbine_names(self):
        """
        The list of names of all turbines

        Returns
        -------
        names: list of str
            The names of all turbines

        """
        return [t.name for t in self.turbines]

    @property
    def xy_array(self):
        """
        Returns an array of the wind farm ground points

        Returns
        -------
        xya: numpy.ndarray
            The turbine ground positions, shape: (n_turbines, 2)

        """
        return np.array([t.xy for t in self.turbines], dtype=config.dtype_double)
        
    def get_xy_bounds(self, extra_space=None, algo=None):
        """
        Returns min max points of the wind farm ground points

        Parameters
        ----------
        extra_space: float or str, optional
            The extra space, either float in m,
            or str for units of D, e.g. '2.5D'
        algo: foxes.core.Algorithm, optional
            The algorithm

        Returns
        -------
        x_mima: numpy.ndarray
            The (x_min, x_max) point
        y_mima: numpy.ndarray
            The (y_min, y_max) point

        """
        if self.boundary is not None:
            xy = None
            p_min, p_max = self.boundary.p_min(), self.boundary.p_max()
        else:
            xy = self.xy_array
            p_min, p_max = np.min(xy, axis=0), np.max(xy, axis=0)

        if extra_space is not None:
            if isinstance(extra_space, str):
                assert algo is not None, f"WindFarm: require algo argument for extra_space '{extra_space}'"
                extra_space = float(extra_space[:-1])
                rds = self.get_rotor_diameters(algo)
                if xy is None:
                    extra_space *= np.max(rds)
                else:
                    p_min = np.min(xy-extra_space*rds[:, None], axis=0)
                    p_max = np.max(xy+extra_space*rds[:, None], axis=0)
                    return p_min, p_max
                
            p_min -= extra_space
            p_max += extra_space

        return p_min, p_max

    def get_rotor_diameters(self, algo):
        """
        Gets the rotor diameters
        
        Parameters
        ----------
        algo: foxes.core.Algorithm
            The algorithm
        
        Returns
        -------
        rds: numpy.ndarray
            The rotor diameters, shape: (n_turbienes,)

        """
        rds = [
            t.D if t.D is not None 
            else algo.farm_controller.turbine_types[i].D
            for i, t in enumerate(self.turbines)
        ]
        return np.array(rds, dtype=config.dtype_double)

    def get_hub_heights(self, algo):
        """
        Gets the hub heights
        
        Parameters
        ----------
        algo: foxes.core.Algorithm
            The algorithm
        
        Returns
        -------
        hhs: numpy.ndarray
            The hub heights, shape: (n_turbienes,)

        """
        hhs = [
            t.H if t.H is not None 
            else algo.farm_controller.turbine_types[i].H
            for i, t in enumerate(self.turbines)
        ]
        return np.array(hhs, dtype=config.dtype_double)
    