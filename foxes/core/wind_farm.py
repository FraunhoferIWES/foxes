import numpy as np

from foxes.config import config
from foxes.utils import get_utm_zone, from_lonlat, to_lonlat


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

    def __init__(
        self,
        name="wind_farm",
        boundary=None,
        input_is_lonlat=False,
        utm_zone="from_farm",
    ):
        """
        Constructor.

        Parameters
        ----------
        name: str
            The wind farm name
        boundary: foxes.utils.geom2d.AreaGeometry, optional
            The wind farm boundary
        input_is_lonlat: bool, optional
            Whether the input coordinates are given in lon, lat. If True,
            the coordinates are converted to UTM as specified by the
            utm_zone parameter.
        utm_zone: str or tuple, optional
            Method for setting UTM zone in config, if not already set.
            Options are:
            - "from_turbine_X": use turbine X coordinates
            - "from_farm": use farm center coordinates
            - "XA": use given number X, letter A
            - (lon, lat): use given lon, lat values
            - None: do not set UTM zone, assume it is already set

        """
        self.name = name
        self.__turbines = []
        self.boundary = boundary

        self.__data_is_lonlat = input_is_lonlat
        self.__utm_zone = utm_zone
        self.__locked = False

    @property
    def data_is_lonlat(self):
        """
        Whether the input coordinates are given in lat, lon.

        Returns
        -------
        data_is_lonlat: bool
            True if the input coordinates are given in lat, lon

        """
        return self.__data_is_lonlat

    @property
    def locked(self):
        """
        Whether the wind farm is locked (no more turbines can be added)

        Returns
        -------
        locked: bool
            True if the wind farm is locked

        """
        return self.__locked

    @property
    def turbines(self):
        """
        The list of wind turbines

        Returns
        -------
        turbines: list of foxes.core.Turbine
            The wind turbines

        """
        if not self.__locked:
            self.__locked = True
            if self.__data_is_lonlat:
                if not config.utm_zone_set and self.__utm_zone is None:
                    raise ValueError(
                        f"WindFarm '{self.name}': input_is_lonlat is True, but config.utm_zone and utm_zone are None"
                    )
                if self.__utm_zone is None:
                    zone = config.utm_zone
                elif self.__utm_zone == "from_farm":
                    lonlat = np.mean([t.xy for t in self.__turbines], axis=0)
                    zone = get_utm_zone(lonlat[None, :])
                elif (
                    isinstance(self.__utm_zone, str)
                    and self.__utm_zone.startswith("from_turbine_")
                    and len(self.__utm_zone) > len("from_turbine_")
                ):
                    idx = int(self.__utm_zone[len("from_turbine_") :])
                    lonlat = self.__turbines[idx].xy
                    zone = get_utm_zone(lonlat[None, :])
                elif isinstance(self.__utm_zone, str):
                    zone = (int(self.__utm_zone[:-1]), self.__utm_zone[-1])
                elif len(self.__utm_zone) == 2:
                    lonlat = np.asarray(self.__utm_zone)
                    zone = get_utm_zone(lonlat[None, :])
                else:
                    raise ValueError(
                        f"WindFarm '{self.name}': invalid utm_zone argument: {self.__utm_zone}"
                    )
                if not config.utm_zone_set:
                    config.set_utm_zone(*zone)
                elif config.utm_zone != zone:
                    raise ValueError(
                        f"WindFarm '{self.name}': input_is_lonlat is True, but config.utm_zone = {config.utm_zone} differs from determined zone {zone}"
                    )
                for t in self.__turbines:
                    t.xy = from_lonlat(t.xy[None, :])[0]
                self.__data_is_lonlat = False
        return self.__turbines

    def lock(self, verbosity=1):
        """
        Lock the wind farm (no more turbines can be added)

        Parameters
        ----------
        verbosity: int
            The output verbosity, 0 = silent

        """
        self.turbines
        if verbosity > 0:
            if config.utm_zone_set:
                utmn, utml = config.utm_zone
                print(
                    f"WindFarm '{self.name}': locked with {self.n_turbines} turbines, UTM zone {utmn}{utml}"
                )
                if verbosity > 1:
                    for t in self.__turbines:
                        print(
                            f"  Turbine {t.index}, {t.name}: UTM {utmn}{utml}, xy=({t.xy[0]:.2f}, {t.xy[1]:.2f}), {', '.join(t.models)}"
                        )
            else:
                print(f"WindFarm '{self.name}': locked with {self.n_turbines} turbines")

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
        assert not self.__locked, (
            f"WindFarm '{self.name}': cannot add turbine, farm is locked"
        )
        if turbine.index is None:
            turbine.index = len(self.__turbines)
        if turbine.name is None:
            turbine.name = f"T{turbine.index}"
        self.__turbines.append(turbine)
        if verbosity > 0:
            if self.data_is_lonlat:
                print(
                    f"Turbine {turbine.index}, {turbine.name}: lonlat=({turbine.xy[0]:.6f}, {turbine.xy[1]:.6f}), {', '.join(turbine.models)}"
                )
            elif config.utm_zone_set:
                utmn, utml = config.utm_zone
                print(
                    f"Turbine {turbine.index}, {turbine.name}: UTM {utmn}{utml}, xy=({turbine.xy[0]:.2f}, {turbine.xy[1]:.2f}), {', '.join(turbine.models)}"
                )
            else:
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
        return len(self.__turbines)

    @property
    def turbine_names(self):
        """
        The list of names of all turbines

        Returns
        -------
        names: list of str
            The names of all turbines

        """
        return [t.name for t in self.__turbines]

    @property
    def xy_array(self):
        """
        Returns an array of the wind farm ground points

        Returns
        -------
        xya: numpy.ndarray
            The turbine ground positions, shape: (n_turbines, 2)

        """
        return np.array([t.xy for t in self.__turbines], dtype=config.dtype_double)

    @property
    def wind_farm_names(self):
        """
        The list of wind farm names for all turbines

        Returns
        -------
        names: list of str
            The wind farm names for all turbines

        """
        return list(
            set(
                [
                    t.wind_farm_name if t.wind_farm_name is not None else self.name
                    for t in self.__turbines
                ]
            )
        )

    def get_wind_farm_mapping(self):
        """
        Returns a mapping from wind farm names to turbine indices

        Returns
        -------
        mapping: dict
            A dictionary, where keys are wind farm names and
            values are lists of turbine indices belonging to that wind farm

        """
        mapping = {}
        for i, t in enumerate(self.__turbines):
            wf_name = t.wind_farm_name if t.wind_farm_name is not None else self.name
            if wf_name not in mapping:
                mapping[wf_name] = []
            mapping[wf_name].append(i)
        return mapping

    def get_xy_bounds(self, extra_space=None, algo=None, lonlat=False, sample_dx=10.0):
        """
        Returns min max points of the wind farm ground points

        Parameters
        ----------
        extra_space: float or str, optional
            The extra space, either float in m,
            or str for units of D, e.g. '2.5D'
        algo: foxes.core.Algorithm, optional
            The algorithm
        lonlat: bool
            Whether to return the points in lon, lat coordinates
        sample_dx: float
            The sampling distance in m for boundary conversion to lonlat

        Returns
        -------
        x_mima: numpy.ndarray
            The (x_min, x_max) point
        y_mima: numpy.ndarray
            The (y_min, y_max) point

        """
        if self.boundary is not None:
            xy = np.stack((self.boundary.p_min(), self.boundary.p_max()), axis=0)
        else:
            xy = self.xy_array

        if extra_space is not None:
            if isinstance(extra_space, str):
                assert algo is not None, (
                    f"WindFarm: require algo argument for extra_space '{extra_space}'"
                )
                assert len(extra_space) > 1 and extra_space[-1] == "D", (
                    f"Expecting float or str like '2.5D', got extra_space = '{extra_space}'"
                )
                extra_space = float(extra_space[:-1])
                rds = self.get_rotor_diameters(algo)
                if self.boundary is not None:
                    extra_space *= np.max(rds)
                else:
                    extra_space *= rds[:, None]

            xy = np.concatenate((xy - extra_space, xy + extra_space), axis=0)

        p_min = np.min(xy, axis=0)
        p_max = np.max(xy, axis=0)

        if lonlat:
            x0, y0 = p_min
            x1, y1 = p_max
            nx = int(np.ceil((x1 - x0) / sample_dx)) + 1
            ny = int(np.ceil((y1 - y0) / sample_dx)) + 1
            xy = np.concatenate(
                (
                    np.linspace([x0, y0], [x0, y1], ny),
                    np.linspace([x0, y1], [x1, y1], nx),
                    np.linspace([x1, y1], [x1, y0], ny),
                    np.linspace([x1, y0], [x0, y0], nx),
                ),
                axis=0,
            )
            xy = to_lonlat(xy)
            p_min = np.min(xy, axis=0)
            p_max = np.max(xy, axis=0)

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
            t.D if t.D is not None else algo.farm_controller.turbine_types[i].D
            for i, t in enumerate(self.__turbines)
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
            t.H if t.H is not None else algo.farm_controller.turbine_types[i].H
            for i, t in enumerate(self.__turbines)
        ]
        return np.array(hhs, dtype=config.dtype_double)

    def get_capacity(self, algo):
        """
        Gets the total capacity of the wind farm

        Parameters
        ----------
        algo: foxes.core.Algorithm
            The algorithm

        Returns
        -------
        capa: float
            The total capacity in W

        """
        ttypes = algo.farm_controller.turbine_types
        assert ttypes is not None, (
            f"WindFarm '{self.name}': turbine types not set in farm controller {algo.farm_controller.name}"
        )

        cap = 0.0
        for tt in ttypes:
            assert tt.P_nominal is not None, (
                f"WindFarm '{self.name}': P_nominal not set for turbine type '{tt.name}' "
            )
            cap += tt.P_nominal
        return cap
