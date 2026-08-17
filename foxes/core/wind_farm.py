from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Any

from foxes.config import config
from foxes.utils import get_utm_zone, from_lonlat, to_lonlat
from foxes.utils.geojson_utils import (
    area_contains_point,
    normalize_areas_input,
)

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm
    from foxes.core.turbine import Turbine


class WindFarm:
    """
    The wind farm.

    Attributes
    ----------
    name
        The wind farm name
    turbines
        The wind turbines
    boundary
        The wind farm boundary

    :group: core

    """

    def __init__(
        self,
        name: str = "wind_farm",
        boundary: Any = None,
        input_is_lonlat: bool = False,
        utm_zone: Any = None,
    ) -> None:
        """
        Construct the wind farm.

        Parameters
        ----------
        name
            The wind farm name.
        boundary
            The wind farm boundary.
        input_is_lonlat
            Whether the input coordinates are given as lon/lat. If True, the
            coordinates are converted to UTM as specified by the utm_zone
            parameter.
        utm_zone
            Method for setting the UTM zone in the config if it is not already
            set. Supported options include values such as "from_turbine_X",
            "from_farm", "XA", a (lon, lat) tuple, or None.

        """
        self.name = name
        self.__turbines: list[Turbine] = []
        self.boundary = boundary

        self.__data_is_lonlat = input_is_lonlat
        self.__utm_zone = utm_zone
        self.__locked = False
        self.__cluster_areas: dict[str, Any] | None = None
        self.__lonlat: np.ndarray | None = None

    @property
    def data_is_lonlat(self) -> bool:
        """
        Return whether input coordinates are given as latitude/longitude.

        Returns
        -------
        data_is_lonlat
            ``True`` if the input coordinates are given in latitude/longitude.

        """
        return self.__data_is_lonlat

    @property
    def locked(self) -> bool:
        """
        Return whether the wind farm is locked.

        Returns
        -------
        locked
            ``True`` if the wind farm is locked and no more turbines may be
            added.

        """
        return self.__locked

    @property
    def turbines(self) -> list[Turbine]:
        """
        Return the list of wind turbines.

        Returns
        -------
        turbines
            The wind turbines.

        """
        if not self.__locked:
            self.__locked = True
            if self.__data_is_lonlat:
                self.__lonlat = np.zeros(
                    (self.n_turbines, 2), dtype=config.dtype_double
                )
                assert self.__lonlat is not None
                for i, t in enumerate(self.__turbines):
                    self.__lonlat[i, :] = t.xy
                if not config.utm_zone_set and self.__utm_zone is None:
                    raise ValueError(
                        f"WindFarm '{self.name}': input_is_lonlat is True, but config.utm_zone and utm_zone are None"
                    )
                if self.__utm_zone is None:
                    zone = config.utm_zone
                elif self.__utm_zone == "from_farm":
                    lonlat = np.mean(self.__lonlat, axis=0)
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

            elif self.__utm_zone is not None:
                if isinstance(self.__utm_zone, str) and len(self.__utm_zone) <= 3:
                    zone = (int(self.__utm_zone[:-1]), self.__utm_zone[-1])
                elif len(self.__utm_zone) == 2:
                    lonlat = np.asarray(self.__utm_zone)
                    zone = get_utm_zone(lonlat[None, :])
                else:
                    raise ValueError(
                        f"WindFarm '{self.name}': invalid utm_zone argument: {self.__utm_zone} for 'input_is_lonlat=False'"
                    )
                if not config.utm_zone_set:
                    config.set_utm_zone(*zone)
                elif config.utm_zone != zone:
                    raise ValueError(
                        f"WindFarm '{self.name}': config.utm_zone = {config.utm_zone} differs from requested zone {zone}"
                    )

        return self.__turbines

    def lock(self, verbosity: int = 1) -> None:
        """
        Lock the wind farm so no more turbines can be added.

        Parameters
        ----------
        verbosity
            The output verbosity; ``0`` is silent.

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

    def reset_turbines(
        self,
        algo: Algorithm,
        turbines: list[Turbine] | None = None,
    ) -> None:
        """
        Reset the wind farm turbines.

        Parameters
        ----------
        algo
            The algorithm.
        turbines
            The new list of turbines. If None, the turbine list is cleared.

        """
        assert not algo.initialized, (
            f"WindFarm '{self.name}': cannot reset turbines, algorithm '{algo.name}' is already initialized"
        )
        self.__locked = False
        if turbines is None:
            self.__turbines = []
        else:
            self.__turbines = turbines
        algo.update_n_turbines()

    def add_turbine(self, turbine: Turbine, verbosity: int = 1) -> None:
        """
        Add a wind turbine to the list.

        Parameters
        ----------
        turbine
            The wind turbine.
        verbosity
            The output verbosity; ``0`` is silent.

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
            wf_cl_str = (
                f", {turbine.wind_farm_name}"
                if turbine.wind_farm_name is not None
                else ""
            )
            wf_cl_str += (
                f", {turbine.cluster_name}" if turbine.cluster_name is not None else ""
            )
            if self.data_is_lonlat:
                print(
                    f"Turbine {turbine.index}, {turbine.name}{wf_cl_str}: lonlat=({turbine.xy[0]:.6f}, {turbine.xy[1]:.6f}), {', '.join(turbine.models)}"
                )
            elif config.utm_zone_set:
                utmn, utml = config.utm_zone
                print(
                    f"Turbine {turbine.index}, {turbine.name}{wf_cl_str}: UTM {utmn}{utml}, xy=({turbine.xy[0]:.2f}, {turbine.xy[1]:.2f}), {', '.join(turbine.models)}"
                )
            else:
                print(
                    f"Turbine {turbine.index}, {turbine.name}{wf_cl_str}: xy=({turbine.xy[0]:.2f}, {turbine.xy[1]:.2f}), {', '.join(turbine.models)}"
                )

    @property
    def lonlat(self) -> np.ndarray | None:
        """
        The lon, lat coordinates of the turbines, if input_is_lonlat was True.

        Returns
        -------
        lonlat
            The lon, lat coordinates of the turbines, shape: (n_turbines, 2), or None if input_is_lonlat was False

        """
        self.turbines
        return self.__lonlat

    def has_lonlat(self) -> bool:
        """
        Check if lon-lat coordinates are available

        Returns
        -------
        has_lonlat
            True if lon-lat coordinates are available, False otherwise

        """
        return self.__lonlat is not None

    def map_turbines_to_areas(
        self,
        areas: Any,
        set_cluster: bool = True,
        geojson_name_key: str | list[str] = "name",
    ) -> dict[str, list[int]]:
        """
        Maps turbines to areas.

        Parameters
        ----------
        areas
            The areas to map turbines to. Accepted forms are:
            - area geometry objects
            - (name, area geometry) pairs for named areas
            - a mapping of names to area geometry objects
            - path to GeoJSON file
            - GeoJSON object
        set_cluster
            If True, set each mapped turbine's cluster_name to
            the mapped area name.
        geojson_name_key
            Preferred GeoJSON feature property key(s) used
            to read area names from GeoJSON inputs.

        Returns
        -------
        mapping
            A dictionary, where keys are area names and values are
            lists of turbine indices belonging to that area.

        """
        area_map = normalize_areas_input(areas, geojson_name_key)

        mapping: dict[str, list[int]] = {name: [] for name in area_map}
        for i, t in enumerate(self.__turbines):
            for name, area in area_map.items():
                if area_contains_point(area, t.xy):
                    mapping[name].append(i)
                    if set_cluster:
                        t.cluster_name = name
                    break

        if set_cluster:
            if self.__cluster_areas is None:
                self.__cluster_areas = area_map
            else:
                self.__cluster_areas.update(area_map)

        return mapping

    @property
    def cluster_areas(self) -> dict[str, Any] | None:
        """
        The cluster areas, if set by map_turbines_to_areas.

        Returns
        -------
        cluster_areas
            The mapping from cluster names to AreaGeometry objects, or
            None if not set

        """
        return self.__cluster_areas

    @property
    def utm_zone(self) -> str | None:
        """
        The UTM zone of the wind farm, if set.

        Returns
        -------
        utm_zone
            The UTM zone as a string, or None if not set

        """
        return (
            f"{config.utm_zone[0]}{config.utm_zone[1]}" if config.utm_zone_set else None
        )

    @property
    def n_turbines(self) -> int:
        """
        The number of turbines in the wind farm

        Returns
        -------
        n_turbines
            The total number of turbines

        """
        return len(self.__turbines)

    @property
    def turbine_names(self) -> list[str]:
        """
        The list of names of all turbines

        Returns
        -------
        names
            The names of all turbines

        """
        return [
            t.name if t.name is not None else f"T{t.index}" for t in self.__turbines
        ]

    @property
    def xy_array(self) -> np.ndarray:
        """
        Returns an array of the wind farm ground points

        Returns
        -------
        xya
            The turbine ground positions, shape: (n_turbines, 2)

        """
        return np.array([t.xy for t in self.__turbines], dtype=config.dtype_double)

    @property
    def wind_farm_names(self) -> list[str] | None:
        """
        The list of wind farm names for all turbines

        Returns
        -------
        names
            The wind farm names for all turbines

        """
        fnames = list(
            set(
                [
                    t.wind_farm_name if t.wind_farm_name is not None else self.name
                    for t in self.__turbines
                ]
            )
        )
        return fnames if fnames != [None] else None

    def get_wind_farm_mapping(self) -> dict[str, list[int]]:
        """
        Returns a mapping from wind farm names to turbine indices

        Returns
        -------
        mapping
            A dictionary, where keys are wind farm names and
            values are lists of turbine indices belonging to that wind farm

        """
        mapping: dict[str, list[int]] = {}
        for i, t in enumerate(self.__turbines):
            wf_name = t.wind_farm_name if t.wind_farm_name is not None else self.name
            if wf_name not in mapping:
                mapping[wf_name] = []
            mapping[wf_name].append(i)
        return mapping

    @property
    def wind_farm_list(self) -> list[str]:
        """
        Returns a list of wind farm names for all turbines

        Returns
        -------
        wf_list
            A list of wind farm names for all turbines

        """
        return [
            t.wind_farm_name if t.wind_farm_name is not None else self.name
            for t in self.__turbines
        ]

    @property
    def cluster_names(self) -> list[str | None] | None:
        """
        The list of cluster names for all turbines

        Returns
        -------
        names
            The cluster names for all turbines

        """
        clusters = list(
            set(
                [
                    t.cluster_name if t.cluster_name is not None else None
                    for t in self.__turbines
                ]
            )
        )
        return clusters if clusters != [None] else None

    def get_cluster_mapping(self) -> dict[str | None, list[int]] | None:
        """
        Returns a mapping from cluster names to turbine indices

        Returns
        -------
        mapping
            A dictionary, where keys are cluster names and
            values are lists of turbine indices belonging to that cluster

        """
        mapping: dict[str | None, list[int]] = {}
        for i, t in enumerate(self.__turbines):
            cluster_name = t.cluster_name if t.cluster_name is not None else None
            if cluster_name not in mapping:
                mapping[cluster_name] = []
            mapping[cluster_name].append(i)
        return mapping if list(mapping.keys()) != [None] else None

    @property
    def cluster_list(self) -> list[str | None]:
        """
        Returns a list of cluster names for all turbines

        Returns
        -------
        cluster_list
            A list of cluster names for all turbines

        """
        return [
            t.cluster_name if t.cluster_name is not None else None
            for t in self.__turbines
        ]

    def get_xy_bounds(
        self,
        extra_space: float | str | None = None,
        algo: Algorithm | None = None,
        lonlat: bool = False,
        sample_dx: float = 10.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns min max points of the wind farm ground points

        Parameters
        ----------
        extra_space
            The extra space, either float in m,
            or str for units of D, e.g. '2.5D'
        algo
            The algorithm
        lonlat
            Whether to return the points in lon, lat coordinates
        sample_dx
            The sampling distance in m for boundary conversion to lonlat

        Returns
        -------
        x_mima
            The (x_min, x_max) point
        y_mima
            The (y_min, y_max) point

        """
        if self.boundary is not None:
            xy = np.stack((self.boundary.p_min(), self.boundary.p_max()), axis=0)
        else:
            xy = self.xy_array

        if extra_space is not None:
            extra_space_value: float | np.ndarray
            if isinstance(extra_space, str):
                assert algo is not None, (
                    f"WindFarm: require algo argument for extra_space '{extra_space}'"
                )
                assert len(extra_space) > 1 and extra_space[-1] == "D", (
                    f"Expecting float or str like '2.5D', got extra_space = '{extra_space}'"
                )
                extra_space_value = float(extra_space[:-1])
                rds = self.get_rotor_diameters(algo)
                if self.boundary is not None:
                    extra_space_value *= np.max(rds)
                else:
                    extra_space_value = extra_space_value * rds[:, None]
            else:
                extra_space_value = float(extra_space)

            xy = np.concatenate(
                (xy - extra_space_value, xy + extra_space_value), axis=0
            )

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

    def get_rotor_diameters(self, algo: Algorithm) -> np.ndarray:
        """
        Gets the rotor diameters

        Parameters
        ----------
        algo
            The algorithm

        Returns
        -------
        rds
            The rotor diameters, shape: (n_turbienes,)

        """
        farm_controller = self._get_farm_controller(algo)
        rds = [
            t.D if t.D is not None else farm_controller.turbine_types[i].D
            for i, t in enumerate(self.__turbines)
        ]
        return np.array(rds, dtype=config.dtype_double)

    def get_hub_heights(self, algo: Algorithm) -> np.ndarray:
        """
        Gets the hub heights

        Parameters
        ----------
        algo
            The algorithm

        Returns
        -------
        hhs
            The hub heights, shape: (n_turbines,)

        """
        farm_controller = self._get_farm_controller(algo)
        hhs = [
            t.H if t.H is not None else farm_controller.turbine_types[i].H
            for i, t in enumerate(self.__turbines)
        ]
        return np.array(hhs, dtype=config.dtype_double)

    def get_capacity(self, algo: Algorithm) -> float:
        """
        Gets the total capacity of the wind farm

        Parameters
        ----------
        algo
            The algorithm

        Returns
        -------
        capa
            The total capacity in W

        """
        farm_controller = self._get_farm_controller(algo)
        ttypes = farm_controller.turbine_types
        assert ttypes is not None, (
            f"WindFarm '{self.name}': turbine types not set in farm controller {farm_controller.name}"
        )

        cap = 0.0
        for tt in ttypes:
            assert tt.P_nominal is not None, (
                f"WindFarm '{self.name}': P_nominal not set for turbine type '{tt.name}' "
            )
            cap += tt.P_nominal
        return cap

    def get_capacity_array(self, algo: Algorithm) -> np.ndarray:
        """
        Gets the capacity array for all turbines (nominal power)

        Parameters
        ----------
        algo
            The algorithm

        Returns
        -------
        capacity_array
            The capacity array (nominal power) for all turbines, shape: (n_turbines,)

        """
        farm_controller = self._get_farm_controller(algo)
        ttypes = farm_controller.turbine_types
        assert ttypes is not None, (
            f"WindFarm '{self.name}': turbine types not set in farm controller {farm_controller.name}"
        )

        capacity_array: np.ndarray = np.zeros(
            self.n_turbines, dtype=config.dtype_double
        )
        for i, t in enumerate(self.__turbines):
            tt = ttypes[i]
            assert tt.P_nominal is not None, (
                f"WindFarm '{self.name}': P_nominal not set for turbine type '{tt.name}' "
            )
            capacity_array[i] = tt.P_nominal
        return capacity_array

    def _get_farm_controller(self, algo: Algorithm) -> Any:
        return algo.farm_controller
