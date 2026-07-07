from __future__ import annotations

import numpy as np
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import Any, cast


class AreaGeometry(metaclass=ABCMeta):
    """
    Abstract base class for closed 2D geometries.

    :group: utils.geom2d

    """

    @abstractmethod
    def p_min(self) -> np.ndarray:
        """
        Returns minimal (x,y) point.

        Returns
        -------
        p_min: numpy.ndarray
            The minimal (x,y) point, shape = (2,)

        """
        pass

    @abstractmethod
    def p_max(self) -> np.ndarray:
        """
        Returns maximal (x,y) point.

        Returns
        -------
        p_min: numpy.ndarray
            The maximal (x,y) point, shape = (2,)

        """
        pass

    def box_centre(self) -> np.ndarray:
        """
        Returns centre (x,y) point of the surrounding box.

        Returns
        -------
        centre: numpy.ndarray
            The centre (x,y) point, shape = (2,)

        """
        return 0.5 * (self.p_min() + self.p_max())

    @abstractmethod
    def points_distance(
        self, points: np.ndarray, return_nearest: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Calculates point distances wrt boundary.

        Parameters
        ----------
        points: numpy.ndarray
            The probe points, shape (n_points, 2)
        return_nearest: bool
            Flag for return of the nearest point on bundary

        Returns
        -------
        dist: numpy.ndarray
            The smallest distances to the boundary,
            shape: (n_points,)
        p_nearest: numpy.ndarray, optional
            The nearest points on the boundary, if
            return_nearest is True, shape: (n_points, 2)

        """
        pass

    @abstractmethod
    def points_inside(self, points: np.ndarray) -> np.ndarray:
        """
        Tests if points are inside the geometry.

        Parameters
        ----------
        points: numpy.ndarray
            The probe points, shape (n_points, 2)

        Returns
        -------
        inside: numpy.ndarray
            True if point is inside, shape: (n_points,)

        """
        pass

    def add_to_figure(
        self,
        ax,
        show_boundary: bool = False,
        fill_mode: str | None = "inside_slategray",
        pars_boundary: dict[str, Any] | None = None,
        pars_distance: dict[str, Any] | None = None,
    ) -> None:
        """
        Add image to (x,y) figure.

        Parameters
        ----------
        ax: matplotlib.pyplot.Axis
            The axis object
        show_boundary: bool
            Add the boundary line to the image
        fill_mode: str, optional
            Fill the area. Options:
            dist, dist_inside, dist_outside, inside_<color>,
            outside_<color>
        pars_boundary: dict
            Parameters for boundary plotting command
        pars_distance: dict
            Parameters for distance plotting command

        """
        pars_boundary = {} if pars_boundary is None else pars_boundary
        pars_distance = {} if pars_distance is None else pars_distance

        if fill_mode is not None:
            if "Nx" in pars_distance or "Ny" in pars_distance:
                Nx = pars_distance.pop("Nx")
                Ny = pars_distance.pop("Ny")
            elif "N" in pars_distance:
                Nx = pars_distance.pop("N")
                Ny = Nx
            else:
                Nx = 500
                Ny = 500

            p0 = pars_distance.pop("p_min", self.p_min())
            p1 = pars_distance.pop("p_max", self.p_max())
            if np.isinf(p0[0]):
                q0 = self.inverse().p_min()
                a0 = ax.get_xlim()[0]
                p0[0] = a0 if a0 < q0[0] else q0[0]
            if np.isinf(p0[1]):
                q0 = self.inverse().p_min()
                a0 = ax.get_ylim()[0]
                p0[1] = a0 if a0 < q0[1] else q0[1]
            if np.isinf(p1[0]):
                q1 = self.inverse().p_max()
                a1 = ax.get_xlim()[1]
                p1[0] = a1 if a1 > q1[0] else q1[0]
            if np.isinf(p1[1]):
                q1 = self.inverse().p_max()
                a1 = ax.get_ylim()[1]
                p1[1] = a1 if a1 > q1[1] else q1[1]

            delta = p1 - p0
            p0 -= 0.05 * delta
            p1 += 0.05 * delta

            x = np.linspace(p0[0], p1[0], Nx)
            y = np.linspace(p0[1], p1[1], Ny)

            pts = np.zeros((Nx, Ny, 2))
            pts[..., 0] = x[:, None]
            pts[..., 1] = y[None, :]
            pts = pts.reshape(Nx * Ny, 2)

            pars = dict(shading="auto", cmap="magma_r", zorder=-100)
            sbar = True
            if fill_mode == "dist":
                dres = self.points_distance(pts)
                dists = cast(np.ndarray, dres).reshape(Nx, Ny)
            elif fill_mode == "dist_inside":
                ins = self.points_inside(pts)
                dists = np.full(Nx * Ny, np.nan, dtype=np.float64)
                dists[ins] = self.points_distance(pts[ins])
                dists = dists.reshape(Nx, Ny)
            elif fill_mode[:7] == "inside_":
                ins = self.points_inside(pts)
                dists = np.full(Nx * Ny, np.nan, dtype=np.float64)
                dists[ins] = 1.0
                dists = dists.reshape(Nx, Ny)
                pars["cmap"] = ListedColormap([fill_mode[7:]])
                sbar = False
            elif fill_mode == "dist_outside":
                ins = self.points_inside(pts)
                dists = np.full(Nx * Ny, np.nan, dtype=np.float64)
                dists[~ins] = self.points_distance(pts[~ins])
                dists = dists.reshape(Nx, Ny)
            elif fill_mode[:8] == "outside_":
                ins = self.points_inside(pts)
                dists = np.full(Nx * Ny, np.nan, dtype=np.float64)
                dists[~ins] = 1
                dists = dists.reshape(Nx, Ny)
                pars["cmap"] = ListedColormap([fill_mode[8:]])
                sbar = False
            else:
                raise ValueError(
                    f"Illegal parameter 'fill_mode = {fill_mode}', expecting: None, dist, dist_inside, dist_outside"
                )

            pars.update(pars_distance)
            im = ax.pcolormesh(x, y, dists.T, **pars)
            if sbar:
                plt.colorbar(im, ax=ax, orientation="vertical", label="distance")

        ax.autoscale_view()
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal", adjustable="box")

    def inverse(self) -> AreaGeometry:
        """
        Get the inverted geometry

        Returns
        -------
        inverted: foxes.utils.geom2d.InvertedAreaGeometry
            The inverted geometry

        """
        return InvertedAreaGeometry(self)

    @staticmethod
    def from_shp(
        fname,
        names=None,
        name_col="Name",
        geom_col="geometry",
        to_utm=True,
        combine_mode="union",
        ret_utm_zone=False,
        **kwargs,
    ):
        """
        Read a shapefile into an ``AreaGeometry``.

        This is a convenience wrapper around
        :func:`foxes.utils.shp2geom2d`.

        Parameters
        ----------
        fname: str
            Path to the ``.shp`` file, or a glob pattern matching
            multiple ``.shp`` files. For glob patterns, matched
            geometries are combined according to `combine_mode`
        names: list of str, optional
            Names of polygons to extract. If None, all are used
        name_col: str
            Column containing polygon names
        geom_col: str
            Name of the geometry column
        to_utm: bool or str
            Convert to UTM coordinates. If str, use the given
            zone+letter (e.g. ``"32U"``)
        combine_mode: str
            The combination mode for multiple areas. Options:
            ``"union"`` (default), ``"intersection"``
        ret_utm_zone: bool
            Return UTM zone plus letter as str in addition to geometry
        kwargs: dict, optional
            Additional parameters forwarded to ``geopandas.read_file``

        Returns
        -------
        geom: foxes.utils.geom2d.AreaGeometry
            The loaded geometry
        utm_zone_str: str, optional
            Returned only if ``ret_utm_zone`` is True

        :group: utils.geom2d

        """
        from ..geopandas_utils import shp2geom2d

        return shp2geom2d(
            fname,
            names=names,
            name_col=name_col,
            geom_col=geom_col,
            to_utm=to_utm,
            combine_mode=combine_mode,
            ret_utm_zone=ret_utm_zone,
            **kwargs,
        )

    def __add__(self, g) -> AreaUnion:
        if isinstance(g, list):
            return AreaUnion([self] + g)
        elif isinstance(g, AreaUnion):
            return AreaUnion([self] + g.geometries)
        else:
            return AreaUnion([self, g])

    def __sub__(self, g) -> AreaIntersection:
        if isinstance(g, list):
            return AreaIntersection([self] + [gi.inverse() for gi in g])
        else:
            return AreaIntersection([self, g.inverse()])


class InvertedAreaGeometry(AreaGeometry):
    """
    Base class for inverted geometries.

    :group: utils.geom2d

    """

    def __init__(self, geometry: AreaGeometry) -> None:
        """
        Constructor.

        Parameters
        ----------
        geometry: geom2d.AreaGeometry
            The original geometry

        """
        self._geometry = geometry

    def p_min(self) -> np.ndarray:
        """
        Returns minimal (x,y) point.

        Returns
        -------
        p_min: numpy.ndarray
            The minimal (x,y) point, shape = (2,)

        """
        pmi = self._geometry.p_min()
        if not np.any(np.isinf(pmi)):
            return np.full(2, -np.inf, dtype=np.float64)
        elif isinstance(self._geometry, InvertedAreaGeometry):
            out = np.full(2, np.inf, dtype=np.float64)
            imi = self._geometry.inverse().p_min()
            for di in range(2):
                if np.isinf(pmi[di]) and not np.isinf(imi[di]):
                    out[di] = np.minimum(out[di], imi[di])
                if not np.isinf(pmi[di]):
                    out[di] = -np.inf
            return out
        else:
            return np.full(2, -np.inf, dtype=np.float64)

    def p_max(self) -> np.ndarray:
        """
        Returns maximal (x,y) point.

        Returns
        -------
        p_min: numpy.ndarray
            The maximal (x,y) point, shape = (2,)

        """
        pma = self._geometry.p_max()
        if not np.any(np.isinf(pma)):
            return np.full(2, np.inf, dtype=np.float64)
        elif isinstance(self._geometry, InvertedAreaGeometry):
            out = np.full(2, -np.inf, dtype=np.float64)
            ima = self._geometry.inverse().p_max()
            for di in range(2):
                if np.isinf(pma[di]) and not np.isinf(ima[di]):
                    out[di] = np.maximum(out[di], ima[di])
                if not np.isinf(pma[di]):
                    out[di] = np.inf
            return out
        else:
            return np.full(2, np.inf, dtype=np.float64)

    def points_distance(
        self, points: np.ndarray, return_nearest: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Calculates point distances wrt boundary.

        Parameters
        ----------
        points: numpy.ndarray
            The probe points, shape (n_points, 2)
        return_nearest: bool
            Flag for return of the nearest point on bundary

        Returns
        -------
        dist: numpy.ndarray
            The smallest distances to the boundary,
            shape: (n_points,)
        p_nearest: numpy.ndarray, optional
            The nearest points on the boundary, if
            return_nearest is True, shape: (n_points, 2)

        """
        return self._geometry.points_distance(points, return_nearest)

    def points_inside(self, points: np.ndarray) -> np.ndarray:
        """
        Tests if points are inside the geometry.

        Parameters
        ----------
        points: numpy.ndarray
            The probe points, shape (n_points, 2)

        Returns
        -------
        inside: numpy.ndarray
            True if point is inside, shape: (n_points,)

        """
        return ~self._geometry.points_inside(points)

    def add_to_figure(
        self,
        ax,
        show_boundary: bool = False,
        fill_mode: str | None = "inside_slategray",
        pars_boundary: dict[str, Any] | None = None,
        pars_distance: dict[str, Any] | None = None,
    ) -> None:
        """
        Add image to (x,y) figure.

        Parameters
        ----------
        ax: matplotlib.pyplot.Axis
            The axis object
        show_boundary: bool
            Add the boundary line to the image
        fill_mode: str, optional
            Fill the area. Options:
            dist, dist_inside, dist_outside, inside_<color>,
            outside_<color>
        pars_boundary: dict
            Parameters for boundary plotting command
        pars_distance: dict
            Parameters for distance plotting command

        """
        pars_boundary = {} if pars_boundary is None else pars_boundary
        pars_distance = {} if pars_distance is None else pars_distance

        self._geometry.add_to_figure(
            ax,
            show_boundary,
            fill_mode=None,
            pars_boundary=pars_boundary,
            pars_distance={},
        )
        super().add_to_figure(
            ax, show_boundary, fill_mode, pars_boundary, pars_distance
        )

    def inverse(self) -> AreaGeometry:
        """
        Get the inverted geometry

        Returns
        -------
        inverted: foxes.utils.geom2d.InvertedAreaGeometry
            The inverted geometry

        """
        return self._geometry


class AreaUnion(AreaGeometry):
    """
    The union of area geometries.

    Attributes
    ----------
    geometries: list of geom2d.AreaGeometry
        The geometries

    :group: utils.geom2d

    """

    def __init__(self, geometries: list[AreaGeometry]) -> None:
        """
        Constructor.

        Parameters
        ----------
        geometries: list of geom2d.AreaGeometry
            The geometries

        """
        self.geometries = geometries

    def p_min(self) -> np.ndarray:
        """
        Returns minimal (x,y) point.

        Returns
        -------
        p_min: numpy.ndarray
            The minimal (x,y) point, shape = (2,)

        """
        out: np.ndarray | None = None
        for g in self.geometries:
            pmi = g.p_min()
            if out is None:
                out = pmi
            else:
                out = np.minimum(out, pmi)
        assert out is not None
        return out

    def p_max(self) -> np.ndarray:
        """
        Returns maximal (x,y) point.

        Returns
        -------
        p_min: numpy.ndarray
            The maximal (x,y) point, shape = (2,)

        """
        out: np.ndarray | None = None
        for g in self.geometries:
            pma = g.p_max()
            if out is None:
                out = pma
            else:
                out = np.maximum(out, pma)
        assert out is not None
        return out

    def points_distance(
        self, points: np.ndarray, return_nearest: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Calculates point distances wrt boundary.

        Parameters
        ----------
        points: numpy.ndarray
            The probe points, shape (n_points, 2)
        return_nearest: bool
            Flag for return of the nearest point on bundary

        Returns
        -------
        dist: numpy.ndarray
            The smallest distances to the boundary,
            shape: (n_points,)
        p_nearest: numpy.ndarray, optional
            The nearest points on the boundary, if
            return_nearest is True, shape: (n_points, 2)

        """
        if len(self.geometries) == 1:
            return self.geometries[0].points_distance(points, return_nearest)

        n_pts = len(points)
        dist = np.full(n_pts, np.inf, dtype=np.float64)
        pins = np.zeros(n_pts, dtype=bool)
        nerst = np.zeros((n_pts, 2), dtype=np.float64) if return_nearest else None
        for g in self.geometries:
            res = g.points_distance(points, return_nearest)
            ins = g.points_inside(points)
            if return_nearest:
                rtuple = cast(tuple[np.ndarray, np.ndarray], res)
                d = rtuple[0]
                nearest = rtuple[1]
            else:
                d = cast(np.ndarray, res)

            # was outside, is outside:
            sel = ~pins & ~ins & (d < dist)
            if np.any(sel):
                dist[sel] = d[sel]
                if return_nearest:
                    assert nerst is not None
                    nerst[sel] = nearest[sel]

            # was outside, is inside:
            sel = ~pins & ins
            if np.any(sel):
                pins[sel] = True
                dist[sel] = d[sel]
                if return_nearest:
                    assert nerst is not None
                    nerst[sel] = nearest[sel]

            # was inside, is inside:
            sel = pins & ins & (d > dist)
            if np.any(sel):
                dist[sel] = d[sel]
                if return_nearest:
                    assert nerst is not None
                    nerst[sel] = nearest[sel]

        if return_nearest:
            assert nerst is not None
            return dist, nerst
        else:
            return dist

    def points_inside(self, points: np.ndarray) -> np.ndarray:
        """
        Tests if points are inside the geometry.

        Parameters
        ----------
        points: numpy.ndarray
            The probe points, shape (n_points, 2)

        Returns
        -------
        inside: numpy.ndarray
            True if point is inside, shape: (n_points,)

        """
        if len(self.geometries) == 1:
            return self.geometries[0].points_inside(points)

        n_pts = len(points)
        inside = np.zeros(n_pts, dtype=bool)
        for g in self.geometries:
            inside = inside | g.points_inside(points)
        return inside

    def add_to_figure(
        self,
        ax,
        show_boundary: bool = False,
        fill_mode: str | None = "inside_slategray",
        pars_boundary: dict[str, Any] | None = None,
        pars_distance: dict[str, Any] | None = None,
    ) -> None:
        """
        Add image to (x,y) figure.

        Parameters
        ----------
        ax: matplotlib.pyplot.Axis
            The axis object
        show_boundary: bool
            Add the boundary line to the image
        fill_mode: str, optional
            Fill the area. Options:
            dist, dist_inside, dist_outside, inside_<color>,
            outside_<color>
        pars_boundary: dict
            Parameters for boundary plotting command
        pars_distance: dict
            Parameters for distance plotting command

        """
        pars_boundary = {} if pars_boundary is None else pars_boundary
        pars_distance = {} if pars_distance is None else pars_distance

        if show_boundary:
            for g in self.geometries:
                g.add_to_figure(
                    ax,
                    show_boundary=True,
                    fill_mode=None,
                    pars_boundary=pars_boundary,
                    pars_distance={},
                )

        super().add_to_figure(
            ax,
            show_boundary=False,
            fill_mode=fill_mode,
            pars_boundary={},
            pars_distance=pars_distance,
        )

    def inverse(self) -> InvertedAreaUnion:
        """
        Get the inverted geometry

        Returns
        -------
        inverted: foxes.utils.geom2d.InvertedAreaGeometry
            The inverted geometry

        """
        return InvertedAreaUnion(self)

    def __add__(self, g) -> AreaUnion:
        if isinstance(g, list):
            return AreaUnion(self.geometries + g)
        elif isinstance(g, AreaUnion):
            return AreaUnion(self.geometries + g.geometries)
        else:
            return AreaUnion(self.geometries + [g])


class InvertedAreaUnion(InvertedAreaGeometry):
    """
    Inversion of a union of areas

    :group: utils.geom2d

    """

    def __init__(self, union: AreaUnion) -> None:
        """
        Constructor.

        Parameters
        ----------
        union: geom2d.AreaUnion
            The original area union geometry

        """
        super().__init__(union)

    def p_min(self) -> np.ndarray:
        """
        Returns minimal (x,y) point.

        Returns
        -------
        p_min: numpy.ndarray
            The minimal (x,y) point, shape = (2,)

        """
        union = cast(AreaUnion, self._geometry)
        if len(union.geometries) == 1:
            return union.geometries[0].inverse().p_min()

        pmi = union.p_min()
        if not np.any(np.isinf(pmi)):
            return np.full(2, -np.inf, dtype=np.float64)
        else:
            out = np.full(2, np.inf, dtype=np.float64)
            for g in union.geometries:
                imi = g.inverse().p_min()
                for di in range(2):
                    if np.isinf(pmi[di]) and not np.isinf(imi[di]):
                        out[di] = np.minimum(out[di], imi[di])
            for di in range(2):
                if not np.isinf(pmi[di]):
                    out[di] = -np.inf
            return out

    def p_max(self) -> np.ndarray:
        """
        Returns maximal (x,y) point.

        Returns
        -------
        p_min: numpy.ndarray
            The maximal (x,y) point, shape = (2,)

        """
        union = cast(AreaUnion, self._geometry)
        if len(union.geometries) == 1:
            return union.geometries[0].inverse().p_max()

        pma = union.p_max()
        if not np.any(np.isinf(pma)):
            return np.full(2, np.inf, dtype=np.float64)
        else:
            out = np.full(2, -np.inf, dtype=np.float64)
            for g in union.geometries:
                ima = g.inverse().p_max()
                for di in range(2):
                    if np.isinf(pma[di]) and not np.isinf(ima[di]):
                        out[di] = np.maximum(out[di], ima[di])
            for di in range(2):
                if not np.isinf(pma[di]):
                    out[di] = np.inf
            return out


class AreaIntersection(AreaGeometry):
    """
    The intersection of area geometries.

    :group: utils.geom2d

    """

    def __init__(self, geometries: list[AreaGeometry]) -> None:
        """
        Constructor.

        Parameters
        ----------
        geometries: list of geom2d.AreaGeometry
            The geometries

        """
        self.geometries = geometries
        self._geometry = AreaUnion([g.inverse() for g in geometries]).inverse()

    def p_min(self) -> np.ndarray:
        return self._geometry.p_min()

    def p_max(self) -> np.ndarray:
        return self._geometry.p_max()

    def points_distance(
        self, points: np.ndarray, return_nearest: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        return self._geometry.points_distance(points, return_nearest)

    def points_inside(self, points: np.ndarray) -> np.ndarray:
        return self._geometry.points_inside(points)

    def inverse(self) -> AreaGeometry:
        return self._geometry.inverse()


def from_shp(*args: Any, **kwargs: Any) -> Any:
    """
    Read a shapefile into an ``AreaGeometry``.

    This is a convenience wrapper for :meth:`AreaGeometry.from_shp`.

    Parameters
    ----------
    args: tuple
        Positional arguments forwarded to :meth:`AreaGeometry.from_shp`
    kwargs: dict
        Keyword arguments forwarded to :meth:`AreaGeometry.from_shp`

    Returns
    -------
    geom: foxes.utils.geom2d.AreaGeometry
        The loaded geometry
    utm_zone_str: str, optional
        Returned only if ``ret_utm_zone`` is True

    :group: utils.geom2d

    """
    return AreaGeometry.from_shp(*args, **kwargs)
