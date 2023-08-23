import numpy as np
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class AreaGeometry(metaclass=ABCMeta):
    """
    Abstract base class for closed 2D geometries.

    :group: utils.geom2d

    """

    @abstractmethod
    def p_min(self):
        """
        Returns minimal (x,y) point.

        Returns
        -------
        p_min: numpy.ndarray
            The minimal (x,y) point, shape = (2,)

        """
        pass

    @abstractmethod
    def p_max(self):
        """
        Returns maximal (x,y) point.

        Returns
        -------
        p_min: numpy.ndarray
            The maximal (x,y) point, shape = (2,)

        """
        pass

    @abstractmethod
    def points_distance(self, points, return_nearest=False):
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
    def points_inside(self, points):
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
        show_boundary=False,
        fill_mode="inside_slategray",
        pars_boundary={},
        pars_distance={},
    ):
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
                dists = self.points_distance(pts).reshape(Nx, Ny)
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

    def inverse(self):
        """
        Get the inverted geometry

        Returns
        -------
        inverted: foxes.utils.geom2d.InvertedAreaGeometry
            The inverted geometry

        """
        return InvertedAreaGeometry(self)

    def __add__(self, g):
        if isinstance(g, list):
            return AreaUnion([self] + g)
        elif isinstance(g, AreaUnion):
            return AreaUnion([self] + g.geometries)
        else:
            return AreaUnion([self, g])

    def __sub__(self, g):
        if isinstance(g, list):
            return AreaIntersection([self] + g.inverse())
        else:
            return AreaIntersection([self, g.inverse()])


class InvertedAreaGeometry(AreaGeometry):
    """
    Base class for inverted geometries.

    :group: utils.geom2d

    """

    def __init__(self, geometry):
        """
        Constructor.

        Parameters
        ----------
        geometry: geom2d.AreaGeometry
            The original geometry

        """
        self._geometry = geometry

    def p_min(self):
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

    def p_max(self):
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

    def points_distance(self, points, return_nearest=False):
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

    def points_inside(self, points):
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
        show_boundary=False,
        fill_mode="inside_slategray",
        pars_boundary={},
        pars_distance={},
    ):
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

    def inverse(self):
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

    def __init__(self, geometries):
        """
        Constructor.

        Parameters
        ----------
        geometries: list of geom2d.AreaGeometry
            The geometries

        """
        self.geometries = geometries

    def p_min(self):
        """
        Returns minimal (x,y) point.

        Returns
        -------
        p_min: numpy.ndarray
            The minimal (x,y) point, shape = (2,)

        """
        out = None
        for g in self.geometries:
            pmi = g.p_min()
            if out is None:
                out = pmi
            else:
                out = np.minimum(out, pmi)
        return out

    def p_max(self):
        """
        Returns maximal (x,y) point.

        Returns
        -------
        p_min: numpy.ndarray
            The maximal (x,y) point, shape = (2,)

        """
        out = None
        for g in self.geometries:
            pma = g.p_max()
            if out is None:
                out = pma
            else:
                out = np.maximum(out, pma)
        return out

    def points_distance(self, points, return_nearest=False):
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
            d = res[0] if return_nearest else res

            # was outside, is outside:
            sel = ~pins & ~ins & (d < dist)
            if np.any(sel):
                dist[sel] = d[sel]
                if return_nearest:
                    nerst[sel] = res[1][sel]

            # was outside, is inside:
            sel = ~pins & ins
            if np.any(sel):
                pins[sel] = True
                dist[sel] = d[sel]
                if return_nearest:
                    nerst[sel] = res[1][sel]

            # was inside, is inside:
            sel = pins & ins & (d > dist)
            if np.any(sel):
                dist[sel] = d[sel]
                if return_nearest:
                    nerst[sel] = res[1][sel]

        if return_nearest:
            return dist, nerst
        else:
            return dist

    def points_inside(self, points):
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
        show_boundary=False,
        fill_mode="inside_slategray",
        pars_boundary={},
        pars_distance={},
    ):
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

    def inverse(self):
        """
        Get the inverted geometry

        Returns
        -------
        inverted: foxes.utils.geom2d.InvertedAreaGeometry
            The inverted geometry

        """
        return InvertedAreaUnion(self)

    def __add__(self, g):
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

    def __init__(self, union):
        """
        Constructor.

        Parameters
        ----------
        union: geom2d.AreaUnion
            The original area union geometry

        """
        super().__init__(union)

    def p_min(self):
        """
        Returns minimal (x,y) point.

        Returns
        -------
        p_min: numpy.ndarray
            The minimal (x,y) point, shape = (2,)

        """
        if len(self._geometry.geometries) == 1:
            return self._geometry.geometries[0].inverse().p_min()

        pmi = self._geometry.p_min()
        if not np.any(np.isinf(pmi)):
            return np.full(2, -np.inf, dtype=np.float64)
        else:
            out = np.full(2, np.inf, dtype=np.float64)
            for g in self._geometry.geometries:
                imi = g.inverse().p_min()
                for di in range(2):
                    if np.isinf(pmi[di]) and not np.isinf(imi[di]):
                        out[di] = np.minimum(out[di], imi[di])
            for di in range(2):
                if not np.isinf(pmi[di]):
                    out[di] = -np.inf
            return out

    def p_max(self):
        """
        Returns maximal (x,y) point.

        Returns
        -------
        p_min: numpy.ndarray
            The maximal (x,y) point, shape = (2,)

        """
        if len(self._geometry.geometries) == 1:
            return self._geometry.geometries[0].inverse().p_max()

        pma = self._geometry.p_max()
        if not np.any(np.isinf(pma)):
            return np.full(2, np.inf, dtype=np.float64)
        else:
            out = np.full(2, -np.inf, dtype=np.float64)
            for g in self._geometry.geometries:
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

    def __new__(cls, geometries):
        """
        Constructor.

        Parameters
        ----------
        geometries: list of geom2d.AreaGeometry
            The geometries

        """
        return AreaUnion([g.inverse() for g in geometries]).inverse()
