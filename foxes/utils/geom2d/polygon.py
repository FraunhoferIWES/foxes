import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

from .area_geometry import AreaGeometry


class ClosedPolygon(AreaGeometry):
    """
    This class represents a closed 2D polygon.

    Attributes
    ----------
    points: numpy.ndarray
        The polygon points
    poly: matplotlib.path.Path
        The closed polygon geometry

    :group: utils.geom2d

    """

    def __init__(self, points):
        """
        Constructor.

        Parameters
        ----------
        points: numpy.ndarray
            The polygon points, shape: (n_points, 2)

        """
        self.points = points

        if not np.all(points[0] == points[-1]):
            self.points = np.append(self.points, points[[0]], axis=0)

        self.poly = Path(self.points, closed=True)

        self._pathp = None

    def p_min(self):
        """
        Returns minimal (x,y) point.

        Returns
        -------
        p_min: numpy.ndarray
            The minimal (x,y) point, shape = (2,)

        """
        return np.min(self.points, axis=0)

    def p_max(self):
        """
        Returns maximal (x,y) point.

        Returns
        -------
        p_min: numpy.ndarray
            The maximal (x,y) point, shape = (2,)

        """
        return np.max(self.points, axis=0)

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

        dists = cdist(points, self.points[:-1])

        if return_nearest:
            mini = np.argmin(dists, axis=1)
            dists = np.take_along_axis(dists, mini[:, None], axis=1)[:, 0]
            minp = self.points[mini]
            del mini
        else:
            dists = np.min(dists, axis=1)

        for pi in range(len(self.points) - 1):
            pA = self.points[pi]
            pB = self.points[pi + 1]
            n = pB - pA
            d = np.linalg.norm(n)

            if d > 0:
                n /= d
                q = points - pA[None, :]
                x = np.einsum("pd,d->p", q, n)

                sel = (x > 0) & (x < d)
                if np.any(sel):
                    x = x[sel]
                    y2 = np.maximum(np.linalg.norm(q[sel], axis=1) ** 2 - x**2, 0.0)

                    dsel = dists[sel]
                    dists[sel] = np.minimum(dsel, np.sqrt(y2))

                    if return_nearest:
                        mini = np.argwhere(np.sqrt(y2) < dsel)
                        hminp = minp[sel]
                        hminp[mini] = pA[None, :] + x[mini, None] * n[None, :]
                        minp[sel] = hminp
                        del mini, hminp

                    del y2, dsel

                del x, sel

        if return_nearest:
            return dists, minp
        else:
            return dists

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
        return self.poly.contains_points(points)

    def add_to_figure(
        self, ax, show_boundary=True, fill_mode=None, pars_boundary={}, pars_distance={}
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
            pars = dict(facecolor="none", edgecolor="darkblue", linewidth=1)
            pars.update(pars_boundary)

            pathpatch = PathPatch(self.poly, **pars)
            ax.add_patch(pathpatch)

        super().add_to_figure(
            ax, show_boundary, fill_mode, pars_boundary, pars_distance
        )


if __name__ == "__main__":
    points = np.array([[1.0, 1.0], [1.3, 6], [5.8, 6.2], [6.5, 0.8]])
    N = 500

    fig, ax = plt.subplots()
    g = ClosedPolygon(points)
    g.add_to_figure(ax, fill_mode="dist_inside")
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    g = ClosedPolygon(points)
    g.add_to_figure(ax, fill_mode="dist_outside")
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    g = ClosedPolygon(points).inverse()
    g.add_to_figure(ax, fill_mode="dist_inside")
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    g = ClosedPolygon(points).inverse()
    g.add_to_figure(ax, fill_mode="dist_outside")
    plt.show()
    plt.close(fig)
