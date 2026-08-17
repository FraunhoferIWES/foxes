import numpy as np
import matplotlib.pyplot as plt
from typing import Any

from .area_geometry import AreaGeometry


class Circle(AreaGeometry):
    """
    This class represents the area of a circle.

    Attributes
    ----------
    centre
        The centre point, shape: (2,)
    radius
        The radius

    :group: utils.geom2d

    """

    def __init__(self, centre: np.ndarray, radius: float) -> None:
        """
        Constructor.

        Parameters
        ----------
        centre
            The centre point, shape: (2,)
        radius
            The radius

        """
        self.centre = np.array(centre, dtype=np.float64)
        self.radius = radius

    def p_min(self) -> np.ndarray:
        """
        Returns minimal (x,y) point.

        Returns
        -------
        p_min
            The minimal (x,y) point, shape = (2,)

        """
        return self.centre - self.radius

    def p_max(self) -> np.ndarray:
        """
        Returns maximal (x,y) point.

        Returns
        -------
        p_min
            The maximal (x,y) point, shape = (2,)

        """
        return self.centre + self.radius

    def points_distance(
        self, points: np.ndarray, return_nearest: bool = False
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Calculates point distances wrt boundary.

        Parameters
        ----------
        points
            The probe points, shape (n_points, 2)
        return_nearest
            Flag for return of the nearest point on bundary

        Returns
        -------
        dist
            The smallest distances to the boundary,
            shape: (n_points,)
        p_nearest
            The nearest points on the boundary, if
            return_nearest is True, shape: (n_points, 2)

        """

        deltas = points - self.centre[None, :]
        magd = np.linalg.norm(deltas, axis=-1)
        dists = np.abs(magd - self.radius)

        if return_nearest:
            sel = magd > 0.0
            if np.all(sel):
                minp = self.centre + deltas / magd[:, None] * self.radius
            else:
                minp = np.zeros_like(points)
                minp[sel] = deltas[sel] / magd[sel]
                minp[~sel][:, 0] = 1
                minp = self.centre + minp * self.radius
            return dists, minp
        else:
            return dists

    def points_inside(self, points: np.ndarray) -> np.ndarray:
        """
        Tests if points are inside the geometry.

        Parameters
        ----------
        points
            The probe points, shape (n_points, 2)

        Returns
        -------
        inside
            True if point is inside, shape: (n_points,)

        """
        magd = np.linalg.norm(points - self.centre[None, :], axis=-1)
        return magd <= self.radius

    def add_to_figure(
        self,
        ax,
        show_boundary: bool = True,
        fill_mode: str | None = None,
        pars_boundary: dict[str, Any] | None = None,
        pars_distance: dict[str, Any] | None = None,
    ) -> None:
        """
        Add image to (x,y) figure.

        Parameters
        ----------
        ax: matplotlib.pyplot.Axis
            The axis object
        show_boundary
            Add the boundary line to the image
        fill_mode
            Fill the area. Options:
            dist, dist_inside, dist_outside, inside_<color>,
            outside_<color>
        pars_boundary
            Parameters for boundary plotting command
        pars_distance
            Parameters for distance plotting command

        """
        pars_boundary = {} if pars_boundary is None else pars_boundary
        pars_distance = {} if pars_distance is None else pars_distance

        if show_boundary:
            pars = dict(color="darkblue", linewidth=1, fill=False)
            pars.update(pars_boundary)

            circle = plt.Circle(tuple(self.centre), self.radius, **pars)
            ax.add_patch(circle)

        super().add_to_figure(
            ax, show_boundary, fill_mode, pars_boundary, pars_distance
        )


if __name__ == "__main__":
    centre = np.array([3.0, 4.0])
    radius = 2.5
    N = 500
    g: AreaGeometry

    fig, ax = plt.subplots()
    g = Circle(centre, radius)
    g.add_to_figure(ax, fill_mode="dist_inside")
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    g = Circle(centre, radius)
    g.add_to_figure(ax, fill_mode="dist_outside")
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    g = Circle(centre, radius).inverse()
    g.add_to_figure(ax, fill_mode="dist_inside")
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    g = Circle(centre, radius).inverse()
    g.add_to_figure(ax, fill_mode="dist_outside")
    plt.show()
    plt.close(fig)
