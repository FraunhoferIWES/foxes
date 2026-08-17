from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from collections.abc import Sequence
from typing import Any

from .area_geometry import AreaGeometry


class HalfPlane(AreaGeometry):
    """
    This class represents a half plane in 2d.

    Attributes
    ----------
    centre
        The centre point, shape: (2,)
    n
        The direction vector to the inside, shape: (2,)
    n
        The direction vector orthogonal to n, shape: (2,)

    :group: utils.geom2d

    """

    def __init__(
        self, centre: np.ndarray | Sequence[float], n: np.ndarray | Sequence[float]
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        centre
            The centre point, shape: (2,)
        n
            The direction vector to the inside, shape: (2,)

        """
        self.centre = np.array(centre, dtype=np.float64)
        self.n = np.array(n, dtype=np.float64)
        self.n /= np.linalg.norm(self.n)
        self.m = np.array([-self.n[1], self.n[0]], dtype=np.float64)

    def p_min(self) -> np.ndarray:
        """
        Returns minimal (x,y) point.

        Returns
        -------
        p_min
            The minimal (x,y) point, shape = (2,)

        """
        if np.linalg.norm(self.n - np.array([1, 0])) < 1e-13:
            return np.array([self.centre[0], -np.inf], dtype=np.float64)
        if np.linalg.norm(self.n - np.array([-1, 0])) < 1e-13:
            return np.array([-np.inf, -np.inf], dtype=np.float64)
        if np.linalg.norm(self.n - np.array([0, 1])) < 1e-13:
            return np.array([-np.inf, self.centre[1]], dtype=np.float64)
        if np.linalg.norm(self.n - np.array([0, -1])) < 1e-13:
            return np.array([-np.inf, -np.inf], dtype=np.float64)

        return np.array([-np.inf, -np.inf], dtype=np.float64)

    def p_max(self) -> np.ndarray:
        """
        Returns maximal (x,y) point.

        Returns
        -------
        p_min
            The maximal (x,y) point, shape = (2,)

        """
        if np.linalg.norm(self.n - np.array([1, 0])) < 1e-13:
            return np.array([np.inf, np.inf], dtype=np.float64)
        if np.linalg.norm(self.n - np.array([-1, 0])) < 1e-13:
            return np.array([self.centre[0], np.inf], dtype=np.float64)
        if np.linalg.norm(self.n - np.array([0, 1])) < 1e-13:
            return np.array([np.inf, np.inf], dtype=np.float64)
        if np.linalg.norm(self.n - np.array([0, -1])) < 1e-13:
            return np.array([np.inf, self.centre[1]], dtype=np.float64)

        return np.array([np.inf, np.inf], dtype=np.float64)

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
        x = np.einsum("pd,d->p", deltas, self.n)

        if return_nearest:
            y = np.einsum("pd,d->p", deltas, self.m)
            nerst = self.centre[None, :] + y[:, None] * self.m[None, :]
            return np.abs(x), nerst
        else:
            return np.abs(x)

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
        deltas = points - self.centre[None, :]
        x = np.einsum("pd,d->p", deltas, self.n)
        return x >= 0.0

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
            pars = dict(color="darkblue", linewidth=1)
            pars.update(pars_boundary)

            ax.axline(tuple(self.centre), tuple(self.centre + self.m), **pars)

        super().add_to_figure(
            ax, show_boundary, fill_mode, pars_boundary, pars_distance
        )

    def inverse(self) -> HalfPlane:
        """
        Get the inverted geometry

        Returns
        -------
        inverted
            The inverted geometry

        """
        return HalfPlane(self.centre, -self.n)


if __name__ == "__main__":
    from .circle import Circle

    p0 = [4, 5]
    n = [1.0, 0.3]

    centre = np.array([3.0, 4.0])
    radius = 2.5
    N = 500
    g: AreaGeometry

    fig, ax = plt.subplots()
    g = Circle(centre, radius) + HalfPlane(p0, n)
    g.add_to_figure(ax, fill_mode="dist_inside")
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    g.add_to_figure(ax, fill_mode="dist_outside")
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    g = Circle(centre, radius) - HalfPlane(p0, n)
    g.add_to_figure(ax, fill_mode="dist_inside")
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    g.add_to_figure(ax, fill_mode="dist_outside")
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    g = g.inverse()
    g.add_to_figure(ax, fill_mode="dist_inside")
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    g.add_to_figure(ax, fill_mode="dist_outside")
    plt.show()
    plt.close(fig)
