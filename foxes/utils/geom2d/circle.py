import numpy as np
import matplotlib.pyplot as plt

from .area_geometry import AreaGeometry


class Circle(AreaGeometry):
    """
    This class represents the area of a circle.

    Attributes
    ----------
    centre: numpy.ndarray
        The centre point, shape: (2,)
    radius: float
        The radius

    :group: utils.geom2d

    """

    def __init__(self, centre, radius):
        """
        Cobnstructor.

        Parameters
        ----------
        centre: numpy.ndarray
            The centre point, shape: (2,)
        radius: float
            The radius

        """
        self.centre = np.array(centre, dtype=np.float64)
        self.radius = radius

    def p_min(self):
        """
        Returns minimal (x,y) point.

        Returns
        -------
        p_min: numpy.ndarray
            The minimal (x,y) point, shape = (2,)

        """
        return self.centre - self.radius

    def p_max(self):
        """
        Returns maximal (x,y) point.

        Returns
        -------
        p_min: numpy.ndarray
            The maximal (x,y) point, shape = (2,)

        """
        return self.centre + self.radius

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
        magd = np.linalg.norm(points - self.centre[None, :], axis=-1)
        return magd <= self.radius

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
            pars = dict(color="darkblue", linewidth=1, fill=False)
            pars.update(pars_boundary)

            circle = plt.Circle(self.centre, self.radius, **pars)
            ax.add_patch(circle)

        super().add_to_figure(
            ax, show_boundary, fill_mode, pars_boundary, pars_distance
        )


if __name__ == "__main__":
    centre = np.array([3.0, 4.0])
    radius = 2.5
    N = 500

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
