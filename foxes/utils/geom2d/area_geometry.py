import numpy as np
from abc import ABCMeta, abstractmethod
import matplotlib.pyplot as plt

class AreaGeometry(metaclass=ABCMeta):
    """
    Abstract base class for closed 2D geometries.
    
    """
    
    @abstractmethod
    def p_min(self):
        """
        Returns minimal (x,y) point.

        Returns
        -------
        p_min : numpy.ndarray
            The minimal (x,y) point, shape = (2,)
        
        """
        pass

    @abstractmethod
    def p_max(self):
        """
        Returns maximal (x,y) point.

        Returns
        -------
        p_min : numpy.ndarray
            The maximal (x,y) point, shape = (2,)
        
        """
        pass

    @abstractmethod
    def points_distance(self, points, return_nearest=False):
        """
        Calculates point distances wrt boundary.

        Parameters
        ----------
        points : numpy.ndarray
            The probe points, shape (n_points, 2)
        return_nearest : bool
            Flag for return of the nearest point on bundary
        
        Returns
        -------
        dist : numpy.ndarray
            The smallest distances to the boundary,
            shape: (n_points,)
        p_nearest : numpy.ndarray, optional
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
        points : numpy.ndarray
            The probe points, shape (n_points, 2)
        
        Returns
        -------
        inside : numpy.ndarray
            True if point is inside, shape: (n_points,)
        
        """
        pass

    def add_to_figure(
            self, 
            ax, 
            show_boundary=True, 
            show_distance=None, 
            pars_boundary={},
            pars_distance={}
        ):
        """
        Add image to (x,y) figure.

        Parameters
        ----------
        ax : matplotlib.pyplot.Axis
            The axis object
        show_boundary : bool
            Add the boundary line to the image
        show_distance : str, optional
            Add distances to image. Options:
            all, inside, outside
        pars_boundary : dict
            Parameters for boundary plotting command
        pars_distance : dict
            Parameters for distance plotting command
        
        """
        if show_distance is not None:

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
            pts = pts.reshape(Nx*Ny, 2)

            if show_distance == "all":
                dists = self.points_distance(pts).reshape(Nx, Ny)
            elif show_distance == "inside":
                ins = self.points_inside(pts)
                dists = np.full(Nx*Ny, np.nan, dtype=np.float64)
                dists[ins] = self.points_distance(pts[ins])
                dists = dists.reshape(Nx, Ny)
            elif show_distance == "outside":
                ins = self.points_inside(pts)
                dists = np.full(Nx*Ny, np.nan, dtype=np.float64)
                dists[~ins] = self.points_distance(pts[~ins])
                dists = dists.reshape(Nx, Ny)
            else:
                raise ValueError(f"Illegal parameter 'show_distance = {show_distance}', expecting: None, all, inside, outside")

            pars = dict(shading="auto", cmap="magma_r", zorder=-100)
            pars.update(pars_distance)
            im = ax.pcolormesh(x, y, dists.T, **pars)
            plt.colorbar(im, ax=ax, orientation="vertical", label="distance")

        ax.autoscale_view()
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="box")

    def inverse(self):
        """
        Get the inverted geometry

        Returns
        -------
        inverted : foxes.utils.geom2d.InvertedAreaGeometry
            The inverted geometry

        """
        return InvertedAreaGeometry(self)

class InvertedAreaGeometry(AreaGeometry):
    """
    Base class for inverted geometries.

    Parameters
    ----------
    geometry : geom2d.AreaGeometry
        The original geometry

    """

    def __init__(self, geometry):
        self._geometry = geometry

    def p_min(self):
        """
        Returns minimal (x,y) point.

        Returns
        -------
        p_min : numpy.ndarray
            The minimal (x,y) point, shape = (2,)
        
        """
        pmi = self._geometry.p_min()
        if not np.any(np.isinf(pmi)):
            return np.full(2, -np.inf, dtype=np.float64)
        else:
            out = np.full(2, np.inf, dtype=np.float64)
            imi = self._geometry.inverse().p_min()
            for di in range(2):
                if np.isinf(pmi[di]) and not np.isinf(imi[di]):
                    out[di] = np.minimum(out[di], imi[di])
                if not np.isinf(pmi[di]):
                    out[di] = -np.inf
            return out

    def p_max(self):
        """
        Returns maximal (x,y) point.

        Returns
        -------
        p_min : numpy.ndarray
            The maximal (x,y) point, shape = (2,)
        
        """
        pma = self._geometry.p_max()
        if not np.any(np.isinf(pma)):
            return np.full(2, np.inf, dtype=np.float64)
        else:
            out = np.full(2, -np.inf, dtype=np.float64)
            ima = self._geometry.inverse().p_max()
            for di in range(2):
                if np.isinf(pma[di]) and not np.isinf(ima[di]):
                    out[di] = np.maximum(out[di], ima[di])
                if not np.isinf(pma[di]):
                    out[di] = np.inf
            return out

    def points_distance(self, points, return_nearest=False):
        """
        Calculates point distances wrt boundary.

        Parameters
        ----------
        points : numpy.ndarray
            The probe points, shape (n_points, 2)
        return_nearest : bool
            Flag for return of the nearest point on bundary
        
        Returns
        -------
        dist : numpy.ndarray
            The smallest distances to the boundary,
            shape: (n_points,)
        p_nearest : numpy.ndarray, optional
            The nearest points on the boundary, if
            return_nearest is True, shape: (n_points, 2)
            
        """
        return self._geometry.points_distance(points, return_nearest)

    def points_inside(self, points):
        """
        Tests if points are inside the geometry.

        Parameters
        ----------
        points : numpy.ndarray
            The probe points, shape (n_points, 2)
        
        Returns
        -------
        inside : numpy.ndarray
            True if point is inside, shape: (n_points,)
        
        """
        return ~self._geometry.points_inside(points)

    def add_to_figure(
            self, 
            ax, 
            show_boundary=True, 
            show_distance=None, 
            pars_boundary={},
            pars_distance={}
        ):
        """
        Add image to (x,y) figure.

        Parameters
        ----------
        ax : matplotlib.pyplot.Axis
            The axis object
        show_boundary : bool
            Add the boundary line to the image
        show_distance : str, optional
            Add distances to image. Options:
            all, inside, outside
        pars_boundary : dict
            Parameters for boundary plotting command
        pars_distance : dict
            Parameters for distance plotting command
        
        """
        self._geometry.add_to_figure(ax, show_boundary, show_distance=None, 
            pars_boundary=pars_boundary, pars_distance={})
        super().add_to_figure(ax, show_boundary, show_distance,
            pars_boundary, pars_distance)
            
    def inverse(self):
        """
        Get the inverted geometry

        Returns
        -------
        inverted : foxes.utils.geom2d.InvertedAreaGeometry
            The inverted geometry

        """
        return self._geometry
