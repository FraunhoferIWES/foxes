import numpy as np

from abc import ABCMeta, abstractmethod

class ClosedGeometry2D(metaclass=ABCMeta):
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
        p_nearest : numpy.ndarray
            The nearest points on the boundary, if
            return_nearest is True, shape: (n_points, 2)
            
        """
        pass

    @abstractmethod
    def points_inside(self, points, min_dist=None):
        """
        Tests if points are inside the geometry.

        Parameters
        ----------
        points : numpy.ndarray
            The probe points, shape (n_points, 2)
        min_dist : float, optional
            Minimal distance to boundary
        
        Returns
        -------
        inside : numpy.ndarray
            True if point is inside (and has
            distance not below min_dist, if
            given), shape: (n_points,)
        
        """
        pass

    @abstractmethod
    def add_to_figure(self, ax, **kwargs):
        """
        Add boundary to (x,y) figure.

        Parameters
        ----------
        ax : matplotlib.pyplot.Axis
            The axis object
        
        """
        pass
