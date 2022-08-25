from abc import ABCMeta, abstractmethod

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
        p_nearest : numpy.ndarray
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

    @abstractmethod
    def inverse(self):
        """
        Get the inverted geometry

        Returns
        -------
        inverted : foxes.utils.geom2d.InvertedAreaGeometry
            The inverted geometry

        """
        pass

class InvertedAreaGeometry(AreaGeometry):
    """
    Abstract base class for inverted geometries.

    Parameters
    ----------
    geometry : foxes.utils.geom2d.AreaGeometry
        The original geometry

    """

    def __init__(self, geometry):
        self._geometry = geometry

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

    def add_to_figure(self, ax, **kwargs):
        """
        Add boundary to (x,y) figure.

        Parameters
        ----------
        ax : matplotlib.pyplot.Axis
            The axis object
        
        """
        self._geometry.add_to_figure(ax, **kwargs)

    def inverse(self):
        """
        Get the inverted geometry

        Returns
        -------
        inverted : foxes.utils.geom2d.InvertedAreaGeometry
            The inverted geometry

        """
        return self._geometry
