import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from scipy.spatial.distance import cdist

from foxes.utils.geom2d.closed_geometry_2D import ClosedGeometry2D

class ClosedPolygon(ClosedGeometry2D):
    """
    This class represents a closed 2D polygon.

    Parameters
    ----------
    points : numpy.ndarray
        The polygon points, shape: (n_points, 2)

    Attributes
    ----------
    points : numpy.ndarray
        The polygon points
    poly : matplotlib.path.Path
        The closed polygon geometry

    """

    def __init__(self, points):

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
        p_min : numpy.ndarray
            The minimal (x,y) point, shape = (2,)
        
        """
        return np.min(self.points, axis=0)

    def p_max(self):
        """
        Returns maximal (x,y) point.

        Returns
        -------
        p_min : numpy.ndarray
            The maximal (x,y) point, shape = (2,)
        
        """
        return np.max(self.points, axis=0)

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

        dists = cdist(points, self.points[:-1])

        if return_nearest:
            mini  = np.argmin(dists, axis=1)
            dists = np.take_along_axis(dists, mini, axis=1)
            minp  = self.points[mini] 
        else:
            dists = np.min(dists, axis=1)

        for pi in range(len(self.points) - 1):

            pA = self.points[pi]
            pB = self.points[pi+1]
            n  = pB - pA
            d  = np.linalg.norm(n)

            if d > 0:

                n  /= d
                q   = points - pA[None, :]
                x   = np.einsum('pd,d->p', q, n)

                sel = (x > 0) & (x < d)
                if np.any(sel):

                    x  = x[sel]
                    y2 = np.linalg.norm(q[sel], axis=1)**2 - x**2

                    dsel       = dists[sel]
                    dists[sel] = np.minimum(dsel, np.sqrt(y2))

                    if return_nearest:
                        mini        = np.argwhere(np.sqrt(y2) < dsel)
                        hminp       = minp[sel]
                        hminp[mini] = pA[None, :] + x[mini, None] * n[None, :]
                        minp[sel]   = hminp
                        del mini, hminp
                    
                    del y2, dsel
                
                del x, sel
            
        if return_nearest:
            return dists, minp
        else:
            return dists

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
            given), shape: (n_points)
        
        """
        ok = self.poly.contains_points(points)

        if min_dist is not None:
            ptsok     = points[ok]
            dists     = np.full(len(points), np.inf)
            dists[ok] = self.points_distance(ptsok)
            ok        = ok & (dists >= min_dist)
        
        return ok
    
    def add_to_figure(self, ax, **kwargs):
        """
        Add boundary to (x,y) figure.

        Parameters
        ----------
        ax : matplotlib.pyplot.Axis
            The axis object
        
        """

        pars = dict(facecolor='none', edgecolor='darkblue', linewidth=1)
        pars.update(kwargs)

        pathpatch = PathPatch(self.poly, **pars)
        ax.add_patch(pathpatch)
