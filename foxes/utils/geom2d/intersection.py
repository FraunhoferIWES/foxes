import numpy as np

from .area_geometry import AreaGeometry

class AreaIntersection(AreaGeometry):
    """
    The intersection of area geometries.

    Parameters
    ----------
    geometries : list of geom2d.AreaGeometry
        The geometries

    Attributes
    ----------
    geometries : list of geom2d.AreaGeometry
        The geometries

    """

    def __init__(self, geometries):
        self.geometries = geometries
    
    def p_min(self):
        """
        Returns minimal (x,y) point.

        Returns
        -------
        p_min : numpy.ndarray
            The minimal (x,y) point, shape = (2,)
        
        """
        out = None
        for g in self.geometries:
            pmi = g.p_min()
            if out is None:
                out = pmi
            else:
                out = np.maximum(out, pmi)
        return out

    def p_max(self):
        """
        Returns maximal (x,y) point.

        Returns
        -------
        p_min : numpy.ndarray
            The maximal (x,y) point, shape = (2,)
        
        """
        out = None
        for g in self.geometries:
            pma = g.p_max()
            if out is None:
                out = pma
            else:
                out = np.minimum(out, pma)
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
        if len(self.geometries) == 1:
            return self.geometries[0].points_distance(points, return_nearest)

        n_pts = len(points)
        dist = np.full(n_pts, np.inf, dtype=np.float64)
        nerst = np.zeros((n_pts, 2), dtype=np.float64) if return_nearest else None
        for g in self.geometries:

            res = g.points_distance(points, return_nearest)
            d = res[0] if return_nearest else res

            sel = d < dist
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
        points : numpy.ndarray
            The probe points, shape (n_points, 2)
        
        Returns
        -------
        inside : numpy.ndarray
            True if point is inside, shape: (n_points,)
        
        """
        if len(self.geometries) == 1:
            return self.geometries[0].points_inside(points)

        n_pts = len(points)
        inside = np.ones(n_pts, dtype=bool)
        for g in self.geometries:
            inside = inside & g.points_inside(points)
        return inside

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
        if show_boundary:
            for g in self.geometries:
                g.add_to_figure(ax, show_boundary=True, show_distance=None,
                    pars_boundary=pars_boundary, pars_distance={})

        super().add_to_figure(ax, show_boundary=False, show_distance=show_distance,
            pars_boundary={}, pars_distance=pars_distance)


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    from .circle import Circle
    from .polygon import ClosedPolygon

    centres = []
    radii = []
    N = 500

    plist = [
        np.array([
        [1.,1.],
        [1.3,6],
        [5.8,6.2],
        [6.5,0.8]]),
        np.array([
        [1.5,1.5],
        [1.5,8],
        [2.5,8],
        [2.5,1.5]]),
    ]

    circles = [Circle(centres[i], radii[i]) for i in range(len(centres))]
    polygons = [ClosedPolygon(plist[i]) for i in range(len(plist))]

    fig, ax = plt.subplots()
    g = AreaIntersection(circles + polygons)
    g.add_to_figure(ax, show_distance="inside")
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    g = AreaIntersection(circles + polygons)
    g.add_to_figure(ax, show_distance="outside")
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    g = AreaIntersection(circles + polygons).inverse()
    g.add_to_figure(ax, show_distance="inside")
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    g = AreaIntersection(circles + polygons).inverse()
    g.add_to_figure(ax, show_distance="outside")
    plt.show()
    plt.close(fig)
