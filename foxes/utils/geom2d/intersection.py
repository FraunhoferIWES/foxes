import numpy as np

from .area_geometry import AreaGeometry
from .union import AreaUnion

class AreaIntersection(AreaGeometry):
    """
    The intersection of area geometries.

    Parameters
    ----------
    geometries : list of geom2d.AreaGeometry
        The geometries

    """

    def __new__(cls, geometries):
        return AreaUnion([g.inverse() for g in geometries]).inverse()
    

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
