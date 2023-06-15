import numpy as np
import matplotlib.pyplot as plt

from .circle import Circle
from .polygon import ClosedPolygon
from .area_geometry import AreaUnion

if __name__ == "__main__":
    boundary = (
        ClosedPolygon(
            np.array([[0, 0], [0, 1200], [1000, 800], [900, -200]], dtype=np.float64)
        )
        + ClosedPolygon(
            np.array([[500, 0], [500, 1500], [1000, 1500], [1000, 0]], dtype=np.float64)
        )
        - Circle([-100.0, -100.0], 700)
    )

    fig, ax = plt.subplots()
    boundary.add_to_figure(ax, fill_mode="dist_inside")
    plt.show()
    plt.close(fig)
    quit()

    centres = [np.array([7.0, 8.0])]
    radii = [0.5]
    N = 500

    plist = [
        np.array([[1.0, 1.0], [1.3, 6], [5.8, 6.2], [6.5, 0.8]]),
        np.array([[1.5, 1.5], [1.5, 8], [2.5, 8], [2.5, 1.5]]),
    ]

    circles = [Circle(centres[i], radii[i]) for i in range(len(centres))]
    polygons = [ClosedPolygon(plist[i]) for i in range(len(plist))]

    fig, ax = plt.subplots()
    g = AreaUnion(circles + polygons)
    g.add_to_figure(ax, fill_mode="dist_inside")
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    g = AreaUnion(circles + polygons)
    g.add_to_figure(ax, fill_mode="dist_outside")
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    g = AreaUnion(circles + polygons).inverse()
    g.add_to_figure(ax, fill_mode="dist_inside")
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    g = AreaUnion(circles + polygons).inverse()
    g.add_to_figure(ax, fill_mode="dist_outside")
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    g = polygons[0] + circles[0]
    g.add_to_figure(ax, fill_mode="dist_inside")
    plt.show()
    plt.close(fig)
