import numpy as np
import matplotlib.pyplot as plt

from .circle import Circle
from .polygon import ClosedPolygon
from .area_geometry import AreaUnion

if __name__ == "__main__":

    centres = [np.array([7.,8.])]
    radii = [0.5]
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
    g = AreaUnion(circles + polygons)
    g.add_to_figure(ax, show_distance="inside")
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    g = AreaUnion(circles + polygons)
    g.add_to_figure(ax, show_distance="outside")
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    g = AreaUnion(circles + polygons).inverse()
    g.add_to_figure(ax, show_distance="inside")
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    g = AreaUnion(circles + polygons).inverse()
    g.add_to_figure(ax, show_distance="outside")
    plt.show()
    plt.close(fig)

    fig, ax = plt.subplots()
    g = polygons[0] + circles[0]
    g.add_to_figure(ax, show_distance="inside")
    plt.show()
    plt.close(fig)
