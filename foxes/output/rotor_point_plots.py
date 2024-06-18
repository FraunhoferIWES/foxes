import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap

from foxes.input.states import SingleStateStates
from foxes.core import WindFarm
from foxes.algorithms import Downwind


class RotorPointPlot:
    """
    Visualizes rotor points and their weights.

    Attributes
    ----------
    rotor_model: foxes.core.RotorModel
        The rotor model
    algo: foxes.core.Algorithm, optional
        The algorithm

    :group: output

    """

    def __init__(self, rotor_model, algo=None):
        """
        Constructor.

        Parameters
        ----------
        rotor_model: foxes.core.RotorModel
            The rotor model
        algo: foxes.core.Algorithm, optional
            The algorithm

        """
        self.rotor_model = rotor_model
        self.algo = algo

        if self.algo is None:
            farm = WindFarm()
            states = SingleStateStates(ws=9, wd=270, ti=0.1, rho=1.225)
            self.algo = Downwind(farm, states, [])

    def get_point_figure(
        self,
        ax=None,
        fig=None,
        figsize=(5, 5),
        title=None,
        **kwargs,
    ):
        """
        Get a scatter plot of the rotor points.

        Parameters
        ----------
        ax: matplotlib.Axes, optional
            The plot axes
        fig: matplotlib.Figure, optional
            The figure object
        figsize: tuple
            The default figure size
        title: str, optional
            The plot title
        kwargs: dict, optional
            Additional arguments for pyplot.scatter

        Returns
        -------
        ax: matplotlib.Axes
            The plot axes

        """
        if fig is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        else:
            ax = fig.axes[0] if ax is None else ax

        if not self.rotor_model.initialized:
            self.rotor_model.initialize(self.algo)

        points = self.rotor_model.design_points()
        weights = self.rotor_model.rotor_point_weights() * 100

        cmap = colormaps[kwargs.pop("cmap", "viridis_r")]
        wlist = np.sort(np.unique(weights))

        im = ax.scatter(points[:, 1], points[:, 2], c=weights, cmap=cmap, **kwargs)
        ax.add_patch(plt.Circle((0, 0), 1, color="black", fill=False, alpha=0.8))

        ax.set_xlabel("x/D")
        ax.set_ylabel("y/D")
        if title is None:
            title = str(self.rotor_model)
        if title != "":
            ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")

        tlabels = [f"{wlist[-1]:.02f}"]
        for i in range(len(wlist) - 2, -1, -1):
            w = wlist[i]
            w1 = wlist[i + 1]
            if (w1 - w) / (wlist[-1] - wlist[0]) > 0.05:
                tlabels.insert(0, f"{w:.2f}")
            elif i == 0:
                tlabels[0] = ""
                tlabels.insert(0, f"{w:.2f}")
            else:
                tlabels.insert(0, "")

        cbar = fig.colorbar(im, ax=ax, label="Point weight [%]", shrink=0.8)
        cbar.set_ticks(ticks=wlist, labels=tlabels)

        return im
