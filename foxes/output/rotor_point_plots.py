# mypy: disable-error-code=arg-type

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import TYPE_CHECKING, Any

from foxes.input.states import SingleStateStates
from foxes.core import WindFarm
from foxes.algorithms import Downwind

from .output import Output

if TYPE_CHECKING:
    from foxes.core import Algorithm, RotorModel


class RotorPointPlot(Output):
    """
    Visualizes rotor points and their weights.
    """

    def __init__(
        self, rotor_model: RotorModel, algo: Algorithm | None = None, **kwargs: Any
    ) -> None:
        """
        Parameters
        ----------
        rotor_model
            The rotor model
        algo
            The algorithm
        kwargs
            Additional parameters for the base class
        """
        super().__init__(**kwargs)
        self.rotor_model = rotor_model
        self.algo = algo

        if self.algo is None:
            farm = WindFarm()
            states = SingleStateStates(ws=9, wd=270, ti=0.1, rho=1.225)
            self.algo = Downwind(farm, states, [])

    def get_point_figure(
        self,
        ax: Axes | None = None,
        fig: Figure | None = None,
        figsize: tuple[int, int] = (5, 5),
        title: str | None = None,
        cmap: str = "viridis_r",
        **kwargs: Any,
    ) -> Any:
        """
        Get a scatter plot of the rotor points.

        Parameters
        ----------
        ax
            The plot axes
        fig
            The figure object
        figsize
            The default figure size
        title
            The plot title
        cmap
            The colormap name
        kwargs
            Additional arguments for pyplot.scatter

        Returns
        -------
        ax
            The plot axes

        """
        if self.nofig:
            return None

        if fig is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        else:
            ax = fig.axes[0] if ax is None else ax

        if not self.rotor_model.initialized:
            self.rotor_model.initialize(self.algo)

        points = self.rotor_model.design_points()
        weights = self.rotor_model.rotor_point_weights() * 100

        cmap_obj = colormaps[cmap]
        wlist = np.sort(np.unique(weights))

        im = ax.scatter(points[:, 1], points[:, 2], c=weights, cmap=cmap_obj, **kwargs)
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
