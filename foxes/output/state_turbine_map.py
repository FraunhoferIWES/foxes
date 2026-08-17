# mypy: disable-error-code=arg-type
# mypy: disable-error-code=misc
# mypy: disable-error-code=union-attr

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from xarray import Dataset
from typing import Any

import foxes.constants as FC

from .output import Output


class StateTurbineMap(Output):
    """
    Creates heat maps with turbines on one axis
    and states on the other axis.

    Attributes
    ----------
    results
        The farm results

    :group: output

    """

    def __init__(self, farm_results: Dataset, **kwargs: Any) -> None:
        """
        Constructor.

        Parameters
        ----------
        farm_results
            The farm results
        kwargs
            Additional parameters for the base class

        """
        super().__init__(**kwargs)
        self.results = farm_results

    def plot_map(
        self,
        variable: str,
        title: str | None = None,
        cbar_label: str | None = None,
        ax: Axes | None = None,
        figsize: tuple[int, int] | None = None,
        rotate_xlabels: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        Plot the heat map for the selected variable.

        Parameters
        ----------
        variable
            The variable to plot
        title
            The plot title
        ax
            The axis
        figsize
            The figsize argument for plt.subplots()
            in case ax is not provided
        rotate_xlabels
            Rotate the x-labels by this number of degrees
        kwargs
            Additional parameters for plt.pcolormesh()

        Returns
        -------
        ax
            The plot axis

        """
        if self.nofig:
            return None

        turbines = self.results[FC.TURBINE].to_numpy()
        states = self.results[FC.STATE].to_numpy()

        if ax is None:
            __, ax = plt.subplots(figsize=figsize)
        fig = ax.get_figure()

        ds = states[-1] - states[-2]
        states = np.append(states, states[-1] + ds)
        turbines = np.arange(len(turbines) + 1)

        y, x = np.meshgrid(turbines, states)
        z = self.results[variable].to_numpy()

        prgs = {"shading": "flat"}
        prgs.update(kwargs)

        c = ax.pcolormesh(x, y, z, **prgs)

        ax.set_yticks(turbines[:-1] + 0.5)
        ax.set_yticklabels(turbines[:-1])
        xt = np.asarray(ax.get_xticks())
        xtl = ax.get_xticklabels()
        try:
            xt, ar = np.unique(xt.astype(int), return_index=True)
            xtl = [int(float(xtl[i].get_text())) for i in ar]
        except ValueError:
            pass
        ax.set_xticks(
            xt[:-1] + 0.5 * (xt[-1] - xt[-2]), xtl[:-1], rotation=rotate_xlabels
        )
        if len(turbines) > 10:
            yt = ax.get_yticks()
            ytl: list[Any] = [None for t in yt]
            ytl[::5] = ax.get_yticklabels()[::5]
            ax.set_yticks(yt, ytl)
        fig.colorbar(c, ax=ax, label=cbar_label)

        ax.set_title(title)
        ax.set_ylabel("Turbine index")
        ax.set_xlabel("State")

        return ax
