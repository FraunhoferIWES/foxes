import matplotlib.pyplot as plt
import numpy as np

import foxes.constants as FC

from .output import Output


class StateTurbineMap(Output):
    """
    Creates heat maps with turbines on one axis
    and states on the other axis.

    Attributes
    ----------
    results: xarray.Dataset
        The farm results

    :group: output

    """

    def __init__(self, farm_results, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        farm_results: xarray.Dataset
            The farm results
        kwargs: dict, optional
            Additional parameters for the base class

        """
        super().__init__(**kwargs)
        self.results = farm_results

    def plot_map(
        self,
        variable,
        title=None,
        ax=None,
        figsize=None,
        rotate_xlabels=None,
        **kwargs,
    ):
        """
        Plot the heat map for the selected variable.

        Parameters
        ----------
        variable: str
            The variable to plot
        title: str, optional
            The plot title
        ax: pyplot.Axis, optional
            The axis
        figsize: tuple
            The figsize argument for plt.subplots()
            in case ax is not provided
        rotate_xlabels: float, optional
            Rotate the x-labels by this number of degrees
        kwargs: dict, optional
            Additional parameters for plt.pcolormesh()

        Returns
        -------
        ax: pyplot.Axis
            The plot axis

        """
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
        xt = ax.get_xticks()
        xtl = ax.get_xticklabels()
        ax.set_xticks(
            xt[:-1] + 0.5 * (xt[-1] - xt[-2]), xtl[:-1], rotation=rotate_xlabels
        )
        if len(turbines) > 10:
            yt = ax.get_yticks()
            ytl = [None for t in yt]
            ytl[::5] = ax.get_yticklabels()[::5]
            ax.set_yticks(yt, ytl)
        fig.colorbar(c, ax=ax)

        t = title if title is not None else variable
        ax.set_title(t)
        ax.set_ylabel("Turbine index")
        ax.set_xlabel("State")

        return ax
