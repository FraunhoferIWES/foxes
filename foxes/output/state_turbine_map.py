import matplotlib.pyplot as plt
import numpy as np

import foxes.variables as FV
from .output import Output


class StateTurbineMap(Output):
    """
    Creates heat maps with turbines on the one
    and states on the other axis.

    Parameters
    ----------
    farm_results : xarray.Dataset
        The farm results

    Attributes
    ----------
    results : xarray.Dataset
        The farm results

    """

    def __init__(self, farm_results):
        self.results = farm_results

    def plot_map(
        self,
        variable,
        title=None,
        ax=None,
        figsize=None,
        **kwargs,
    ):
        """
        Plot the heat map for the selected variable.

        Parameters
        ----------
        variable : str
            The variable to plot
        title : str, optional
            The plot title
        ax : pyplot.Axis, optional
            The axis
        figsize : tuple
            The figsize argument for plt.subplots()
            in case ax is not provided
        kwargs : dict, optional
            Additional parameters for plt.pcolormesh()

        Returns
        -------
        ax : pyplot.Axis
            The plot axis

        """
        turbines = self.results[FV.TURBINE].to_numpy()
        states = self.results[FV.STATE].to_numpy()

        if ax is None:
            __, ax = plt.subplots(figsize=figsize)
        fig = ax.get_figure()

        ds = states[-1] - states[-2]
        states = np.append(states, states[-1] + ds)
        turbines = np.arange(len(turbines) + 1)

        y, x = np.meshgrid(states, turbines)
        z = self.results[variable].to_numpy()

        prgs = {"shading": "flat"}
        prgs.update(kwargs)

        c = ax.pcolormesh(x, y, z.T, **prgs)

        ax.set_xticks(turbines[:-1] + 0.5)
        ax.set_xticklabels(turbines[:-1])
        yt = ax.get_yticks()
        ytl = ax.get_yticklabels()
        ax.set_yticks(yt[:-1] + 0.5 * (yt[-1] - yt[-2]))
        ax.set_yticklabels(ytl[:-1])
        fig.colorbar(c, ax=ax)

        t = title if title is not None else variable
        ax.set_title(t)
        ax.set_xlabel("Turbine index")
        ax.set_ylabel("State")

        return ax
