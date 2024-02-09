import pandas as pd
import matplotlib.pyplot as plt 
from windrose import WindroseAxes

FIGSIZE_DEFAULT = (8, 8)
DPI_DEFAULT = 80

class StochasticWindroseAxes(WindroseAxes):
    """
    A derivate of the wind rose axes that runs
    on stochastic data instead of time series data

    :group: utils

    """

    @staticmethod
    def from_ax(
        ax=None,
        fig=None,
        rmax=None,
        figsize=FIGSIZE_DEFAULT,
        rect=None,
        *args,
        **kwargs,
    ):
        """
        Return a WindroseAxes object for the figure `fig`.
        """
        if ax is None:
            if fig is None:
                fig = plt.figure(
                    figsize=figsize,
                    dpi=DPI_DEFAULT,
                    facecolor="w",
                    edgecolor="w",
                )
            if rect is None:
                rect = [0.1, 0.1, 0.8, 0.8]
            ax = StochasticWindroseAxes(fig, rect, *args, **kwargs)
            fig.add_axes(ax)
            return ax
        else:
            return ax
        
    def _init_plot(self, direction, var, **kwargs):

        bins, nbins, nsector, colors, angles, kwargs = super()._init_plot(
             direction, var, **kwargs
        )

        angle = 360.0 / nsector
        dir_bins = np.arange(-angle / 2, 360.0 + angle, angle, dtype=float)
        dir_edges = dir_bins.tolist()
        dir_edges.pop(-1)
        dir_edges[0] = dir_edges.pop(-1)
        dir_bins[0] = 0.0

        table = np.histogram2d(x=var, y=direction, 
                bins=[self._info["bins"], dir_bins], 
                density=False, weights=self._weights)[0]
        table[:, 0] = table[:, 0] + table[:, -1]
        table = table[:, :-1]

        self._info["table"] = table*100

        return bins, nbins, nsector, colors, angles, kwargs

    def bar(self, direction, var, weights, **kwargs):
        self._weights = weights
        return super().bar(direction, var, **kwargs)
        
    def contourf(self, direction, var, weights, **kwargs):
        self._weights = weights
        return super().contourf(direction, var, **kwargs)

    def contour(self, direction, var, weights, **kwargs):
        self._weights = weights
        return super().contour(direction, var, **kwargs)

if __name__ == "__main__":

    import numpy as np
    import pandas as pd
    from foxes import StaticData, STATES

    sdata = StaticData()
    fpath = sdata.get_file_path(STATES, "wind_rose_bremen.csv")
    print(fpath)
    data = pd.read_csv(fpath, index_col=0)
    print(data)

    wd = data["wd"].to_numpy()
    ws = data["ws"].to_numpy()
    weights = data["weight"].to_numpy()

    ax = StochasticWindroseAxes.from_ax()
    print(type(ax))
    #ax.contourf(wd, ws, weights, bins=[0,3,8,13], cmap=plt.cm.Blues)
    #ax.contour(wd, ws, weights, bins=[0,3,8,13], colors='black')
    ax.bar(wd, ws, weights, bins=[0,3,5,8,10,13,16,20], cmap=plt.cm.Blues)
    ax.set_legend()
    plt.show()
