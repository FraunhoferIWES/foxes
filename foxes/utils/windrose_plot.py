import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from windrose import WindroseAxes

FIGSIZE_DEFAULT = (8, 8)
DPI_DEFAULT = 80


class TabWindroseAxes(WindroseAxes):
    """
    A derivate of the wind rose axes that runs
    on stochastic data (bins with weights) instead
    of time series data

    :group: utils

    """

    @staticmethod
    def from_ax(
        ax=None,
        fig=None,
        figsize=None,
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
                    figsize=FIGSIZE_DEFAULT if figsize is None else figsize,
                    dpi=DPI_DEFAULT,
                    facecolor="w",
                    edgecolor="w",
                )
            if rect is None:
                rect = [0.1, 0.1, 0.8, 0.8]
            ax = TabWindroseAxes(fig, rect, *args, **kwargs)
            fig.add_axes(ax)
            return ax
        else:
            return ax

    def _init_plot(self, direction, var, **kwargs):

        # self.clear()
        kwargs.pop("zorder", None)
        weights = kwargs.pop("weights")

        # Init of the bins array if not set
        if "bin_min_var" in kwargs:
            bins = list(kwargs.pop("bin_min_var"))
            bins.append(max(kwargs.pop("bin_max_var")))
        else:
            bins = kwargs.pop("bins", None)
            if bins is None:
                bins = np.linspace(np.min(var), np.max(var), 6)
            if isinstance(bins, int):
                bins = np.linspace(np.min(var), np.max(var), bins)
            bins = np.asarray(bins).tolist()
        nbins = len(bins)
        bins.append(np.inf)

        # Sets the colors table based on the colormap or the "colors" argument
        colors = kwargs.pop("colors", None)
        cmap = kwargs.pop("cmap", None)
        if colors is not None:
            if isinstance(colors, str):
                colors = [colors] * nbins
            if isinstance(colors, (tuple, list)):
                if len(colors) != nbins:
                    raise ValueError("colors and bins must have same length")
        else:
            if cmap is None:
                cmap = plt.get_cmap()
            colors = self._colors(cmap, nbins)

        if "bin_min_dir" in kwargs:
            angles = 90 - np.sort(np.unique(direction))
            angles[angles > 180] -= 360
            angles *= np.pi / 180

            dir_min = kwargs.pop("bin_min_dir")
            dir_edges = np.mod(dir_min, 360.0).tolist()

            dir_bins = dir_min.copy().tolist()
            if dir_bins[0] < 0:
                dir_bins.append(360 + dir_bins[0])
                dir_bins[0] = 0
                dir_bins.append(360 + dir_bins[1])

            nsector = len(angles)

        else:
            nsector = kwargs.pop("nsector", None)
            if nsector is None:
                nsector = 16
            angles = np.arange(0, -2 * np.pi, -2 * np.pi / nsector) + np.pi / 2

            angle = 360.0 / nsector
            dir_bins = np.arange(-angle / 2, 360.0 + angle, angle, dtype=float)
            dir_edges = dir_bins.tolist()

            dir_edges.pop(-1)
            dir_edges[0] = dir_edges.pop(-1)
            dir_bins[0] = 0.0

        table = np.histogram2d(
            x=var, y=direction, bins=[bins, dir_bins], density=False, weights=weights
        )[0]
        table[:, 0] = table[:, 0] + table[:, -1]
        table = table[:, :-1]

        self._info["dir"], self._info["bins"], self._info["table"] = (
            dir_edges,
            bins,
            table,
        )
        return bins, nbins, nsector, colors, angles, kwargs

    def legend(self, loc="upper right", *args, **kwargs):
        return super().legend(loc, *args, **kwargs)


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

    ax = TabWindroseAxes.from_ax()
    # ax.contourf(wd, ws, weights=weights, bins=[0,3,8,13], cmap=plt.cm.Blues)
    # ax.contour(wd, ws, weights=weights, bins=[0,3,8,13], colors='black')
    ax.bar(
        wd, ws, weights=weights, bins=[0, 3, 5, 8, 10, 13, 16, 20], cmap=plt.cm.Blues
    )
    ax.set_legend()
    plt.show()
