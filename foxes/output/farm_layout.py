import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import foxes.constants as FC
import foxes.variables as FV
from foxes.output.output import Output


class FarmLayoutOutput(Output):
    """
    Plot the farm layout

    Attributes
    ----------
    farm: foxes.WindFarm
        The wind farm
    fres: xarray.Dataset
        The wind farm calculation results
    from_res: bool
        Flag for coordinates from results data
    results_state: int
        The state index, for from_res
    D: float
        The rotor diameter, if not from data

    :group: output

    """

    def __init__(
        self, farm, farm_results=None, from_results=False, results_state=None, D=None
    ):
        """
        Constructor.

        Parameters
        ----------
        farm: foxes.WindFarm
            The wind farm
        farm_results: xarray.Dataset, optional
            The wind farm calculation results
        from_results: bool, optional
            Flag for coordinates from results data
        results_state: int, optional
            The state index, for from_res
        D: float, optional
            The rotor diameter, if not from data

        """
        self.farm = farm
        self.fres = farm_results
        self.from_res = from_results
        self.rstate = results_state
        self.D = D

        if from_results and farm_results is None:
            raise ValueError(f"Missing farm_results for switch from_results.")

        if from_results and results_state is None:
            raise ValueError(f"Please specify results_state for switch from_results.")

    def get_layout_data(self):
        """
        Returns wind farm layout.

        Returns
        -------
        numpy.ndarray :
            The wind farm layout, shape:
            (n_turbines, 3) where the 3
            represents x, y, h

        """

        data = np.zeros([self.farm.n_turbines, 3], dtype=FC.DTYPE)

        if self.from_res:
            data[:, 0] = self.fres[FV.X][self.rstate]
            data[:, 1] = self.fres[FV.Y][self.rstate]
            data[:, 2] = self.fres[FV.H][self.rstate]

        else:
            for ti, t in enumerate(self.farm.turbines):
                data[ti, :2] = t.xy
                data[ti, 2] = t.H

        return data

    def get_layout_dict(self):
        """
        Returns wind farm layout.

        Returns
        -------
        dict :
            The wind farm layout in dict
            format, as in json output

        """

        data = self.get_layout_data()

        out = {self.farm.name: {}}
        for ti, p in enumerate(data):
            t = self.farm.turbines[ti]
            out[self.farm.name][t.name] = {
                "id": t.index,
                "name": t.name,
                "UTMX": p[0],
                "UTMY": p[1],
            }

        return out

    def get_figure(
        self,
        color_by=None,
        fontsize=8,
        figsize=None,
        annotate=1,
        title=None,
        fig=None,
        ax=None,
        normalize_D=False,
        ret_im=False,
        bargs={},
        anno_delx=0,
        anno_dely=0,
        **kwargs,
    ):
        """
        Creates farm layout figure.

        Parameters
        ----------
        color_by: str, optional
            Set scatter color by variable results.
            Use "mean_REWS" etc for means, also
            min, max, sum. All wrt states
        fontsize: int, optional
            Size of the turbine numbers
        figsize: tuple, optional
            The figsize for plt.Figure
        annotate: int, optional
            Turbine index printing, Choices:
            0 = No annotation
            1 = Turbine indices
            2 = Turbine names
        title: str, optional
            The plot title, or None for automatic
        fig: matplotlib.pyplot.Figure, optional
            The figure object to which to add
        ax: matplotlib.pyplot.Axis, optional
            The axis object, to which to add
        normalize_D: bool
            Normalize x, y wrt rotor diameter
        ret_im: bool
            Flag for returned image object
        bargs: dict
            Arguments for boundary plotting
        anno_delx: float
            The annotation delta x
        anno_dely: float
            The annotation delta y
        kwargs: dict, optional
            Parameters forwarded to `matplotlib.pyplot.scatter`

        Returns
        -------
        ax: matplotlib.pyplot.Axis
            The axis object
        im: matplotlib.pyplot.PathCollection, optional
            The image object

        """

        if fig is None:
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        else:
            ax = fig.axes[0] if ax is None else ax

        D = self.D
        x = None
        if self.farm.n_turbines:
            if normalize_D and D is None:
                if self.from_res:
                    if self.fres[FV.D].min() != self.fres[FV.D].max():
                        raise ValueError(
                            f"Expecting uniform D, found {self.fres[FV.D]}"
                        )
                    D = self.fres[FV.D][0]
                else:
                    D = None
                    for ti, t in enumerate(self.farm.turbines):
                        hD = t.D
                        if D is None:
                            D = hD
                        elif D != hD:
                            raise ValueError(
                                f"Turbine {ti} has wrong rotor diameter, expecting D = {D} m, found D = {hD} m"
                            )
                    if D is None:
                        raise ValueError(
                            f"Variable '{FV.D}' not found in turbines. Maybe set explicitely, or try from_results?"
                        )

            data = self.get_layout_data()
            x = data[:, 0] / D if normalize_D else data[:, 0]
            y = data[:, 1] / D if normalize_D else data[:, 1]
            n = range(len(x))

            kw = {"c": "orange"}
            kw.update(**kwargs)

            if color_by is not None:
                if self.fres is None:
                    raise ValueError(f"Missing farm_results for color_by '{color_by}'")
                if color_by[:5] == "mean_":
                    kw["c"] = np.einsum(
                        "st,st->t", self.fres[color_by[5:]], self.fres[FV.WEIGHT]
                    )
                elif color_by[:4] == "sum_":
                    kw["c"] = np.sum(self.fres[color_by[4:]], axis=0)
                elif color_by[:4] == "min_":
                    kw["c"] = np.min(self.fres[color_by[4:]], axis=0)
                elif color_by[:4] == "max_":
                    kw["c"] = np.max(self.fres[color_by[4:]], axis=0)
                else:
                    raise KeyError(
                        f"Unknown color_by '{color_by}'. Choose: mean_X, sum_X, min_X, max_X, where X is a farm_results variable"
                    )

            im = ax.scatter(x, y, **kw)

            if annotate == 1:
                for i, txt in enumerate(n):
                    ax.annotate(
                        int(txt), (x[i] + anno_delx, y[i] + anno_dely), size=fontsize
                    )
            elif annotate == 2:
                for i, t in enumerate(self.farm.turbines):
                    ax.annotate(
                        t.name, (x[i] + anno_delx, y[i] + anno_dely), size=fontsize
                    )

        if self.farm.boundary is not None:
            hbargs = {"fill_mode": "inside_lightgray"}
            hbargs.update(bargs)
            self.farm.boundary.add_to_figure(ax, **hbargs)

        ti = (
            title
            if title is not None
            else (
                self.farm.name
                if D is None or not normalize_D
                else f"{self.farm.name} (D = {D} m)"
            )
        )
        ax.set_title(ti)

        ax.set_xlabel("x [m]" if not normalize_D else "x [D]")
        ax.set_ylabel("y [m]" if not normalize_D else "y [D]")
        ax.grid()

        # if len(self.farm.boundary_geometry) \
        #    or ( min(x) != max(x) and min(y) != max(y) ):
        if x is None or (min(x) != max(x) and min(y) != max(y)):
            ax.set_aspect("equal", adjustable="box")

        ax.autoscale_view(tight=True)

        if color_by is not None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)

        if ret_im:
            return ax, im

        return ax

    def write_plot(self, file_path=None, fontsize=8, **kwargs):
        """
        Writes the layout plot to file.

        The kwargs are forwarded to self.get_figure

        Parameters
        ----------
        file_path: str
            The file into which to plot, or None
            for default
        fontsize: int
            Size of the turbine numbers

        """

        ax = self.get_figure(fontsize=fontsize, ret_im=False, **kwargs)
        fig = ax.get_figure()

        fname = file_path if file_path is not None else self.farm.name + ".png"
        fig.savefig(fname, bbox_inches="tight")

        plt.close(fig)

    def write_xyh(self, file_path=None):
        """
        Writes xyh layout file.

        Parameters
        ----------
        file_path: str
            The file into which to plot, or None
            for default

        """

        data = self.get_layout_data()

        fname = file_path if file_path is not None else self.farm.name + ".xyh"
        np.savetxt(fname, data, header="x y h")

    def write_csv(self, file_path=None):
        """
        Writes csv layout file.

        Parameters
        ----------
        file_path: str
            The file into which to plot, or None
            for default

        """

        data = self.get_layout_data()

        fname = file_path if file_path is not None else self.farm.name + ".csv"

        lyt = pd.DataFrame(
            index=range(len(data)), columns=["id", "name", "x", "y", "h", "D"]
        )
        lyt.index.name = "index"
        lyt["id"] = [t.info["id"] for t in self.farm.turbines]
        lyt["name"] = [t.info["name"] for t in self.farm.turbines]
        lyt["x"] = data[:, 0]
        lyt["y"] = data[:, 1]
        lyt["h"] = data[:, 2]
        lyt["D"] = [t.D for t in self.farm.turbines]

        lyt.to_csv(fname)

    def write_json(self, file_path=None):
        """
        Writes xyh layout file.

        Parameters
        ----------
        file_path: str
            The file into which to plot, or None
            for default

        """

        data = self.get_layout_dict()

        fname = file_path if file_path is not None else self.farm.name + ".json"
        with open(fname, "w") as outfile:
            json.dump(data, outfile, indent=4)
