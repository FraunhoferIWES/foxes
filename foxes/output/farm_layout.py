# mypy: disable-error-code=arg-type
# mypy: disable-error-code=assignment
# mypy: disable-error-code=misc
# mypy: disable-error-code=union-attr

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from xarray import Dataset
from typing import TYPE_CHECKING, Any

from foxes.config import config
from foxes.output.output import Output
import foxes.variables as FV
import foxes.constants as FC

if TYPE_CHECKING:
    from foxes.core import Algorithm, WindFarm


class FarmLayoutOutput(Output):
    """
    Plot the farm layout
    """

    def __init__(
        self,
        farm: WindFarm,
        farm_results: Dataset | None = None,
        from_results: bool = False,
        results_state: int | None = None,
        D: float | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        farm
            The wind farm
        farm_results
            The wind farm calculation results
        from_results
            Flag for coordinates from results data
        results_state
            The state index, for from_res
        D
            The rotor diameter, if not from data
        kwargs
            Additional parameters for the base class
        """
        super().__init__(**kwargs)
        self.farm = farm
        self.fres = farm_results
        self.from_res = from_results
        self.rstate = results_state
        self.D = D

        if from_results and farm_results is None:
            raise ValueError("Missing farm_results for switch from_results.")

        if from_results and results_state is None:
            raise ValueError("Please specify results_state for switch from_results.")

    def get_layout_data(self, lonlat: bool = False) -> np.ndarray:
        """
        Returns wind farm layout.

        Parameters
        ----------
        lonlat
            Flag for lonlat coordinates, if available

        Returns
        -------
        Layout data:
            The wind farm layout, shape:
            (n_turbines, 3) where the 3
            represents x, y, h

        """

        data: np.ndarray = np.zeros(
            [self.farm.n_turbines, 3], dtype=config.dtype_double
        )

        if lonlat:
            if not self.farm.has_lonlat():
                raise ValueError(
                    f"WindFarm '{self.farm.name}': lonlat coordinates not available"
                )
            data[:, :2] = self.farm.lonlat
            data[:, 2] = [t.H for t in self.farm.turbines]

        elif self.from_res:
            assert self.fres is not None
            data[:, 0] = self.fres[FV.X][self.rstate]
            data[:, 1] = self.fres[FV.Y][self.rstate]
            data[:, 2] = self.fres[FV.H][self.rstate]

        else:
            for ti, t in enumerate(self.farm.turbines):
                data[ti, :2] = t.xy
                data[ti, 2] = t.H

        return data

    def get_layout_dict(self) -> dict[str, dict[str, dict[str, Any]]]:
        """
        Returns wind farm layout.

        Returns
        -------
        dict :
            The wind farm layout in dict
            format, as in json output

        """

        data = self.get_layout_data()

        out: dict[str, dict[str, dict[str, Any]]] = {self.farm.name: {}}
        for ti, p in enumerate(data):
            t = self.farm.turbines[ti]
            turbine_name = t.name or str(t.index)
            out[self.farm.name][turbine_name] = {
                "id": t.index,
                "name": t.name,
                "UTMX": p[0],
                "UTMY": p[1],
            }

        return out

    def get_figure(
        self,
        color_by: str | None = None,
        fontsize: int = 8,
        figsize: Any = None,
        annotate: int = 1,
        title: str | None = None,
        fig: Figure | None = None,
        ax: Axes | None = None,
        normalize_D: bool = False,
        ret_im: bool = False,
        bargs: dict[str, Any] | None = None,
        anno_delx: float = 0,
        anno_dely: float = 0,
        lonlat: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Creates farm layout figure.

        Parameters
        ----------
        color_by
            Set scatter color by variable results.
            Use "mean_REWS", etc, for means, also
            min, max, sum. All wrt states
        fontsize
            Size of the turbine numbers
        figsize
            The figsize for plt.Figure
        annotate
            Turbine index printing, Choices:
            0 = No annotation
            1 = Turbine indices
            2 = Turbine names
            3 = Wind farm names
        title
            The plot title, or None for automatic
        fig
            The figure object to which to add
        ax
            The axis object, to which to add
        normalize_D
            Normalize x, y wrt rotor diameter
        ret_im
            Flag for returned image object
        bargs
            Arguments for boundary plotting
        anno_delx
            The annotation delta x
        anno_dely
            The annotation delta y
        lonlat
            Flag for lonlat coordinates, if available
        kwargs
            Parameters forwarded to `matplotlib.pyplot.scatter`

        Returns
        -------
        ax
            The axis object
        im
            The image object

        """
        if self.nofig:
            return None, None

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
                    assert self.fres is not None
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

            data = self.get_layout_data(lonlat=lonlat)
            x = data[:, 0] / D if normalize_D and not lonlat else data[:, 0]
            y = data[:, 1] / D if normalize_D and not lonlat else data[:, 1]
            n = range(len(x))

            kw = {"c": "orange"}
            kw.update(**kwargs)

            if color_by is not None:
                if self.fres is None:
                    raise ValueError(f"Missing farm_results for color_by '{color_by}'")
                if color_by in self.fres and self.fres[color_by].dims == (FC.TURBINE,):
                    kw["c"] = self.fres[color_by]
                elif color_by == FC.FARM:
                    kw["c"] = self.farm.wind_farm_list
                elif color_by == FC.CLUSTER:
                    kw["c"] = self.farm.cluster_list
                elif color_by[:5] == "mean_":
                    weights = self.fres[FV.WEIGHT]
                    if weights.dims == (FC.STATE,):
                        wx = "s"
                    elif weights.dims == (FC.STATE, FC.TURBINE):
                        wx = "st"
                    else:
                        raise ValueError(
                            f"Unsupported dimensions for '{FV.WEIGHT}': Expecting '{(FC.STATE,)}' or '{(FC.STATE, FC.TURBINE)}', got '{weights.dims}'"
                        )
                    kw["c"] = np.einsum(f"st,{wx}->t", self.fres[color_by[5:]], weights)
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

            c = kw.pop("c", "orange")
            if c is None or isinstance(c, str) or np.all(np.isreal(c)):
                im = ax.scatter(x, y, c=c, **kw)
                legend = False
            else:
                legend = True
                lbls = np.array(c)
                assert lbls.shape == (len(x),), (
                    f"Expecting color_by variable with shape {(len(x),)}, got {lbls.shape}"
                )
                u = np.unique(lbls)
                for lbl in u:
                    sel = lbls == lbl
                    im = ax.scatter(x[sel], y[sel], c=c[sel], label=lbl, **kw)
                    ax.legend(
                        title=color_by, loc="center left", bbox_to_anchor=(1, 0.5)
                    )

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
            elif annotate == 3:
                for wf_name, turb_indices in self.farm.get_wind_farm_mapping().items():
                    xc = np.mean(x[turb_indices])
                    yc = np.mean(y[turb_indices])
                    ax.text(xc, yc, wf_name, dict(size=fontsize))

        if self.farm.boundary is not None:
            hbargs = {"fill_mode": "inside_lightgray"}
            if bargs is not None:
                hbargs.update(bargs)
            self.farm.boundary.add_to_figure(ax, **hbargs)

        if title is not None or annotate != 3:
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

        if lonlat:
            ax.set_xlabel("Longitude [deg]")
            ax.set_ylabel("Latitude [deg]")
        else:
            ax.set_xlabel("x [m]" if not normalize_D else "x [D]")
            ax.set_ylabel("y [m]" if not normalize_D else "y [D]")
        ax.grid()

        # if len(self.farm.boundary_geometry) \
        #    or ( min(x) != max(x) and min(y) != max(y) ):
        if x is None or (min(x) != max(x) and min(y) != max(y)):
            ax.set_aspect("equal", adjustable="box")

        ax.autoscale_view(tight=True)

        if color_by is not None and not legend:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax)

        if ret_im:
            return ax, im

        return ax

    def write_plot(
        self, file_name: str | None = None, fontsize: int = 8, **kwargs: Any
    ) -> None:
        """
        Writes the layout plot to file.

        Parameters
        ----------
        file_name
            Name of the file into which to plot, or None
            for default
        fontsize
            Size of the turbine numbers
        kwargs
            Additional arguments for get_figure()

        """

        ax = self.get_figure(fontsize=fontsize, ret_im=False, **kwargs)
        fig = ax.get_figure()

        fname = file_name if file_name is not None else self.farm.name + ".png"
        fpath = self.get_fpath(fname)
        fig.savefig(fpath, bbox_inches="tight")

        plt.close(fig)

    def write_xyh(self, file_path: str | None = None) -> None:
        """
        Writes xyh layout file.

        Parameters
        ----------
        file_path
            The file into which to plot, or None
            for default

        """
        fname = file_path if file_path is not None else self.farm.name + ".xyh"
        data = self.get_layout_data(lonlat=False)
        if not self.farm.has_lonlat():
            np.savetxt(fname, data, header="x y h")
        else:
            data = np.concatenate((self.get_layout_data(lonlat=True), data), axis=1)
            np.savetxt(fname, data, header="lon lat x y h")

    def get_dataframe(
        self,
        type_col: str | None = None,
        algo: Algorithm | None = None,
        col_farm: str = "wind_farm",
        col_cluster: str = "cluster",
    ) -> pd.DataFrame:
        """
        Returns a pandas DataFrame with the layout data.

        Parameters
        ----------
        type_col
            Name of the turbine type column
        algo
            The algorithm, needed for turbine types
        col_farm
            The wind farm name column
        col_cluster
            The cluster name column

        Returns
        -------
        lyt
            The layout data

        """
        lonlat = self.farm.has_lonlat()
        if lonlat:
            cols = ["name", "lon", "lat", "x", "y", "h", "D"]
        else:
            cols = ["name", "x", "y", "h", "D"]

        if self.farm.wind_farm_names is not None:
            cols.append(col_farm)
            wfarms = [t.wind_farm_name for t in self.farm.turbines]
        else:
            wfarms = None
        if self.farm.cluster_names is not None:
            cols.append(col_cluster)
            clusters = [t.cluster_name for t in self.farm.turbines]
        else:
            clusters = None

        lyt = pd.DataFrame(index=range(self.farm.n_turbines), columns=cols)
        lyt.index.name = "index"
        lyt["name"] = [t.name for t in self.farm.turbines]
        if lonlat:
            data = self.get_layout_data(lonlat=True)
            lyt["lon"] = np.round(data[:, 0], 6)
            lyt["lat"] = np.round(data[:, 1], 6)
        data = self.get_layout_data(lonlat=False)
        lyt["x"] = np.round(data[:, 0], 4)
        lyt["y"] = np.round(data[:, 1], 4)
        lyt["h"] = np.round(data[:, 2], 4)
        lyt["D"] = [t.D for t in self.farm.turbines]

        if type_col is not None:
            lyt[type_col] = [m.name for m in algo.farm_controller.turbine_types]

        if wfarms is not None:
            lyt[col_farm] = wfarms
        if clusters is not None:
            lyt[col_cluster] = clusters

        return lyt

    def write_csv(
        self, file_name: str | None = None, verbosity: int = 1, **kwargs: Any
    ) -> None:
        """
        Writes the layout data to csv file.

        Parameters
        ----------
        file_name
            Name of the file into which to plot, or None
            for default
        verbosity
            The verbosity level, 0 = silent
        kwargs
            Additional arguments for get_dataframe()

        """
        fname = file_name if file_name is not None else self.farm.name + ".csv"
        fpath = self.get_fpath(fname)
        if verbosity > 0:
            print(f"Writing farm layout to '{fpath}'")
        self.get_dataframe(**kwargs).to_csv(fpath)

    def write_json(self, file_name: str | None = None) -> None:
        """
        Writes xyh layout file.

        Parameters
        ----------
        file_name
            Name of the file into which to plot, or None
            for default

        """

        data = self.get_layout_dict()

        fname = file_name if file_name is not None else self.farm.name + ".json"
        fpath = self.get_fpath(fname)
        with open(fpath, "w") as outfile:
            json.dump(data, outfile, indent=4)
