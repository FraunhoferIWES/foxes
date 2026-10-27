# mypy: disable-error-code=arg-type
# mypy: disable-error-code=assignment
# mypy: disable-error-code=misc
# mypy: disable-error-code=union-attr

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.collections import PatchCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from xarray import Dataset

import foxes.constants as FC
import foxes.variables as FV
from foxes.config import config
from foxes.output.output import Output

if TYPE_CHECKING:
    from foxes.core import Algorithm, WindFarm


_MIN_ROTOR_DIAMETER_PIXELS = 4.0


class FarmLayoutOutput(Output):
    """
    Plot the farm layout
    """

    def __init__(
        self,
        farm: WindFarm | None = None,
        farm_results: Dataset | None = None,
        from_results: bool = False,
        results_state: int | None = None,
        D: float | None = None,
        algo: Algorithm | None = None,
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
        algo
            The algorithm, needed by some functions
        kwargs
            Additional parameters for the base class
        """
        super().__init__(**kwargs)
        self.farm = farm
        self.fres = farm_results
        self.from_res = from_results
        self.rstate = results_state
        self.algo = algo
        self.D = D

        if self.farm is not None and self.algo is not None:
            assert self.farm is self.algo.farm, "Mismatch between farm and algo.farm"
        elif self.farm is None and self.algo is not None:
            self.farm = self.algo.farm

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

    def _get_auto_figsize(
        self,
        data: np.ndarray,
        lonlat: bool,
        min_rotor_diameter: float | None = None,
    ) -> tuple[float, float] | None:
        """Derive a bounded figure aspect ratio from plotted extents."""
        points = data[:, :2]
        points = points[np.all(np.isfinite(points), axis=1)]
        bounds = [points] if len(points) else []
        if not lonlat and self.farm.boundary is not None:
            boundary = np.stack(
                [self.farm.boundary.p_min(), self.farm.boundary.p_max()]
            )
            if np.all(np.isfinite(boundary)):
                bounds.append(boundary)
        if not bounds:
            return None

        extent = np.ptp(np.concatenate(bounds), axis=0)
        max_extent = np.max(extent)
        if max_extent <= 0.0:
            aspect = 1.0
        else:
            aspect_extent = np.maximum(extent, max_extent / 2.0)
            aspect = aspect_extent[0] / aspect_extent[1]
        default = np.asarray(plt.rcParams["figure.figsize"], dtype=float)
        base_size = np.sqrt(np.prod(default))
        figsize = base_size * np.array([np.sqrt(aspect), 1.0 / np.sqrt(aspect)])

        if min_rotor_diameter is not None:
            subplot_fraction = np.array(
                [
                    plt.rcParams["figure.subplot.right"]
                    - plt.rcParams["figure.subplot.left"],
                    plt.rcParams["figure.subplot.top"]
                    - plt.rcParams["figure.subplot.bottom"],
                ]
            )
            margins = np.array(
                [plt.rcParams["axes.xmargin"], plt.rcParams["axes.ymargin"]]
            )
            plot_extent = (extent + min_rotor_diameter) * (1.0 + 2.0 * margins)
            required = (
                plot_extent
                / min_rotor_diameter
                * _MIN_ROTOR_DIAMETER_PIXELS
                / (float(plt.rcParams["figure.dpi"]) * subplot_fraction)
            )
            figsize *= max(1.0, np.max(required / figsize))

        return float(figsize[0]), float(figsize[1])

    def _get_turbine_diameters(self) -> np.ndarray:
        """Return one finite positive rotor diameter per turbine."""
        if self.from_res and self.fres is not None and FV.D in self.fres:
            values = self.fres[FV.D]
            if FC.STATE in values.dims:
                values = values.isel({FC.STATE: self.rstate})
            diameters = np.asarray(values, dtype=config.dtype_double).reshape(-1)
        elif self.algo is not None:
            diameters = self.farm.get_rotor_diameters(self.algo)
        else:
            diameters = np.asarray(
                [
                    np.nan if turbine.D is None else turbine.D
                    for turbine in self.farm.turbines
                ],
                dtype=config.dtype_double,
            )
            if self.D is not None:
                diameters[~np.isfinite(diameters)] = self.D

        if diameters.size == 1 and self.farm.n_turbines != 1:
            diameters = np.full(self.farm.n_turbines, diameters.item())
        if diameters.shape != (self.farm.n_turbines,):
            raise ValueError(
                f"Expected {self.farm.n_turbines} rotor diameters, "
                f"got shape {diameters.shape}"
            )
        if np.any(~np.isfinite(diameters)) or np.any(diameters <= 0.0):
            raise ValueError(
                "True turbine radii require finite positive rotor diameters"
            )
        return diameters

    @staticmethod
    def _add_turbine_circles(
        ax: Axes,
        x: np.ndarray,
        y: np.ndarray,
        radii: np.ndarray,
        colors: Any,
        **kwargs: Any,
    ) -> PatchCollection:
        """Add physical turbine circles to an axis."""
        vmin = kwargs.pop("vmin", None)
        vmax = kwargs.pop("vmax", None)
        kwargs.setdefault("edgecolors", "face")
        circles = [Circle((xi, yi), radius) for xi, yi, radius in zip(x, y, radii)]
        if colors is not None and not (
            not isinstance(colors, str)
            and np.issubdtype(np.asarray(colors).dtype, np.number)
        ):
            kwargs["facecolors"] = colors
        collection = PatchCollection(circles, **kwargs)
        if (
            colors is not None
            and not isinstance(colors, str)
            and np.issubdtype(np.asarray(colors).dtype, np.number)
        ):
            collection.set_array(np.asarray(colors))
            collection.set_clim(vmin, vmax)
        ax.add_collection(collection)
        return collection

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
        legend_labels: dict[str, str] | None = None,
        anno_delx: float = 0,
        anno_dely: float = 0,
        lonlat: bool = False,
        true_turbine_radii: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Creates farm layout figure.

        Parameters
        ----------
        color_by
            Set turbine color by variable results.
            Use "mean_REWS", etc, for means, also
            min, max, sum. All wrt states
        fontsize
            Size of the turbine numbers
        figsize
            The figsize for plt.Figure, or None to derive it from plot extents
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
            Arguments for boundary plotting. The optional ``boundary`` entry
            overrides the farm geometry for this figure only; ``None`` disables
            the boundary overlay without changing the farm or its bounds.
        legend_labels
            Mapping from marker colors to labels for an upper-left legend
        anno_delx
            The annotation delta x
        anno_dely
            The annotation delta y
        lonlat
            Flag for lonlat coordinates, if available
        true_turbine_radii
            Replace scatter markers by circles using the physical turbine
            radii in data coordinates while preserving direct and ``color_by``
            fill colors. This is not available for lon/lat plots.
        kwargs
            Parameters forwarded to `matplotlib.pyplot.scatter`, or to a
            `matplotlib.collections.PatchCollection` for true turbine radii.

        Returns
        -------
        ax
            The axis object
        im
            The image object

        """
        if self.nofig:
            return None, None
        if true_turbine_radii and lonlat:
            raise ValueError("True turbine radii are not available for lon/lat plots")

        data = self.get_layout_data(lonlat=lonlat)
        D = self.D
        diameters = None
        if self.farm.n_turbines:
            if true_turbine_radii or (normalize_D and D is None):
                diameters = self._get_turbine_diameters()
            if normalize_D and D is None:
                assert diameters is not None
                if np.min(diameters) != np.max(diameters):
                    raise ValueError(f"Expecting uniform D, found {diameters}")
                D = diameters[0]

        if fig is None:
            if figsize is None:
                min_rotor_diameter = (
                    float(np.min(diameters))
                    if true_turbine_radii and diameters is not None
                    else None
                )
                figsize = self._get_auto_figsize(
                    data, lonlat, min_rotor_diameter=min_rotor_diameter
                )
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
        else:
            ax = fig.axes[0] if ax is None else ax

        x = None
        if self.farm.n_turbines:
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
            radii = None
            if true_turbine_radii:
                assert diameters is not None
                if normalize_D:
                    assert D is not None
                    radii = diameters / (2.0 * D)
                else:
                    radii = diameters / 2.0
            if (
                color_by is None
                or c is None
                or isinstance(c, str)
                or np.issubdtype(np.asarray(c).dtype, np.number)
            ):
                if true_turbine_radii:
                    assert radii is not None
                    im = self._add_turbine_circles(ax, x, y, radii, c, **kw)
                else:
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
                    if true_turbine_radii:
                        assert radii is not None
                        im = self._add_turbine_circles(
                            ax, x[sel], y[sel], radii[sel], c[sel], label=lbl, **kw
                        )
                    else:
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

        hbargs = {"fill_mode": "inside_lightgray"}
        if bargs is not None:
            hbargs.update(bargs)
        boundary = hbargs.pop("boundary", self.farm.boundary)
        if boundary is not None:
            boundary.add_to_figure(ax, **hbargs)

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

        if legend_labels:
            handles = [
                Line2D(
                    [],
                    [],
                    color=color,
                    marker="o",
                    linestyle="none",
                    label=label,
                )
                for color, label in legend_labels.items()
            ]
            ax.legend(handles=handles, loc="upper left")

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

        wfnames = self.farm.wind_farm_names
        if wfnames is not None and len(wfnames) > 1:
            cols.append(col_farm)
        clnames = self.farm.cluster_names
        if clnames is not None and len(clnames) > 1:
            cols.append(col_cluster)

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
        lyt["h"] = np.round(data[:, 2], 2)
        if np.any(np.isnan(lyt["h"])):
            if self.algo is not None:
                lyt["h"] = np.round(self.farm.get_hub_heights(algo=self.algo), 2)
            else:
                lyt["h"] = [t.H for t in self.farm.turbines]
        if self.algo is not None:
            lyt["D"] = np.round(self.farm.get_rotor_diameters(algo=self.algo), 2)
        else:
            lyt["D"] = [t.D for t in self.farm.turbines]

        if type_col is not None:
            lyt[type_col] = [m.name for m in algo.farm_controller.turbine_types]

        if col_farm in cols:
            lyt[col_farm] = self.farm.wind_farm_list
        if col_cluster in cols:
            lyt[col_cluster] = self.farm.cluster_list

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
