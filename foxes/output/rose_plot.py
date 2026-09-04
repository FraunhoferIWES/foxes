# mypy: disable-error-code=arg-type
# mypy: disable-error-code=operator
# mypy: disable-error-code=union-attr

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from xarray import Dataset
from matplotlib.projections.polar import PolarAxes
from matplotlib.lines import Line2D
from typing import Any

from foxes.algorithms import Downwind
from foxes.core import States, Turbine, WindFarm
from foxes.models import ModelBook
import foxes.variables as FV
import foxes.constants as FC

from .output import Output


class RosePlotOutput(Output):
    """
    Class for rose plot creation
    """

    def __init__(
        self,
        farm_results: Dataset | None = None,
        point_results: Dataset | None = None,
        use_points: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        farm_results
            The farm results
        point_results
            The point results
        use_points
            Flag for using points in cases where both
            farm and point results are given
        kwargs
            Additional parameters for the base class
        """
        super().__init__(**kwargs)
        if use_points or (farm_results is None and point_results is not None):
            self.results = point_results
            self._rtype = FC.POINT
        elif farm_results is not None:
            self.results = farm_results
            self._rtype = FC.TURBINE
        else:
            raise KeyError("Require either farm_results or point_results")

    @classmethod
    def get_data_info(cls, dname: str) -> tuple[str, str]:
        """
        Returns default description for a variable.

        Parameters
        ----------
        dname
            The variable name

        Returns
        -------
        title
            The long name of the variable
        legend
            The legend/axis text

        """

        if dname == FV.D:
            return "Rotor diameter", f"{FV.D} [m]"
        if dname == FV.H:
            return "Hub height", f"{FV.H} [m]"

        if dname == FV.WS:
            return "Wind speed", f"{FV.WS} [m/s]"
        if dname == FV.REWS:
            return "Rotor equivalent wind speed", f"{FV.REWS} [m/s]"
        if dname == FV.REWS2:
            return "Rotor equivalent wind speed (2nd moment)", f"{FV.REWS2} [m/s]"
        if dname == FV.REWS3:
            return "Rotor equivalent wind speed (3rd moment)", f"{FV.REWS3} [m/s]"
        if dname == FV.WD:
            return "Wind direction", f"{FV.WD} [deg]"
        if dname == FV.TI:
            return "Turbulence intensity", f"{FV.TI} [1]"
        if dname == FV.RHO:
            return "Air density", f"{FV.RHO} [kg/m3]"
        if dname == FV.CT:
            return "Thrust coefficient", f"{FV.CT} [1]"
        if dname == FV.P:
            return "Power", f"{FV.P} [kW]"
        if dname == FV.YAW:
            return "Yaw angle", f"{FV.YAW} [deg]"
        if dname == FV.YAWM:
            return "Yaw misalignment", f"{FV.YAWM} [deg]"

        if dname in FV.amb2var:
            title, legend = cls.get_data_info(FV.amb2var[dname])
            return f"Ambient {title.lower()}", f"AMB_{legend}"

        return dname, dname

    def get_data(
        self,
        wd_sectors: int,
        ws_var: str,
        ws_bins: Any,
        wd_var: str = FV.AMB_WD,
        turbine: int = 0,
        point: int = 0,
        add_inf: bool = False,
    ) -> Dataset:
        """
        Generates the plot data

        Parameters
        ----------
        wd_sectors
            The number of wind rose sectors
        ws_var
            The wind speed variable
        ws_bins
            The wind speed bins
        wd_var
            The wind direction variable
        turbine
            The turbine index, for weights and for
            data if farm_results are given
        point
            The point index, for data if point_results
            are given
        add_inf
            Add an upper bin up to infinity

        Returns
        -------
        data
            The plot data

        """
        assert self.results is not None
        if self.results[FV.WEIGHT].dims == (FC.STATE,):
            w = self.results[FV.WEIGHT].to_numpy()
        elif self.results[FV.WEIGHT].dims == (FC.STATE, FC.TURBINE):
            w = self.results[FV.WEIGHT].to_numpy()[:, turbine]
        elif self.results[FV.WEIGHT].dims == (FC.STATE, FC.POINT):
            w = self.results[FV.WEIGHT].to_numpy()[:, point]
        else:
            raise ValueError(
                f"Wrong dimensions for '{FV.WEIGHT}'. Expecting {(FC.STATE,)}, {(FC.STATE, FC.TURBINE)} or {(FC.STATE, FC.POINT)}, got {self.results[FV.WEIGHT].dims}"
            )

        if add_inf:
            ws_bins = list(ws_bins) + [np.inf]
        t = turbine if self._rtype == FC.TURBINE else point
        ws = self.results[ws_var].to_numpy()[:, t]
        wd = self.results[wd_var].to_numpy()[:, t].copy()
        wd_delta = 360 / wd_sectors
        wd[wd >= 360 - wd_delta / 2] -= 360
        wd_bins = np.arange(-wd_delta / 2, 360, wd_delta)
        ws_bins = np.asarray(ws_bins, dtype=ws.dtype)

        freq = 100 * np.histogram2d(wd, ws, (wd_bins, ws_bins), weights=w)[0]

        data = Dataset(
            coords={
                wd_var: np.arange(0, 360, wd_delta),
                ws_var: 0.5 * (ws_bins[:-1] + ws_bins[1:]),
            },
            data_vars={
                f"bin_min_{wd_var}": (wd_var, wd_bins[:-1]),
                f"bin_max_{wd_var}": (wd_var, wd_bins[1:]),
                f"bin_min_{ws_var}": (ws_var, ws_bins[:-1]),
                f"bin_max_{ws_var}": (ws_var, ws_bins[1:]),
                "frequency": ((wd_var, ws_var), freq),
            },
            attrs={
                f"{wd_var}_bounds": wd_bins,
                f"{ws_var}_bounds": ws_bins,
            },
        )

        return data

    def get_figure(
        self,
        wd_sectors: int,
        ws_var: str,
        ws_bins: Any,
        wd_var: str = FV.AMB_WD,
        fig: Figure | None = None,
        ax: Axes | None = None,
        figsize: Any = None,
        freq_delta: float = 3,
        cmap: str = "summer",
        title: str | None = None,
        legend_pars: dict[str, Any] | None = None,
        ret_data: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Creates the figure

        Parameters
        ----------
        wd_sectors
            The number of wind rose sectors
        ws_var
            The wind speed variable
        ws_bins
            The wind speed bins
        wd_var
            The wind direction variable
        fig
            The figure object
        ax
            The axes object
        figsize
            The figsize argument for plt.subplots
        freq_delta
            The frequency delta for the label
            in percent
        cmap
            The color map
        title
            The title
        legend_pars
            Parameters for the legend
        ret_data
            Flag for returning wind rose data
        kwargs
            Additional parameters for get_data

        Returns
        -------
        ax
            The axes object
        data
            The plot data

        """
        if self.nofig:
            return None

        data = self.get_data(wd_sectors, ws_var, ws_bins, wd_var, **kwargs)

        n_wsb = data.sizes[ws_var]
        n_wdb = data.sizes[wd_var]
        ws_bins = np.asarray(data.attrs[f"{ws_var}_bounds"])
        wd_cent = np.mod(90 - data[wd_var].to_numpy(), 360)
        wd_cent = np.radians(wd_cent)
        wd_delta = 360 / n_wdb
        wd_width = np.radians(0.9 * wd_delta)
        freq = data["frequency"].to_numpy()

        if ax is not None:
            if not isinstance(ax, PolarAxes):
                raise TypeError(
                    f"Require axes of type '{PolarAxes.__name__}' for '{type(self).__name__}', got '{type(ax).__name__}'"
                )
        else:
            fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})

        bcmap = plt.get_cmap(cmap, n_wsb)
        color_list = bcmap(np.linspace(0, 1, n_wsb))

        bottom = np.zeros(n_wdb)
        for wsi in range(n_wsb):
            ax.bar(
                wd_cent,
                freq[:, wsi],
                bottom=bottom,
                width=wd_width,
                color=color_list[wsi],
            )
            bottom += freq[:, wsi]

        fmax = np.max(np.sum(freq, axis=1))
        freq_delta = int(freq_delta)
        freq_ticks = np.arange(0, fmax + freq_delta / 2, freq_delta, dtype=np.int32)[1:]

        tksl = np.arange(0, 360, max(wd_delta, 30))
        tks = np.radians(np.mod(90 - tksl, 360))
        ax.set_xticks(tks, [f"{int(d)}°" for d in tksl])
        ax.set_yticks(freq_ticks, [f"{f}%" for f in freq_ticks])
        ax.set_title(title)

        llines = [Line2D([0], [0], color=c, lw=10) for c in np.flip(color_list, axis=0)]
        lleg = [
            f"[{ws_bins[i]:.1f}, {ws_bins[i + 1]:.1f})"
            for i in range(n_wsb - 1, -1, -1)
        ]
        lpars = dict(
            loc="upper left",
            bbox_to_anchor=(0.8, 0.5),
            title=f"{ws_var}",
        )
        wsl = [FV.WS, FV.REWS, FV.REWS2, FV.REWS3]
        wsl += [FV.var2amb[v] for v in wsl]
        if ws_var in wsl:
            lpars["title"] += " [m/s]"
        if legend_pars is not None:
            lpars.update(legend_pars)
        ax.legend(llines, lleg, **lpars)

        if ret_data:
            return ax, data
        else:
            return ax

    def write_figure(
        self, file_name: str, ret_data: bool = False, **kwargs: Any
    ) -> Any:
        """
        Write rose plot to file

        Parameters
        ----------
        file_name
            Name of the output file
        args
            Additional parameters for get_figure
        ret_data
            Flag for returning wind rose data
        kwargs
            Additional parameters for get_figure

        Returns
        -------
        data
            The wind rose data

        """
        if self.nofig:
            return None

        r = self.get_figure(ret_data=ret_data, **kwargs)
        fpath = self.get_fpath(file_name)
        if ret_data:
            r[0].get_figure().savefig(fpath, bbox_inches="tight")
            return r[1]
        else:
            r.get_figure().savefig(fpath, bbox_inches="tight")


class StatesRosePlotOutput(RosePlotOutput):
    """
    Class for rose plot creation directly from states
    """

    def __init__(
        self,
        states: States,
        point: np.ndarray,
        mbook: ModelBook | None = None,
        ws_var: str = FV.AMB_REWS,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        states
            The states from which to compute the wind rose
        point
            The evaluation point, shape: (3,)
        mbook
            The model book
        ws_var
            The wind speed variable name
        kwargs
            Additional parameters for the base class
        """
        farm = WindFarm()
        farm.add_turbine(
            Turbine(
                xy=point[:2],
                H=point[2],
                turbine_models=["null_type"],
            ),
            verbosity=0,
        )

        mbook = mbook if mbook is not None else ModelBook()
        algo = Downwind(farm, states, wake_models=[], mbook=mbook, verbosity=0)

        results = algo.calc_farm(ambient=True).rename_vars({ws_var: FV.AMB_WS})

        super().__init__(results, **kwargs)


class WindRoseBinPlot(Output):
    """
    Plots mean data in wind rose bins
    """

    def __init__(self, farm_results: Dataset, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        farm_results
            The wind farm results
        kwargs
            Parameters for the base class
        """
        super().__init__(**kwargs)
        self.farm_results = farm_results

    def get_data(
        self,
        variable: str,
        ws_bins: Any,
        wd_sectors: int = 12,
        wd_var: str = FV.AMB_WD,
        ws_var: str = FV.AMB_REWS,
        turbine: int = 0,
        contraction: str = "weights",
    ) -> Dataset:
        """
        Generates the plot data

        Parameters
        ----------
        variable
            The variable name
        ws_bins
            The wind speed bins
        wd_var
            The wind direction variable
        ws_var
            The wind speed variable
        turbine
            The turbine index
        contraction
            The contraction method for states:
            weights, mean_no_weights, sum_no_weights

        Returns
        -------
        data
            The plot data

        """
        if self.farm_results[FV.WEIGHT].dims == (FC.STATE,):
            w = self.farm_results[FV.WEIGHT].to_numpy()
        elif self.farm_results[FV.WEIGHT].dims == (FC.STATE, FC.TURBINE):
            w = self.farm_results[FV.WEIGHT].to_numpy()[:, turbine]
        else:
            raise ValueError(
                f"Wrong dimensions for '{FV.WEIGHT}'. Expecting {(FC.STATE,)} or {(FC.STATE, FC.TURBINE)}, got {self.farm_results[FV.WEIGHT].dims}"
            )

        var = self.farm_results[variable].to_numpy()[:, turbine]
        ws = self.farm_results[ws_var].to_numpy()[:, turbine]
        wd = self.farm_results[wd_var].to_numpy()[:, turbine].copy()
        wd_delta = 360 / wd_sectors
        wd[wd >= 360 - wd_delta / 2] -= 360
        wd_bins = np.arange(-wd_delta / 2, 360, wd_delta)
        ws_bins = np.asarray(ws_bins)

        if contraction == "weights":
            z = np.histogram2d(wd, ws, (wd_bins, ws_bins), weights=w)[0]
            z[z < 1e-13] = np.nan
            z = np.histogram2d(wd, ws, (wd_bins, ws_bins), weights=w * var)[0] / z
        elif contraction == "mean_no_weights":
            z = np.histogram2d(wd, ws, (wd_bins, ws_bins))[0].astype(w.dtype)
            z[z < 1] = np.nan
            z = np.histogram2d(wd, ws, (wd_bins, ws_bins), weights=var)[0] / z
        elif contraction == "sum_no_weights":
            z = np.histogram2d(wd, ws, (wd_bins, ws_bins), weights=var)[0]
        else:
            raise KeyError(
                f"Contraction '{contraction}' not supported. Choices: weights, mean_no_weights, sum_no_weights"
            )

        data = Dataset(
            coords={
                wd_var: 0.5 * (wd_bins[:-1] + wd_bins[1:]),
                ws_var: 0.5 * (ws_bins[:-1] + ws_bins[1:]),
            },
            data_vars={
                variable: ((wd_var, ws_var), z),
            },
            attrs={
                f"{wd_var}_bounds": wd_bins,
                f"{ws_var}_bounds": ws_bins,
            },
        )

        return data

    def get_figure(
        self,
        variable: str,
        ws_bins: Any,
        wd_sectors: int = 12,
        wd_var: str = FV.AMB_WD,
        ws_var: str = FV.AMB_REWS,
        turbine: int = 0,
        contraction: str = "weights",
        fig: Figure | None = None,
        ax: Axes | None = None,
        title: str | None = None,
        figsize: Any = None,
        ret_data: bool = False,
        **kwargs: Any,
    ) -> Any:
        """
        Creates the figure

        Parameters
        ----------
        variable
            The variable name
        ws_bins
            The wind speed bins
        wd_var
            The wind direction variable
        ws_var
            The wind speed variable
        turbine
            The turbine index
        contraction
            The contraction method for states:
            weights, mean_no_weights, sum_no_weights
        fig
            The figure object
        ax
            The axes object
        title
            The title
        figsize
            The figsize argument for plt.subplots
        ret_data
            Flag for returning wind rose data
        kwargs
            Additional parameters for plt.pcolormesh

        Returns
        -------
        ax
            The axes object

        """
        if self.nofig:
            return None

        data = self.get_data(
            variable=variable,
            ws_bins=ws_bins,
            wd_sectors=wd_sectors,
            wd_var=wd_var,
            ws_var=ws_var,
            turbine=turbine,
            contraction=contraction,
        )

        wd_delta = 360 / data.sizes[wd_var]
        wd_bins = np.mod(90 - data.attrs[f"{wd_var}_bounds"], 360)
        wd_bins = np.radians(wd_bins)
        ws_bins = data.attrs[f"{ws_var}_bounds"]

        if ax is not None:
            if not isinstance(ax, PolarAxes):
                raise TypeError(
                    f"Require axes of type '{PolarAxes.__name__}' for '{type(self).__name__}', got '{type(ax).__name__}'"
                )
        else:
            fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})

        y, x = np.meshgrid(ws_bins, wd_bins)
        z = data[variable].to_numpy()

        prgs = {"shading": "flat"}
        prgs.update(kwargs)

        img = ax.pcolormesh(x, y, z, **prgs)

        tksl = np.arange(0, 360, max(wd_delta, 30))
        tks = np.radians(np.mod(90 - tksl, 360))
        ax.set_xticks(tks, [f"{d}°" for d in tksl])
        ax.set_yticks(ws_bins)
        ax.set_title(title)
        cbar = fig.colorbar(img, ax=ax, pad=0.12)
        cbar.ax.set_title(variable)

        if ret_data:
            return ax, data
        else:
            return ax

    def write_figure(
        self, file_name: str, ret_data: bool = False, **kwargs: Any
    ) -> Any:
        """
        Write rose plot to file

        Parameters
        ----------
        file_name
            Name of the output file
        args
            Additional parameters for get_figure
        ret_data
            Flag for returning wind rose data
        kwargs
            Additional parameters for get_figure

        Returns
        -------
        data
            The wind rose data

        """
        if self.nofig:
            return None

        r = self.get_figure(ret_data=ret_data, **kwargs)
        fpath = self.get_fpath(file_name)
        if ret_data:
            r[0].get_figure().savefig(fpath, bbox_inches="tight")
            return r[1]
        else:
            r.get_figure().savefig(fpath, bbox_inches="tight")
