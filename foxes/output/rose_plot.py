import numpy as np
import matplotlib.pyplot as plt
from xarray import Dataset
from matplotlib.cm import ScalarMappable
from matplotlib.projections.polar import PolarAxes
from matplotlib.lines import Line2D

from foxes.algorithms import Downwind
from foxes.core import WindFarm, Turbine
from foxes.models import ModelBook
import foxes.variables as FV
import foxes.constants as FC

from .output import Output


class RosePlotOutput(Output):
    """
     Class for rose plot creation

     Attributes
     ----------
     results: pandas.DataFrame
         The calculation results (farm or points)

    :group: output

    """

    def __init__(
        self,
        farm_results=None,
        point_results=None,
        use_points=False,
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        farm_results: xarray.Dataset, optional
            The farm results
        point_results: xarray.Dataset, optional
            The point results
        use_points: bool
            Flag for using points in cases where both
            farm and point results are given
        kwargs: dict, optional
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
            raise KeyError(f"Require either farm_results or point_results")

    @classmethod
    def get_data_info(cls, dname):
        """
        Returns default description for a variable.

        Parameters
        ----------
        dname: str
            The variable name

        Returns
        -------
        title: str
            The long name of the variable
        legend: str
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
        wd_sectors,
        ws_var,
        ws_bins,
        wd_var=FV.AMB_WD,
        turbine=0,
        point=0,
        add_inf=False,
    ):
        """
        Generates the plot data

        Parameters
        ----------
        wd_sectors: int
            The number of wind rose sectors
        ws_var: str
            The wind speed variable
        ws_bins: list of float
            The wind speed bins
        wd_var: str
            The wind direction variable
        turbine: int
            The turbine index, for weights and for
            data if farm_results are given
        point: int
            The point index, for data if point_results
            are given
        add_inf: bool
            Add an upper bin up to infinity

        Returns
        -------
        data: xarray.Dataset
            The plot data

        """
        if add_inf:
            ws_bins = list(ws_bins) + [np.inf]
        w = self.results[FV.WEIGHT].to_numpy()[:, turbine]
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
        wd_sectors,
        ws_var,
        ws_bins,
        wd_var=FV.AMB_WD,
        fig=None,
        ax=None,
        figsize=None,
        freq_delta=3,
        cmap="summer",
        title=None,
        legend_pars=None,
        ret_data=False,
        **kwargs,
    ):
        """
        Creates the figure

        Parameters
        ----------
        wd_sectors: int
            The number of wind rose sectors
        ws_var: str
            The wind speed variable
        ws_bins: list of float
            The wind speed bins
        wd_var: str
            The wind direction variable
        fig: pyplot.Figure, optional
            The figure object
        ax: pyplot.Axes, optional
            The axes object
        figsize: tuple, optional
            The figsize argument for plt.subplots
        freq_delta: int
            The frequency delta for the label
            in percent
        cmap: str
            The color map
        title: str, optional
            The title
        legend_pars: dict, optional
            Parameters for the legend
        ret_data: bool
            Flag for returning wind rose data
        kwargs: dict, optional
            Additional parameters for get_data

        Returns
        -------
        ax: pyplot.Axes
            The axes object
        data: xarray.Dataset, optional
            The plot data

        """
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
            f"[{ws_bins[i]:.1f}, {ws_bins[i+1]:.1f})" for i in range(n_wsb - 1, -1, -1)
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

    def write_figure(self, file_name, *args, ret_data=False, **kwargs):
        """
        Write rose plot to file

        Parameters
        ----------
        file_name: str
            Name of the output file
        args: tuple, optional
            Additional parameters for get_figure
        ret_data: bool
            Flag for returning wind rose data
        kwargs: dict, optional
            Additional parameters for get_figure

        Returns
        -------
        data: pd.DataFrame, optional
            The wind rose data

        """

        r = self.get_figure(*args, ret_data=ret_data, **kwargs)
        fpath = self.get_fpath(file_name)
        if ret_data:
            r[0].get_figure().savefig(fpath, bbox_inches="tight")
            return r[1]
        else:
            r.get_figure().savefig(fpath, bbox_inches="tight")


class StatesRosePlotOutput(RosePlotOutput):
    """
    Class for rose plot creation directly from states
    :group: output
    """

    def __init__(
        self,
        states,
        point,
        mbook=None,
        ws_var=FV.AMB_REWS,
        **kwargs,
    ):
        """
        Constructor.

        Parameters
        ----------
        states: foxes.core.States
            The states from which to compute the wind rose
        point: numpy.ndarray
            The evaluation point, shape: (3,)
        mbook: foxes.models.ModelBook, optional
            The model book
        ws_var: str
            The wind speed variable name
        kwargs: dict, optional
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

    Attributes
    ----------
    farm_results: xarray.Dataset
        The wind farm results

    :group: output

    """

    def __init__(self, farm_results, **kwargs):
        """
        Constructor

        Parameters
        ----------
        farm_results: xarray.Dataset
            The wind farm results
        kwargs: dict, optional
            Parameters for the base class

        """
        super().__init__(**kwargs)
        self.farm_results = farm_results

    def get_data(
        self,
        variable,
        ws_bins,
        wd_sectors=12,
        wd_var=FV.AMB_WD,
        ws_var=FV.AMB_REWS,
        turbine=0,
        contraction="weights",
    ):
        """
        Generates the plot data

        Parameters
        ----------
        variable: str
            The variable name
        ws_bins: list of float
            The wind speed bins
        wd_var: str
            The wind direction variable
        ws_var: str
            The wind speed variable
        turbine: int
            The turbine index
        contraction: str
            The contraction method for states:
            weights, mean_no_weights, sum_no_weights

        Returns
        -------
        data: xarray.Dataset
            The plot data

        """
        var = self.farm_results[variable].to_numpy()[:, turbine]
        w = self.farm_results[FV.WEIGHT].to_numpy()[:, turbine]
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
        variable,
        ws_bins,
        wd_sectors=12,
        wd_var=FV.AMB_WD,
        ws_var=FV.AMB_REWS,
        turbine=0,
        contraction="weights",
        fig=None,
        ax=None,
        title=None,
        figsize=None,
        ret_data=False,
        **kwargs,
    ):
        """
        Creates the figure

        Parameters
        ----------
        variable: str
            The variable name
        ws_bins: list of float
            The wind speed bins
        wd_var: str
            The wind direction variable
        ws_var: str
            The wind speed variable
        turbine: int
            The turbine index
        contraction: str
            The contraction method for states:
            weights, mean_no_weights, sum_no_weights
        fig: pyplot.Figure, optional
            The figure object
        ax: pyplot.Axes, optional
            The axes object
        title: str, optional
            The title
        figsize: tuple, optional
            The figsize argument for plt.subplots
        ret_data: bool
            Flag for returning wind rose data
        kwargs: dict, optional
            Additional parameters for plt.pcolormesh

        Returns
        -------
        ax: pyplot.Axes
            The axes object

        """
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

    def write_figure(self, file_name, *args, ret_data=False, **kwargs):
        """
        Write rose plot to file

        Parameters
        ----------
        file_name: str
            Name of the output file
        args: tuple, optional
            Additional parameters for get_figure
        ret_data: bool
            Flag for returning wind rose data
        kwargs: dict, optional
            Additional parameters for get_figure

        Returns
        -------
        data: pd.DataFrame, optional
            The wind rose data

        """

        r = self.get_figure(*args, ret_data=ret_data, **kwargs)
        fpath = self.get_fpath(file_name)
        if ret_data:
            r[0].get_figure().savefig(fpath, bbox_inches="tight")
            return r[1]
        else:
            r.get_figure().savefig(fpath, bbox_inches="tight")
