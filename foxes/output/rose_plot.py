import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.projections.polar import PolarAxes

from foxes.utils import wd2uv, uv2wd
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
        sectors,
        var,
        var_bins,
        wd_var=FV.AMB_WD,
        weight_var=FV.WEIGHT,
        turbine=None,
        point=None,
        start0=False,
    ):
        """
        Get pandas DataFrame with wind rose data.

        Parameters
        ----------
        sectors: int
            The number of wind direction sectors
        var: str
            The data variable name
        var_bins: list of float
            The variable bin separation values
        wd_var: str, optional
            The wind direction variable name
        weight_var: str, optional
            The weights variable name
        turbine: int, optional
            Only relevant in case of farm results.
            If None, mean over all turbines.
            Else, data from a single turbine
        point: int, optional
            Only relevant in case of point results.
            If None, mean over all points.
            Else, data from a single point
        start0: bool
            Flag for starting the first sector at
            zero degrees instead of minus half width

        Returns
        -------
        pd.DataFrame:
            The wind rose data

        """

        dwd = 360.0 / sectors
        wds = np.arange(0.0, 360.0, dwd)
        wdb = np.append(wds, 360) if start0 else np.arange(-dwd / 2, 360.0, dwd)
        lgd = f"interval_{var}"

        data = self.results[[wd_var, weight_var]].copy()
        data[lgd] = self.results[var]
        uv = wd2uv(data[wd_var].to_numpy())
        data["u"] = uv[:, 0]
        data["v"] = uv[:, 1]

        data[weight_var] *= 100
        data = data.rename(columns={weight_var: "frequency"})

        el = turbine if self._rtype == FC.TURBINE else point
        if el is None:
            data = data.groupby(level=0).mean()
        else:
            sname = data.index.names[0]
            grp = data.reset_index().groupby(self._rtype)
            data = grp.get_group(el).set_index(sname)

        data["wd"] = uv2wd(data[["u", "v"]].to_numpy())
        data.drop(["u", "v"], axis=1, inplace=True)
        if not start0:
            data.loc[data["wd"] > 360.0 - dwd / 2, "wd"] -= 360.0

        data[wd_var] = pd.cut(data["wd"], wdb, labels=wds)
        data[lgd] = pd.cut(data[lgd], var_bins, right=False, include_lowest=True)

        grp = data[[wd_var, lgd, "frequency"]].groupby([wd_var, lgd], observed=False)
        data = grp.sum().reset_index()

        data[wd_var] = data[wd_var].astype(np.float64)
        data[lgd] = list(data[lgd])
        if start0:
            data[wd_var] += dwd / 2

        ii = pd.IntervalIndex(data[lgd])
        data[var] = ii.mid
        data[f"bin_min_{var}"] = ii.left
        data[f"bin_max_{var}"] = ii.right
        data[f"bin_min_{wd_var}"] = data[wd_var] - dwd / 2
        data[f"bin_max_{wd_var}"] = data[wd_var] + dwd / 2
        data["sector"] = (data[wd_var] / dwd).astype(int)

        data = data[
            [
                wd_var,
                var,
                "sector",
                f"bin_min_{wd_var}",
                f"bin_max_{wd_var}",
                f"bin_min_{var}",
                f"bin_max_{var}",
                lgd,
                "frequency",
            ]
        ]
        data.index.name = "bin"

        return data

    def get_figure(
            self,
            wd_sectors,
            ws_var,
            ws_bins,
            wd_var=FV.AMB_WD,
            turbine=0,
            point=0,
            fig=None,
            ax=None,
            figsize=None,
            freq_delta=3,
            cmap="Greens",
            title=None,
            cbar_title=None,
            add_inf=False,
        ):
        """
        Gather the plot data
        
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
        cbar_title: str, optional
            The title of the colorbar
        add_inf: bool
            Add an upper bin up to infinity

        Returns
        -------
        ax: pyplot.Axes
            The axes object
        
        """
        if add_inf:
            ws_bins = list(ws_bins) + [np.inf]
        w = self.results[FV.WEIGHT].to_numpy()[:, turbine]
        t = turbine if self._rtype == FC.TURBINE else point
        ws = self.results[ws_var].to_numpy()[:, t]
        wd = self.results[wd_var].to_numpy()[:, t].copy()
        wd_delta = 360/wd_sectors
        wd[wd>=360-wd_delta/2] -= 360
        wd_bins = np.arange(-wd_delta/2, 360, wd_delta)
        wd_cent = np.arange(0, 360, wd_delta)
        wd_cent = np.radians(np.mod(90 - wd_cent, 360))
        wd_width = np.radians(wd_delta*0.9)
        n_wdb = len(wd_bins) - 1
        n_wsb = len(ws_bins) - 1

        if ax is not None:
            if not isinstance(ax, PolarAxes):
                raise TypeError(f"Require axes of type '{PolarAxes.__name__}' for '{type(self).__name__}', got '{type(ax).__name__}'")
        else:
            fig, ax = plt.subplots(
                figsize=figsize, subplot_kw={"projection": "polar"})
        
        freq = 100*np.histogram2d(wd, ws, (wd_bins, ws_bins), weights=w)[0]

        bcmap = plt.get_cmap(cmap, n_wsb)
        color_list = bcmap(np.linspace(0, 1, n_wsb))
    
        bottom = np.zeros(n_wdb)
        for wsi in range(n_wsb):
            ax.bar(wd_cent, freq[:, wsi], bottom=bottom, width=wd_width, color=color_list[wsi])
            bottom += freq[:, wsi]

        fmax = np.max(np.sum(freq, axis=1))
        freq_delta = int(freq_delta)
        freq_ticks = np.arange(0, fmax+freq_delta/2, freq_delta, dtype=np.int32)[1:]

        tksl = np.arange(0,360,wd_delta)
        tks = np.radians(np.mod(90 - tksl, 360))
        ax.set_xticks(tks, [f"{int(d)}°" for d in tksl])
        ax.set_yticks(freq_ticks, [f"{f}%" for f in freq_ticks])
        ax.set_title(title)

        sm = ScalarMappable(cmap=bcmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, ticks=np.linspace(0,1,n_wsb+1), pad=0.1)
        clabels = [f"{ws}" for ws in ws_bins]
        clabels[-1] = f""
        cbar.ax.set_yticklabels(ws_bins)

        if cbar_title is None:
            cbar_title = f"{ws_var}"
            wsl = [FV.WS, FV.REWS, FV.REWS2, FV.REWS3]
            wsl += [FV.var2amb[v] for v in wsl]
            if ws_var in wsl:
                cbar_title += " [m/s]"
        cbar.ax.set_title(cbar_title)
        
        return ax

    def write_figure(
        self,
        file_name,
        sectors,
        var,
        var_bins,
        ret_data=False,
        **kwargs,
    ):
        """
        Write rose plot to file

        Parameters
        ----------
        file_name: str
            Name of the output file
        sectors: int
            The number of wind direction sectors
        var: str
            The data variable name
        var_bins: list of float
            The variable bin separation values
        ret_data: bool
            Flag for returning wind rose data
        kwargs: dict, optional
            Additional parameters for get_figure()

        Returns
        -------
        data: pd.DataFrame, optional
            The wind rose data

        """

        r = self.get_figure(
            sectors=sectors,
            var=var,
            var_bins=var_bins,
            ret_data=ret_data,
            **kwargs,
        )
        fpath = self.get_fpath(file_name)
        if ret_data:
            r[0].write_image(fpath)
            return r[1]
        else:
            r.write_image(fpath)


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

    def get_figure(
            self,
            turbine,
            variable,
            ws_bins,
            wd_sectors=12,
            wd_var=FV.AMB_WD,
            ws_var=FV.AMB_REWS,
            contraction="weights",
            fig=None,
            ax=None,
            title=None,
            figsize=None,
            **kwargs,
        ):
        """
        Gather the plot data
        
        Parameters
        ----------
        turbine: int
            The turbine index
        variable: str   
            The variable name
        ws_bins: list of float
            The wind speed bins
        wd_var: str
            The wind direction variable
        ws_var: str
            The wind speed variable
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
        kwargs: dict, optional
            Additional parameters for plt.ax_hist2d
        
        Returns
        -------
        ax: pyplot.Axes
            The axes object
        
        """
        var = self.farm_results[variable].to_numpy()[:, turbine]
        w = self.farm_results[FV.WEIGHT].to_numpy()[:, turbine]
        ws = self.farm_results[ws_var].to_numpy()[:, turbine]
        wd = self.farm_results[wd_var].to_numpy()[:, turbine].copy()
        wd_delta = 360/wd_sectors
        wd[wd>=360-wd_delta/2] -= 360
        wd = np.mod(90 - wd, 360)
        wd = np.radians(wd)
        wd_bins = np.arange(-wd_delta/2, 360, wd_delta)
        wd_bins = np.radians(wd_bins)

        if ax is not None:
            if not isinstance(ax, PolarAxes):
                raise TypeError(f"Require axes of type '{PolarAxes.__name__}' for '{type(self).__name__}', got '{type(ax).__name__}'")
        else:
            fig, ax = plt.subplots(
                figsize=figsize, subplot_kw={"projection": "polar"})
        
        y, x = np.meshgrid(ws_bins, wd_bins)
        if contraction == "weights":
            z = np.histogram2d(wd, ws, (wd_bins, ws_bins), weights=w)[0]
            z = np.histogram2d(wd, ws, (wd_bins, ws_bins), weights=w*var)[0] / z
        elif contraction == "mean_no_weights":
            z = np.histogram2d(wd, ws, (wd_bins, ws_bins))[0]
            z = np.histogram2d(wd, ws, (wd_bins, ws_bins), weights=var)[0] / z
        elif contraction == "sum_no_weights":
            z = np.histogram2d(wd, ws, (wd_bins, ws_bins), weights=var)[0]
        else:
            raise KeyError(f"Contraction '{contraction}' not supported. Choices: weights, mean_no_weights, sum_no_weights")

        prgs = {"shading": "flat"}
        prgs.update(kwargs)

        img = ax.pcolormesh(x, y, z, **prgs)

        tksl = np.arange(0,360,wd_delta)
        tks = np.radians(np.mod(90 - tksl, 360))
        ax.set_xticks(tks, [f"{d}°" for d in tksl])
        ax.set_yticks(ws_bins)
        ax.set_title(title)
        cbar = fig.colorbar(img, ax=ax, pad=0.12)
        cbar.ax.set_title(variable)
        
        return ax
