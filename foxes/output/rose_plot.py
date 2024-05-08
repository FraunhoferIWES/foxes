import numpy as np
import pandas as pd

import foxes.variables as FV
import foxes.constants as FC
from foxes.utils import wd2uv, uv2wd, TabWindroseAxes
from foxes.algorithms import Downwind
from foxes.core import WindFarm, Turbine
from foxes.models import ModelBook
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

    def __init__(self, results):
        """
        Constructor.

        Parameters
        ----------
        results: xarray.Dataset
            The calculation results (farm or points)

        """
        dims = list(results.sizes.keys())
        if dims[1] == FC.TURBINE:
            self._rtype = FC.TURBINE
        elif dims[1] == FC.POINT:
            self._rtype = FC.POINT
        else:
            raise KeyError(
                f"Results dimension 1 is neither '{FC.TURBINE}' nor '{FC.POINT}': dims = {results.dims}"
            )

        self.results = results.to_dataframe()

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
            The variable bin seperation values
        wd_var: str, optional
            The wind direction variable name
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

        data = self.results[[wd_var, FV.WEIGHT]].copy()
        data[lgd] = self.results[var]
        uv = wd2uv(data[wd_var].to_numpy())
        data["u"] = uv[:, 0]
        data["v"] = uv[:, 1]

        data[FV.WEIGHT] *= 100
        data = data.rename(columns={FV.WEIGHT: "frequency"})

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
        sectors,
        var,
        var_bins,
        wd_var=FV.AMB_WD,
        turbine=None,
        point=None,
        title=None,
        legend=None,
        design="bar",
        start0=False,
        fig=None,
        figsize=None,
        rect=None,
        ret_data=False,
        **kwargs,
    ):
        """
        Creates figure object

        Parameters
        ----------
        sectors: int
            The number of wind direction sectors
        var: str
            The data variable name
        var_bins: list of float
            The variable bin seperation values
        wd_var: str, optional
            The wind direction variable name
        turbine: int, optional
            Only relevant in case of farm results.
            If None, mean over all turbines.
            Else, data from a single turbine
        point: int, optional
            Only relevant in case of point results.
            If None, mean over all points.
            Else, data from a single point
        title. str, optional
            The title
        legend: str, optional
            The data legend string
        design: str
            The wind rose design: bar, contour, ...
        start0: bool
            Flag for starting the first sector at
            zero degrees instead of minus half width
        fig: matplotlib.Figure
            The figure to which to add an axis
        figsize: tuple, optional
            The figsize of the newly created figure
        rect: list, optional
            The rectangle of the figure which to fill,
            e.g. [0.1, 0.1, 0.8, 0.8]
        ret_data: bool
            Flag for returning wind rose data
        kwargs: dict, optional
            Additional arguments for TabWindroseAxes
            plot function

        Returns
        -------
        fig: matplotlib.pyplot.Figure
            The rose plot figure
        data: pd.DataFrame, optional
            The wind rose data

        """
        lg = legend
        if title is None or legend is None:
            ttl, lg = self.get_data_info(var)
        if title is not None:
            ttl = title

        wrdata = self.get_data(
            sectors=sectors,
            var=var,
            var_bins=var_bins,
            wd_var=wd_var,
            turbine=turbine,
            point=point,
            start0=start0,
        )

        ax = TabWindroseAxes.from_ax(fig=fig, rect=rect, figsize=figsize)
        fig = ax.get_figure()

        plfun = getattr(ax, design)
        plfun(
            direction=wrdata[wd_var].to_numpy(),
            var=wrdata[var].to_numpy(),
            weights=wrdata["frequency"].to_numpy(),
            bin_min_dir=np.sort(wrdata[f"bin_min_{wd_var}"].unique()),
            bin_min_var=np.sort(wrdata[f"bin_min_{var}"].unique()),
            bin_max_var=np.sort(wrdata[f"bin_max_{var}"].unique()),
            **kwargs,
        )
        ax.set_legend(title=lg)
        ax.set_title(ttl)

        if ret_data:
            return fig, wrdata
        else:
            return fig

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
            Path to the output file
        sectors: int
            The number of wind direction sectors
        var: str
            The data variable name
        var_bins: list of float
            The variable bin seperation values
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
        if ret_data:
            r[0].write_image(file_name)
            return r[1]
        else:
            r.write_image(file_name)


class StatesRosePlotOutput(RosePlotOutput):
    """
     Class for rose plot creation directly from states

     Parameters
     ----------
     states: foxes.core.States
         The states from which to compute the wind rose
     point: numpy.ndarray
         The evaluation point, shape: (3,)
     mbook: foxes.models.ModelBook, optional
         The model book

    :group: output

    """

    def __init__(self, states, point, mbook=None, ws_var=FV.AMB_REWS):
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

        super().__init__(results)
