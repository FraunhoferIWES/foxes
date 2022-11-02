import numpy as np
import pandas as pd
import plotly.express as px

import foxes.variables as FV
from foxes.utils import wd2uv, uv2wd
from foxes.algorithms import Downwind
from foxes.core import WindFarm, Turbine
from foxes.models import ModelBook
from .output import Output


class RosePlotOutput(Output):
    """
    Class for rose plot creation

    Parameters
    ----------
    results : xarray.Dataset
        The calculation results (farm or points)

    Attributes
    ----------
    results : pandas.DataFrame
        The calculation results (farm or points)

    """

    def __init__(self, results):

        dims = list(results.dims.keys())
        if dims[1] == FV.TURBINE:
            self._rtype = FV.TURBINE
        elif dims[1] == FV.POINT:
            self._rtype = FV.POINT
        else:
            raise KeyError(
                f"Results dimension 1 is neither '{FV.TURBINE}' nor '{FV.POINT}': dims = {results.dims}"
            )

        self.results = results.to_dataframe()

    @classmethod
    def get_data_info(cls, dname):
        """
        Returns default description for a variable.

        Parameters
        ----------
        dname : str
            The variable name

        Returns
        -------
        title : str
            The long name of the variable
        legend : str
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
        legend=None,
    ):
        """
        Get pandas DataFrame with wind rose data.

        Parameters
        ----------
        sectors : int
            The number of wind direction sectors
        var : str
            The data variable name
        var_bins : list of float
            The variable bin seperation values
        wd_var : str, optional
            The wind direction variable name
        turbine : int, optional
            Only relevant in case of farm results.
            If None, mean over all turbines.
            Else, data from a single turbine
        point : int, optional
            Only relevant in case of point results.
            If None, mean over all points.
            Else, data from a single point
        legend : str, optional
            The data legend string

        Returns
        -------
        pd.DataFrame :
            The wind rose data

        """

        dwd = 360.0 / sectors
        wds = np.arange(0.0, 360.0, dwd)
        wdb = np.arange(-dwd / 2, 360.0, dwd)
        lgd = legend if legend is not None else var

        data = self.results[[wd_var, FV.WEIGHT]].copy()
        data[lgd] = self.results[var]
        uv = wd2uv(data[wd_var].to_numpy())
        data["u"] = uv[:, 0]
        data["v"] = uv[:, 1]

        data[FV.WEIGHT] *= 100
        data = data.rename(columns={FV.WEIGHT: "frequency"})

        el = turbine if self._rtype == FV.TURBINE else point
        if el is None:
            data = data.groupby(level=0).mean()
        else:
            sname = data.index.names[0]
            grp = data.reset_index().groupby(self._rtype)
            data = grp.get_group(el).set_index(sname)

        data["wd"] = uv2wd(data[["u", "v"]].to_numpy())
        data.drop(["u", "v"], axis=1, inplace=True)
        data.loc[data["wd"] > 360.0 - dwd / 2, "wd"] -= 360.0

        data[wd_var] = pd.cut(data["wd"], wdb, labels=wds)
        data[lgd] = pd.cut(data[lgd], var_bins, right=False, include_lowest=True)

        grp = data[[wd_var, lgd, "frequency"]].groupby([wd_var, lgd])
        data = grp.sum().reset_index()

        return data

    def get_figure(
        self,
        sectors,
        var,
        var_bins,
        wd_var=FV.AMB_WD,
        turbine=None,
        point=None,
        cmap="Turbo",
        title=None,
        legend=None,
        layout_dict={},
        title_dict={},
    ):
        """
        Creates px figure object

        Parameters
        ----------
        sectors : int
            The number of wind direction sectors
        var : str
            The data variable name
        var_bins : list of float
            The variable bin seperation values
        wd_var : str, optional
            The wind direction variable name
        turbine : int, optional
            Only relevant in case of farm results.
            If None, mean over all turbines.
            Else, data from a single turbine
        point : int, optional
            Only relevant in case of point results.
            If None, mean over all points.
            Else, data from a single point
        legend : str, optional
            The data legend string
        layout_dict : dict, optional
            Optional parameters for the px figure layout
        title_dict : dict, optional
            Optional parameters for the px title layout

        Returns
        -------
        px.Figure :
            The rose plot figure

        """

        if title is None or legend is None:
            ttl, lg = self.get_data_info(var)
        if title is not None:
            ttl = title
        if legend is not None:
            lg = legend

        wrdata = self.get_data(
            sectors=sectors,
            var=var,
            var_bins=var_bins,
            wd_var=wd_var,
            turbine=turbine,
            point=point,
            legend=lg,
        )

        cols = px.colors.sequential.__dict__.keys()
        cols = [c for c in cols if isinstance(px.colors.sequential.__dict__[c], list)]
        cmap = px.colors.sequential.__dict__[cols[cols.index(cmap)]]

        fig = px.bar_polar(
            wrdata, r="frequency", theta=wd_var, color=lg, color_discrete_sequence=cmap
        )

        tdict = dict(xanchor="center", yanchor="top", x=0.5, y=0.97)
        tdict.update(title_dict)

        ldict = dict(width=500, height=400, margin=dict(l=50, r=50, t=50, b=50))
        ldict.update(layout_dict)

        fig = fig.update_layout(title={"text": ttl, **tdict}, **ldict)

        return fig

    def write_figure(
        self,
        file_name,
        sectors,
        var,
        var_bins,
        wd_var=FV.AMB_WD,
        turbine=None,
        point=None,
        cmap="Turbo",
        title=None,
        legend=None,
        layout_dict={},
        title_dict={},
    ):
        """
        Write rose plot to file

        Parameters
        ----------
        file_name : str
            Path to the output file
        sectors : int
            The number of wind direction sectors
        var : str
            The data variable name
        var_bins : list of float
            The variable bin seperation values
        wd_var : str, optional
            The wind direction variable name
        turbine : int, optional
            Only relevant in case of farm results.
            If None, mean over all turbines.
            Else, data from a single turbine
        point : int, optional
            Only relevant in case of point results.
            If None, mean over all points.
            Else, data from a single point
        legend : str, optional
            The data legend string
        layout_dict : dict, optional
            Optional parameters for the px figure layout
        title_dict : dict, optional
            Optional parameters for the px title layout

        """

        fig = self.get_figure(
            sectors=sectors,
            var=var,
            var_bins=var_bins,
            wd_var=wd_var,
            turbine=turbine,
            point=point,
            cmap=cmap,
            title=title,
            legend=legend,
            layout_dict=layout_dict,
            title_dict=title_dict,
        )

        fig.write_image(file_name)


class StatesRosePlotOutput(RosePlotOutput):
    """
    Class for rose plot creation directly from states

    Parameters
    ----------
    states : foxes.core.States
        The states from which to compute the wind rose
    point : numpy.ndarray
        The evaluation point, shape: (3,)
    mbook : foxes.models.ModelBook, optional
        The model book

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
        algo = Downwind(mbook, farm, states, wake_models=[], verbosity=0)

        results = algo.calc_farm(ambient=True).rename_vars({ws_var: FV.AMB_WS})

        algo.finalize()
        states.finalize(algo, results, clear_mem=False, verbosity=0)

        super().__init__(results)
