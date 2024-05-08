from tabnanny import verbose
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from foxes.input.states import StatesTable
from foxes.core import WindFarm, Turbine
from foxes.algorithms import Downwind
from foxes.models.turbine_models import SetFarmVars
import foxes.variables as FV
import foxes.constants as FC
from .output import Output


class TurbineTypeCurves(Output):
    """
    Creates power and ct curves for turbine
    types, optionally including derating/boost.

    Attributes
    ----------
    mbook: foxes.models.ModelBook
        The model book

    :group: output

    """

    def __init__(self, mbook):
        """
        Constructor.

        Parameters
        ----------
        mbook: foxes.models.ModelBook
            The model book

        """
        self.mbook = mbook

    def plot_curves(
        self,
        turbine_type,
        variables,
        P_max=None,
        titles=None,
        x_label=None,
        y_labels=None,
        ws_min=0.0,
        ws_max=30.0,
        ws_step=0.1,
        ti=0.05,
        rho=1.225,
        axs=None,
        figsize=None,
        pmax_args={},
        **kwargs,
    ):
        """
        Plot the power or ct curve.

        Parameters
        ----------
        turbine_type: str
            The turbine type name from the
            model book
        variables: str or list of str
            For example FV.P or FV.CT
        P_max: float, optional
            The power mask value, if of interest
        titles: list of str, optional
            The plot titles, one for each variable
        x_label: str, optional
            The x axis label
        y_labels: list of str, optional
            The y axis lables, one for each variable
        ws_min: float
            The minimal wind speed
        ws_max: float
            The maximal wind speed
        ws_step: float
            The wind speed step size
        ti: float
            The TI value
        rho: float
            The air density value
        axs: list of pyplot.Axis, optional
            The axis, one for each variable
        figsize: tuple
            The figsize argument for plt.subplots()
            in case ax is not provided
        pmax_args: dict, optional
            Additionals parameters for plt.plot()
            for power mask case
        kwargs: dict, optional
            Additional parameters for plt.plot()

        Returns
        -------
        axs: list of pyplot.Axis
            The plot axes, one for each variable

        """
        vars = [variables] if isinstance(variables, str) else variables
        if isinstance(titles, str):
            titles = [titles]
        elif titles is None:
            titles = [None for v in vars]
        if isinstance(y_labels, str):
            y_labels = [y_labels]
        elif y_labels is None:
            y_labels = [None for v in vars]
        if not isinstance(axs, (list, tuple, np.ndarray)):
            axs = [axs]

        ws = np.arange(ws_min, ws_max + ws_step, ws_step, dtype=FC.DTYPE)
        n_states = len(ws)
        sdata = pd.DataFrame(index=range(n_states))
        sdata.index.name = FC.STATE
        sdata[FV.WS] = ws

        models = [turbine_type]

        states = StatesTable(
            sdata,
            output_vars={FV.WS, FV.WD, FV.TI, FV.RHO},
            fixed_vars={FV.WD: 270.0, FV.TI: ti, FV.RHO: rho},
        )

        farm = WindFarm()
        farm.add_turbine(
            Turbine(xy=[0.0, 0.0], turbine_models=models),
            verbosity=0,
        )

        algo = Downwind(farm, states, wake_models=[], mbook=self.mbook, verbosity=0)

        results = algo.calc_farm()

        if P_max is not None:
            sname = f"_{type(self).__name__}_set_Pmax"
            self.mbook.turbine_models[sname] = SetFarmVars()
            self.mbook.turbine_models[sname].add_var(FV.MAX_P, P_max)
            models += [sname, "PMask"]

            farm = WindFarm()
            farm.add_turbine(
                Turbine(xy=[0.0, 0.0], turbine_models=models),
                verbosity=0,
            )

            algo = Downwind(farm, states, wake_models=[], mbook=self.mbook, verbosity=0)

            results1 = algo.calc_farm()

            del self.mbook.turbine_models[sname]

        for i, v in enumerate(vars):
            ax = axs[i]
            if ax is None:
                __, ax = plt.subplots(figsize=figsize)

            pargs = {"linewidth": 2.5}
            pargs.update(kwargs)
            ax.plot(ws, results[v][:, 0], label="default", **pargs)
            if P_max is not None:
                pargs = {"linewidth": 1.8}
                pargs.update(pmax_args)
                ax.plot(ws, results1[v][:, 0], label="PMask", **pargs)

            if v == FV.P:
                vv = "Power curve"
            elif v == FV.CT:
                vv = "Thrust curve"
            else:
                vv = v
            t = f"{vv}, {turbine_type}" if titles[i] is None else titles[i]
            ax.set_title(t)

            l = "Wind speed [m/s]" if x_label is None else x_label
            ax.set_xlabel(l)

            if y_labels[i] is None:
                if v == FV.P:
                    l = f"Power [kW]"
                elif v == FV.CT:
                    l = "ct [-]"
                else:
                    l = v
            else:
                l = y_labels[i]
            ax.set_ylabel(l)

            ax.grid()

            if P_max is not None:
                ax.legend()

        return axs
