import numpy as np
import pandas as pd
from cycler import cycler
import matplotlib.pyplot as plt

from .output import Output
from foxes.config import config
import foxes.variables as FV
import foxes.constants as FC


class FarmResultsEval(Output):
    """
    Evaluates farm results data.

    This sums over turbines and/or states,
    given the state-turbine farm_calc results.

    Attributes
    ----------
    results: xarray.Dataset
        The farm results

    :group: output

    """

    def __init__(self, farm_results):
        """
        Constructor.

        Parameters
        ----------
        farm_results: xarray.Dataset
            The farm results

        """
        self.results = farm_results

    def weinsum(self, rhs, *vars):
        """
        Calculates Einstein sum, adding weights
        as last argument to the given fields.

        It's all about treating NaN values.

        Parameters
        ----------
        rhs: str
            The right-hand side of the einsum expression.
            Convention: 's' for states, 't' for turbines
        vars: tuple of str or np.ndarray
            The variables mentioned in the expression,
            but without the obligatory weights that will
            be added at the end

        Returns
        -------
        result: np.ndarray
            The results array

        """
        nas = None
        fields = []
        for v in vars:
            if isinstance(v, str):
                vdata = self.results[v].to_numpy()
                nns = np.sum(np.isnan(vdata))
                assert nns == 0, (
                    f"Found {nns} nan values for variable '{v}' of shape {vdata.shape}"
                )
                fields.append(vdata)
            else:
                fields.append(v)
            if nas is None:
                nas = np.zeros_like(fields[-1], dtype=bool)
            nas = nas | np.isnan(fields[-1])

        inds = ["st" for __ in fields]
        if self.results[FV.WEIGHT].dims == (FC.STATE,):
            inds += ["s"]

            if np.any(nas):
                sel = ~np.any(nas, axis=1)
                fields = [f[sel] for f in fields]

                weights0 = self.results[FV.WEIGHT].to_numpy()
                w0 = np.sum(weights0)
                weights = weights0[sel]
                w1 = np.sum(weights)
                weights *= w0 / w1
                fields.append(weights)

            else:
                fields.append(self.results[FV.WEIGHT].to_numpy())

        elif self.results[FV.WEIGHT].dims == (FC.STATE, FC.TURBINE):
            inds += ["st"]

            if np.any(nas):
                sel = ~np.any(nas, axis=1)
                fields = [f[sel] for f in fields]

                weights0 = self.results[FV.WEIGHT].to_numpy()
                w0 = np.sum(weights0, axis=0)[None, :]
                weights = weights0[sel]
                w1 = np.sum(weights, axis=0)[None, :]
                weights *= w0 / w1
                fields.append(weights)

            else:
                fields.append(self.results[FV.WEIGHT].to_numpy())

        else:
            raise ValueError(
                f"Expecting '{FV.WEIGHT}' variable with dimensions {(FC.STATE,)} or {(FC.STATE, FC.TURBINE)}, got {self.results[FV.WEIGHT].dims}"
            )
        expr = ",".join(inds) + "->" + rhs

        return np.einsum(expr, *fields)

    def reduce_states(self, vars_op):
        """
        Reduces the states dimension by some operation

        Parameters
        ----------
        vars_op: dict
            The operation per variable. Key: str, the variable
            name. Value: str, the operation, choices
            are: weights, mean_no_weights, sum, min, max.

        Returns
        -------
        data: pandas.DataFrame
            The results per turbine

        """
        n_turbines = self.results.sizes[FC.TURBINE]

        rdata = {}
        for v, op in vars_op.items():
            vdata = self.results[v].to_numpy()
            nns = np.sum(np.isnan(vdata))
            assert nns == 0, (
                f"Found {nns} nan values for variable '{v}' of shape {vdata.shape}"
            )

            if op == "weights":
                rdata[v] = self.weinsum("t", vdata)
            elif op == "mean_no_weights":
                rdata[v] = np.mean(vdata, axis=0)
            elif op == "sum":
                rdata[v] = np.sum(vdata, axis=0)
            elif op == "min":
                rdata[v] = np.min(vdata, axis=0)
            elif op == "max":
                rdata[v] = np.max(vdata, axis=0)
            elif op == "std":
                rdata[v] = np.std(vdata, axis=0)
            else:
                raise KeyError(
                    f"Unknown operation '{op}' for variable '{v}'. Please choose: weights, mean_no_weights, sum, min, max"
                )

        data = pd.DataFrame(index=range(n_turbines), data=rdata)
        data.index.name = FC.TURBINE

        return data

    def reduce_turbines(self, vars_op):
        """
        Reduces the turbine dimension by some operation

        Parameters
        ----------
        vars_op: dict
            The operation per variable. Key: str, the variable
            name. Value: str, the operation, choices
            are: weights, mean_no_weights, sum, min, max.

        Returns
        -------
        data: pandas.DataFrame
            The results per state

        """
        states = self.results.coords[FC.STATE].to_numpy()

        rdata = {}
        for v, op in vars_op.items():
            vdata = self.results[v].to_numpy()
            nns = np.sum(np.isnan(vdata))
            assert nns == 0, (
                f"Found {nns} nan values for variable '{v}' of shape {vdata.shape}"
            )

            if op == "weights":
                rdata[v] = self.weinsum("s", vdata)
            elif op == "mean_no_weights":
                rdata[v] = np.mean(vdata, axis=1)
            elif op == "sum":
                rdata[v] = np.sum(vdata, axis=1)
            elif op == "min":
                rdata[v] = np.min(vdata, axis=1)
            elif op == "max":
                rdata[v] = np.max(vdata, axis=1)
            else:
                raise KeyError(
                    f"Unknown operation '{op}' for variable '{v}'. Please choose: weights, mean_no_weights, sum, min, max"
                )

        data = pd.DataFrame(index=states, data=rdata)
        data.index.name = FC.STATE

        return data

    def reduce_all(self, states_op, turbines_op):
        """
        Reduces states and turbine dimension by some operation

        Parameters
        ----------
        states_op: dict
            The states contraction operations.
            Key: str, the variable name. Value:
            str, the operation, choices are:
            sum, mean, min, max.
        turbines_op: dict
            The turbines contraction operations.
            Key: str, the variable name. Value:
            str, the operation, choices are:
            sum, mean, min, max.

        Returns
        -------
        data: dict
            The fully contracted results

        """
        sdata = self.reduce_states(states_op)

        rdata = {}
        for v, op in turbines_op.items():
            vdata = sdata[v].to_numpy()
            nns = np.sum(np.isnan(vdata))
            assert nns == 0, (
                f"Found {nns} nan values for variable '{v}' of shape {vdata.shape}"
            )

            if op == "weights":
                if states_op[v] == "weights":
                    rdata[v] = self.weinsum("", v)
                else:
                    rdata[v] = self.weinsum("", vdata[None, :])
            elif op == "mean_no_weights":
                rdata[v] = np.sum(vdata)
            elif op == "sum":
                rdata[v] = np.sum(vdata)
            elif op == "min":
                rdata[v] = np.min(vdata)
            elif op == "max":
                rdata[v] = np.max(vdata)
            else:
                raise KeyError(
                    f"Unknown operation '{op}' for variable '{v}'. Please choose: sum, mean, min, max, weights"
                )

        return rdata

    def calc_states_mean(self, vars, use_weights=True):
        """
        Calculates the mean wrt states.

        Parameters
        ----------
        vars: list of str
            The variables
        use_weights: bool
            Flag for using states weights for the mean

        Returns
        -------
        data: pandas.DataFrame
            The results per turbine

        """
        r = "weights" if use_weights else "mean_no_weights"
        if isinstance(vars, str):
            return self.reduce_states({vars: r})
        return self.reduce_states({v: r for v in vars})

    def calc_states_sum(self, vars):
        """
        Calculates the sum wrt states.

        Parameters
        ----------
        vars: list of str
            The variables

        Returns
        -------
        data: pandas.DataFrame
            The results per turbine

        """
        return self.reduce_states({v: "sum" for v in vars})

    def calc_states_std(self, vars):
        """
        Calculates the standard deviation wrt states.

        Args:
            vars (_type_): _description_

        Returns:
            _type_: _description_
        """

        return self.reduce_states({v: "std" for v in vars})

    def calc_turbine_mean(self, vars):
        """
        Calculates the mean wrt turbines.

        Parameters
        ----------
        vars: list of str
            The variables

        Returns
        -------
        data: pandas.DataFrame
            The results per state

        """
        return self.reduce_turbines({v: "mean_no_weights" for v in vars})

    def calc_turbine_sum(self, vars):
        """
        Calculates the sum wrt turbines.

        Parameters
        ----------
        vars: list of str
            The variables

        Returns
        -------
        data: pandas.DataFrame
            The results per state

        """
        return self.reduce_turbines({v: "sum" for v in vars})

    def calc_farm_mean(self, vars):
        """
        Calculates the mean over states and turbines.

        Parameters
        ----------
        vars: list of str
            The variables

        Returns
        -------
        data: dict
            The fully contracted results

        """
        op = {v: "weights" for v in vars}
        return self.reduce_all(states_op=op, turbines_op=op)

    def calc_farm_sum(self, vars):
        """
        Calculates the sum over states and turbines.

        Parameters
        ----------
        vars: list of str
            The variables

        Returns
        -------
        data: dict
            The fully contracted results

        """
        op = {v: "sum" for v in vars}
        return self.reduce_all(states_op=op, turbines_op=op)

    def calc_mean_farm_power(self, ambient=False):
        """
        Calculates the mean total farm power.

        Parameters
        ----------
        ambient: bool
            Flag for ambient power

        Returns
        -------
        data: float
            The mean wind farm power

        """
        v = FV.P if not ambient else FV.AMB_P
        cdata = self.reduce_all(states_op={v: "weights"}, turbines_op={v: "sum"})
        return cdata[v]

    def calc_turbine_yield(
        self,
        algo=None,
        annual=False,
        ambient=False,
        hours=None,
        delta_t=None,
        P_unit_W=None,
    ):
        """
        Calculates the yield per turbine

        Parameters
        ----------
        algo: foxes.core.Algorithm, optional
            The algorithm, for P_nominal lookup
        annual: bool, optional
            Flag for returning annual results, by default False
        ambient: bool, optional
            Flag for ambient power, by default False
        hours: int, optional
            The duration time in hours, if not timeseries states
        delta_t: np.datetime64, optional
            The time delta step in case of time series data,
            by default automatically determined
        P_unit_W: float
            The power unit in Watts, 1000 for kW. Looked up
            in algorithm if not given

        Returns
        -------
        pandas.DataFrame
            A dataframe of yield values by turbine in GWh

        """
        if ambient:
            var_in = FV.AMB_P
            var_out = FV.AMB_YLD
        else:
            var_in = FV.P
            var_out = FV.YLD

        if algo is not None and P_unit_W is None:
            P_unit_W = np.array(
                [FC.P_UNITS[t.P_unit] for t in algo.farm_controller.turbine_types],
                dtype=config.dtype_double,
            )[:, None]
        elif algo is None and P_unit_W is not None:
            pass
        else:
            raise KeyError("Expecting either 'algo' or 'P_unit_W'")

        # compute yield per turbine
        if np.issubdtype(self.results[FC.STATE].dtype, np.datetime64):
            if hours is not None:
                raise KeyError("Unexpected parameter 'hours' for timeseries data")
            times = self.results[FC.STATE].to_numpy()
            if delta_t is None:
                delta_t = times[-1] - times[-2]
            duration = times[-1] - times[0] + delta_t
            duration_seconds = np.int64(duration.astype(np.int64) / 1e9)
            duration_hours = duration_seconds / 3600
        elif hours is None and annual:
            duration_hours = 8760
        elif hours is None:
            raise ValueError(
                "Expecting parameter 'hours' for non-timeseries data, or 'annual=True'"
            )
        else:
            duration_hours = hours
        yld = self.calc_states_mean(var_in) * duration_hours * P_unit_W / 1e9

        if annual:
            # convert to annual values
            yld *= 8760 / duration_hours

        yld.rename(columns={var_in: var_out}, inplace=True)
        return yld

    def add_capacity(self, algo=None, P_nom=None, ambient=False, verbosity=1):
        """
        Adds capacity to the farm results

        Parameters
        ----------
        algo: foxes.core.Algorithm, optional
            The algorithm, for nominal power calculation
        P_nom: list of float, optional
            Nominal power values for each turbine, if algo not given
        ambient: bool, optional
            Flag for calculating ambient capacity, by default False
        verbosity: int
            The verbosity level, 0 = silent

        """
        if ambient:
            var_in = FV.AMB_P
            var_out = FV.AMB_CAP
        else:
            var_in = FV.P
            var_out = FV.CAP

        # get results data for the vars variable (by state and turbine)
        vdata = self.results[var_in]

        if algo is not None and P_nom is None:
            P_nom = np.array(
                [t.P_nominal for t in algo.farm_controller.turbine_types],
                dtype=config.dtype_double,
            )
        elif algo is None and P_nom is not None:
            P_nom = np.array(P_nom, dtype=config.dtype_double)
        else:
            raise KeyError("Expecting either 'algo' or 'P_nom'")

        # add to farm results
        self.results[var_out] = vdata / P_nom[None, :]
        if verbosity > 0:
            if ambient:
                print("Ambient capacity added to farm results")
            else:
                print("Capacity added to farm results")

    def calc_farm_yield(self, turbine_yield=None, power_uncert=None, **kwargs):
        """
        Calculates yield, P75 and P90 at the farm level

        Parameters
        ----------
        turbine_yield: pandas.DataFrame, optional
            Yield values by turbine
        power_uncert: float, optional
            Uncertainty in the power value. Triggers
            P75 and P90 outputs
        kwargs: dict, optional
            Parameters for calc_turbine_yield(). Apply if
            turbine_yield is not given

        Returns
        -------
        farm_yield: float
            Farm yield result, same unit as turbine yield
        P75: float, optional
            The P75 value, same unit as turbine yield
        P90: float, optional
            The P90 value, same unit as turbine yield

        """
        if turbine_yield is None:
            yargs = dict(annual=True)
            yargs.update(kwargs)
            turbine_yield = self.calc_turbine_yield(**yargs)
        farm_yield = turbine_yield.sum()

        if power_uncert is not None:
            P75 = farm_yield * (1.0 - (0.675 * power_uncert))
            P90 = farm_yield * (1.0 - (1.282 * power_uncert))
            return farm_yield["YLD"], P75["YLD"], P90["YLD"]

        return farm_yield["YLD"]

    def add_efficiency(self, verbosity=1):
        """
        Adds efficiency to the farm results

        Parameters
        ----------
        verbosity: int
            The verbosity level, 0 = silent

        """
        P = self.results[FV.P].to_numpy()
        P0 = np.maximum(self.results[FV.AMB_P].to_numpy(), 1e-12)
        eff = np.minimum(P / P0, 1)
        eff[P < 1e-10] = 0
        self.results[FV.EFF] = (self.results[FV.AMB_P].dims, eff)
        if verbosity > 0:
            print("Efficiency added to farm results")

    def calc_farm_efficiency(self):
        """
        Calculates farm efficiency

        Returns
        -------
        eff: float
            The farm efficiency

        """
        P = self.calc_mean_farm_power()
        P0 = np.maximum(self.calc_mean_farm_power(ambient=True), 1e-14)
        return np.minimum(P / P0, 1)

    def gen_stdata(
        self,
        turbines,
        variable,
        fig=None,
        ax=None,
        figsize=None,
        legloc="lower right",
        animated=True,
        ret_im=True,
    ):
        """
        Generates state-turbine data,
        intended to be used in animations

        Parameters
        ----------
        turbines: list of int
            The turbines for which to scatter data
        variable: str
            The variable name
        fig: plt.Figure, optional
            The figure object
        ax: plt.Axes, optional
            The figure axes
        figsize: tuple, optional
            The figsize for plt.Figure
        legloc: str
            The legend location
        animated: bool
            Flag for animated output
        ret_im: bool
            Flag for image return,

        Yields
        ------
        fig: matplotlib.Figure
            The figure object
        im: List of matplotlib.collections.PathCollection, optional
            The scatter artists

        """

        if fig is None:
            hfig = plt.figure(figsize=figsize)
        else:
            hfig = fig
        if ax is None:
            hax = hfig.add_subplot(111)
        else:
            hax = ax

        hax.set_xlabel("State")
        hax.set_ylabel(variable)
        cc = cycler(color="bgrcmyk")

        data = self.results[variable].to_numpy()
        hasl = set()
        for si in range(len(data)):
            im = []
            hax.set_prop_cycle(cc)
            for ti in turbines:
                lbl = None if ti in hasl else f"Turbine {ti}"
                im += hax.plot(range(si), data[:si, ti], label=lbl, animated=animated)
                hasl.add(ti)

            hax.legend(loc=legloc)

            if ret_im:
                yield hfig, im
            else:
                yield hfig
