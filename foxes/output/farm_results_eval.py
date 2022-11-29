import numpy as np
import pandas as pd

from .output import Output
import foxes.variables as FV


class FarmResultsEval(Output):
    """
    Evaluates farm results data.

    This sums over turbines and/or states,
    given the state-turbine farm_calc results.

    Parameters
    ----------
    farm_results : xarray.Dataset
        The farm results

    Attributes
    ----------
    results : xarray.Dataset
        The farm results

    """

    def __init__(self, farm_results):
        self.results = farm_results

    def reduce_states(self, vars_op):
        """
        Reduces the states dimension by some operation

        Parameters
        ----------
        vars_op : dict
            The operation per variable. Key: str, the variable
            name. Value: str, the operation, choices
            are: sum, mean, min, max.

        Returns
        -------
        data : pandas.DataFrame
            The results per turbine

        """
        weights = self.results[FV.WEIGHT].to_numpy()
        n_turbines = weights.shape[1]

        rdata = {}
        for v, op in vars_op.items():
            vdata = self.results[v].to_numpy()
            if op == "mean":
                rdata[v] = np.einsum("st,st->t", vdata, weights)
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
                    f"Unknown operation '{op}' for variable '{v}'. Please choose: sum, mean, min, max"
                )

        data = pd.DataFrame(index=range(n_turbines), data=rdata)
        data.index.name = FV.TURBINE

        return data

    def reduce_turbines(self, vars_op):
        """
        Reduces the turbine dimension by some operation

        Parameters
        ----------
        vars_op : dict
            The operation per variable. Key: str, the variable
            name. Value: str, the operation, choices
            are: sum, mean, min, max.

        Returns
        -------
        data : pandas.DataFrame
            The results per state

        """
        weights = self.results[FV.WEIGHT].to_numpy()
        states = self.results.coords[FV.STATE].to_numpy()

        rdata = {}
        for v, op in vars_op.items():
            vdata = self.results[v].to_numpy()
            if op == "mean":
                rdata[v] = np.einsum("st,st->s", vdata, weights)
            elif op == "sum":
                rdata[v] = np.sum(vdata, axis=1)
            elif op == "min":
                rdata[v] = np.min(vdata, axis=1)
            elif op == "max":
                rdata[v] = np.max(vdata, axis=1)
            else:
                raise KeyError(
                    f"Unknown operation '{op}' for variable '{v}'. Please choose: sum, mean, min, max"
                )

        data = pd.DataFrame(index=states, data=rdata)
        data.index.name = FV.STATE

        return data

    def reduce_all(self, states_op, turbines_op):
        """
        Reduces states and turbine dimension by some operation

        Parameters
        ----------
        states_op : dict
            The states contraction operations.
            Key: str, the variable name. Value:
            str, the operation, choices are:
            sum, mean, min, max.
        turbines_op : dict
            The turbines contraction operations.
            Key: str, the variable name. Value:
            str, the operation, choices are:
            sum, mean, min, max.

        Returns
        -------
        data : dict
            The fully contracted results

        """
        weights = self.results[FV.WEIGHT].to_numpy()
        sdata = self.reduce_states(states_op)

        rdata = {}
        for v, op in turbines_op.items():
            vdata = sdata[v].to_numpy()
            if op == "mean":
                if states_op[v] == "mean":
                    vdata = self.results[v].to_numpy()
                    rdata[v] = np.einsum("st,st->", vdata, weights)
                else:
                    rdata[v] = np.einsum("st,st->", vdata[None, :], weights)
            elif op == "sum":
                rdata[v] = np.sum(vdata)
            elif op == "min":
                rdata[v] = np.min(vdata)
            elif op == "max":
                rdata[v] = np.max(vdata)
            else:
                raise KeyError(
                    f"Unknown operation '{op}' for variable '{v}'. Please choose: sum, mean, min, max"
                )

        return rdata

    def calc_states_mean(self, vars):
        """
        Calculates the mean wrt states.

        Parameters
        ----------
        vars : list of str
            The variables

        Returns
        -------
        data : pandas.DataFrame
            The results per turbine

        """
        if isinstance(vars, str):
            return self.reduce_states({vars: "mean"})
        return self.reduce_states({v: "mean" for v in vars})

    def calc_states_sum(self, vars):
        """
        Calculates the sum wrt states.

        Parameters
        ----------
        vars : list of str
            The variables

        Returns
        -------
        data : pandas.DataFrame
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
        vars : list of str
            The variables

        Returns
        -------
        data : pandas.DataFrame
            The results per state

        """
        return self.reduce_turbines({v: "mean" for v in vars})

    def calc_turbine_sum(self, vars):
        """
        Calculates the sum wrt turbines.

        Parameters
        ----------
        vars : list of str
            The variables

        Returns
        -------
        data : pandas.DataFrame
            The results per state

        """
        return self.reduce_turbines({v: "sum" for v in vars})

    def calc_farm_mean(self, vars):
        """
        Calculates the mean over states and turbines.

        Parameters
        ----------
        vars : list of str
            The variables

        Returns
        -------
        data : dict
            The fully contracted results

        """
        op = {v: "mean" for v in vars}
        return self.reduce_all(states_op=op, turbines_op=op)

    def calc_farm_sum(self, vars):
        """
        Calculates the sum over states and turbines.

        Parameters
        ----------
        vars : list of str
            The variables

        Returns
        -------
        data : dict
            The fully contracted results

        """
        op = {v: "sum" for v in vars}
        return self.reduce_all(states_op=op, turbines_op=op)

    def calc_mean_farm_power(self, ambient=False):
        """
        Calculates the mean total farm power.

        Parameters
        ----------
        ambient : bool
            Flag for ambient power

        Returns
        -------
        data : float
            The mean wind farm power

        """
        v = FV.P if not ambient else FV.AMB_P
        cdata = self.reduce_all(states_op={v: "mean"}, turbines_op={v: "sum"})
        return cdata[v]

    def calc_turbine_yield(
        self,
        annual=False,
        ambient=False,
        hours=None,
        P_unit_W=1e3,
    ):
        """
        Calculates the yield per turbine

        Parameters
        ----------
        annual : bool, optional
            Flag for returing annual results, by default False
        ambient : bool, optional
            Flag for ambient power, by default False
        hours : int, optional
            The duration time in hours, if not timeseries states
        P_unit_W : float
            The power unit in Watts, 1000 for kW

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

        # compute yield per turbine
        if np.issubdtype(self.results[FV.STATE].dtype, np.datetime64):
            if hours is not None:
                raise KeyError("Unexpected parameter 'hours' for timeseries data")
            duration = self.results[FV.STATE][-1] - self.results[FV.STATE][0]
            duration_seconds = int(duration.astype(int) / 1e9)
            duration_hours = duration_seconds / 3600
        elif hours is None and annual == True:
            duration_hours = 8760
        elif hours is None:
            raise ValueError(
                f"Expecting parameter 'hours' for non-timeseries data, or 'annual=True'"
            )
        else:
            duration_hours = hours
        yld = self.calc_states_mean(var_in) * duration_hours * P_unit_W / 1e9

        if annual:
            # convert to annual values
            yld *= 8760 / duration_hours

        yld.rename(columns={var_in: var_out}, inplace=True)
        return yld

    def add_capacity(self, algo=None, P_nom=None, ambient=False):
        """
        Adds capacity to the farm results

        Parameters
        ----------
        algo : foxes.core.Algorithm, optional
            The algorithm, for nominal power calculation
        P_nom : list of float, optional
            Nominal power values for each turbine, if algo not given
        ambient : bool, optional
            Flag for calculating ambient capacity, by default False

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
            P_nom = [t.P_nominal for t in algo.farm_controller.turbine_types]
        elif algo is None and P_nom is not None:
            pass
        else:
            raise KeyError("Expecting either 'algo' or 'P_nom'")

        # add to farm results
        self.results[var_out] = vdata / P_nom
        if ambient:
            print("Ambient capacity added to farm results")
        else:
            print("Capacity added to farm results")

    def calc_farm_yield(self, turbine_yield=None, power_uncert=None, **kwargs):
        """
        Calculates yield, P75 and P90 at the farm level

        Parameters
        ----------
        turbine_yield : pandas.DataFrame, optional
            Yield values by turbine
        power_uncert : float, optional
            Uncertainty in the power value. Triggers
            P75 and P90 outputs
        kwargs : dict, optional
            Parameters for calc_turbine_yield(). Apply if
            turbine_yield is not given

        Returns
        -------
        farm_yield : float
            Farm yield result, same unit as turbine yield
        P75 : float, optional
            The P75 value, same unit as turbine yield
        P90 : float, optional
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

    def add_efficiency(self):
        """
        Adds efficiency to the farm results
        """
        P = self.results[FV.P]
        P0 = self.results[FV.AMB_P] + 1e-14
        self.results[FV.EFF] = P / P0  # add to farm results
        print("Efficiency added to farm results")

    def calc_farm_efficiency(self):
        """
        Calculates farm efficiency

        Returns
        -------
        eff : float
            The farm efficiency

        """
        P = self.calc_mean_farm_power()
        P0 = self.calc_mean_farm_power(ambient=True) + 1e-14
        return P / P0
