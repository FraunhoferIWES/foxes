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
