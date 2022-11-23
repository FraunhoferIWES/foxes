import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
                rdata[v] = np.einsum('st,st->t', vdata, weights)
            elif op == "sum":
                rdata[v] = np.sum(vdata, axis=0)
            elif op == "min":
                rdata[v] = np.min(vdata, axis=0)
            elif op == "max":
                rdata[v] = np.max(vdata, axis=0)
            elif op == "std":
                rdata[v] = np.std(vdata, axis=0)
            else:
                raise KeyError(f"Unknown operation '{op}' for variable '{v}'. Please choose: sum, mean, min, max")

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
                rdata[v] = np.einsum('st,st->s', vdata, weights)
            elif op == "sum":
                rdata[v] = np.sum(vdata, axis=1)
            elif op == "min":
                rdata[v] = np.min(vdata, axis=1)
            elif op == "max":
                rdata[v] = np.max(vdata, axis=1)
            else:
                raise KeyError(f"Unknown operation '{op}' for variable '{v}'. Please choose: sum, mean, min, max")

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
                    rdata[v] = np.einsum('st,st->', vdata, weights)
                else:
                    rdata[v] = np.einsum('st,st->', vdata[None, :], weights)
            elif op == "sum":
                rdata[v] = np.sum(vdata)
            elif op == "min":
                rdata[v] = np.min(vdata)
            elif op == "max":
                rdata[v] = np.max(vdata)
            else:
                raise KeyError(f"Unknown operation '{op}' for variable '{v}'. Please choose: sum, mean, min, max")
        
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

    def calc_times(self, states):
        if str(states) == 'Timeseries':
            # get timestep from timeseries data
            timestep_delta = self.results.state[1] - self.results.state[0]
            timestep_seconds = int(timestep_delta.astype(int)/1e9)
            timestep_hours = timestep_seconds / 3600
            return timestep_hours


    def calc_yield(self, timestep=1, power_factor=0.000001, power_uncert=0.08, ambient=False):

        if ambient:
            var_in= FV.AMB_P
            var_out = FV.AMB_YLD
        else: 
            var_in = FV.P
            var_out = FV.YLD

        # get results data for the vars variable (by state and turbine)
        vdata = self.results[var_in]

        # compute yield per turbine (default is MWh)
        YLD = vdata * timestep
        
        # add to farm results
        self.results[var_out] = YLD
        if ambient:
            print("Ambient yield data added to farm results")
        else: print("Yield data added to farm results")
        
    def calc_turbine_yield(self, power_factor=1e-6, annual=False, ambient=False):

        if ambient:
            vars = [FV.AMB_YLD]
        else: 
            vars = [FV.YLD]

        # reduce the states 
        tdata = self.calc_states_sum(vars) # for the duration of the timeseries

        if annual:
            # convert to annual values
            duration = self.results.state[-1] - self.results.state[0]
            duration_seconds = int(duration.astype(int)/1e9)
            duration_hours = duration_seconds / 3600
            tdata = tdata* 24*365 / duration_hours
        
        return tdata * power_factor 

    def calc_capacity(self, P_nom, timestep=1, annual=False, ambient=False):

        if ambient:
            var_in= FV.AMB_P
            var_out = FV.AMB_CAP
        else: 
            var_in = FV.P
            var_out = FV.CAP

        # get results data for the vars variable (by state and turbine)
        vdata = self.results[var_in]

        CAP = vdata / P_nom

        # add to farm results
        self.results[var_out] = CAP
        print("Capacity added to farm results")

    def calc_farm_yield(self, power_factor=1e-6, power_uncert=0.08, annual=False, ambient=False):

        if ambient:
            vars = [FV.AMB_YLD]
        else: 
            vars = [FV.YLD]

        # reduce the states and turbines by summing
        print()
        farm_yield = self.calc_farm_sum(vars)['YLD'] * power_factor # for the duration of the timeseries

        if annual:
            # convert to annual values
            duration = self.results.state[-1] - self.results.state[0]
            duration_seconds = int(duration.astype(int)/1e9)
            duration_hours = duration_seconds / 3600
            farm_yield = farm_yield* 24*365 / duration_hours

        # P75 and P90
        P75 = farm_yield * (1.0 - (0.675 * power_uncert))
        P90 = farm_yield * (1.0 - (1.282 * power_uncert))
                
        return farm_yield, P75, P90

    def calc_efficiency(self):
        P = self.results[FV.P]
        P0 = self.results[FV.AMB_P]
        self.results[FV.EFF] = P / P0 # add to farm results
        print('Efficiency added to farm results')