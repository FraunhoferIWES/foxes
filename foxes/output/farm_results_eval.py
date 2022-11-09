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

    def calc_yield(self, hours=24*365, power_factor=1, ambient=False):
        """ Calculates yield based on power over all states

        Args:
            hours (int, optional): _description_. Defaults to 24*365.
            power_factor (int, optional): _description_. Defaults to 1.
            ambient (bool, optional): _description_. Defaults to False.

        Returns:
            pandas.Dataframe: a dataframe containing yield, P75 and P90 per turbine
        """
        if ambient:
            vars='AMB_P'
        else: vars = 'P'

        # get results data for the vars variable (by state and turbine)
        vdata = self.results[vars].to_dataframe()
        
        # compute yield per turbine (average power over states * hours * power_factor)
        YLD = vdata[vars].groupby(['turbine']).mean() * hours * power_factor
        
        # P75 and P90 based on mean yield per turbine
        UNCERT = YLD.std() / YLD.mean()
        P75 = YLD.mean() * (1.0 - (0.675 * UNCERT))
        P90 = YLD.mean() * (1.0 - (1.282 * UNCERT))
        print(f"\nMean yield per turbine is {YLD}")
        print(f"\nP75 is {P75}")
        print(f"\nP90 is {P90}")
        print("\nTotal farm yield is", YLD.sum())

        # histogram
        plt.hist(YLD, bins=15)
        plt.axvline(YLD.mean(), c="orange")
        plt.axvline(P75, c="red", linestyle="dotted")
        plt.axvline(P90, c="red", linestyle="dotted")
        plt.text(20.1, 8, "P90")
        plt.text(20.8, 8, "P75")
        plt.xlabel("Yield per turbine")
        plt.show()
        #plt.savefig("Hist.png") # sort of normally distributed...!

        print()

        # hist for turbine 0
        turb_0 = vdata.query("turbine == 0") * hours * power_factor
        plt.hist(turb_0['P'])
        plt.show()
        #plt.savefig("Hist_turbine_0.png") # not normally distributed so P75 and P90 for each turbine would be meaningless

        ydata = pd.DataFrame({'YLD_per_turbine': [YLD.mean()],
        'P75': [P75],
        'P90': [P90]})
        return ydata