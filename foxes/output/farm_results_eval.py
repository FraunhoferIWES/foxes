from __future__ import annotations

import numpy as np
import pandas as pd
from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Any, Iterator, cast
from xarray import Dataset
from matplotlib.lines import Line2D
from typing import TYPE_CHECKING

from .output import Output
from foxes.config import config
from foxes.utils import write_nc
import foxes.variables as FV
import foxes.constants as FC

if TYPE_CHECKING:
    from foxes.core.algorithm import Algorithm


class FarmResultsEval(Output):
    """
    Evaluates farm results data.

    This sums over turbines and/or states,
    given the state-turbine farm_calc results.

    Attributes
    ----------
    algo
        The algorithm object


    """

    def __init__(
        self, farm_results: Dataset | None, algo: Algorithm | None = None, **kwargs: Any
    ) -> None:
        """
        Constructor.

        Parameters
        ----------
        farm_results
            The farm results, if available
        algo
            The algorithm object
        kwargs
            Additional parameters for the base class

        """
        super().__init__(**kwargs)
        self.algo = algo
        self._results = cast(Dataset, farm_results)
        self._LEVEL = FC.TURBINE

    @property
    def results(self) -> Dataset:
        """
        Get the farm results.

        Returns
        -------
        The evaluated dataset
            The farm results

        """
        return self._results

    def weinsum(self, rhs: str, *vars: str | np.ndarray) -> np.ndarray:
        """
        Calculates Einstein sum, adding weights
        as last argument to the given fields.

        It's all about treating NaN values.

        Parameters
        ----------
        rhs
            The right-hand side of the einsum expression.
            Use 's' for states and 't' for turbines.
        vars
            The variables mentioned in the expression,
            but without the obligatory weights that will
            be added at the end

        Returns
        -------
        result
            The results array

        """
        fields: list[np.ndarray] = []
        for v in vars:
            if isinstance(v, str):
                vdata = self.results[v].to_numpy()
                nns = np.sum(np.isnan(vdata))
                assert nns == 0, (
                    f"Found {nns} nan values for variable '{v}' of shape {vdata.shape}"
                )
                fields.append(vdata)
            elif isinstance(v, np.ndarray):
                fields.append(v)
            else:
                raise TypeError(
                    f"Expecting variable name as str or array, got {type(v)}"
                )

        if not fields:
            raise ValueError("No data fields supplied for einsum reduction.")

        nan_mask = np.zeros_like(fields[0], dtype=bool)
        for field in fields:
            nan_mask = nan_mask | np.isnan(field)

        inds = ["st" for __ in fields]
        if self.results[FV.WEIGHT].dims == (FC.STATE,):
            inds += ["s"]

            if np.any(nan_mask):
                sel = ~np.any(nan_mask, axis=1)
                fields = [f[sel] for f in fields]

                weights0 = self.results[FV.WEIGHT].to_numpy()
                w0 = np.sum(weights0)
                weights = weights0[sel]
                w1 = np.sum(weights)
                weights *= w0 / w1
                fields.append(weights)

            else:
                fields.append(self.results[FV.WEIGHT].to_numpy())

        elif self.results[FV.WEIGHT].dims == (FC.STATE, self._LEVEL):
            inds += ["st"]

            if np.any(nan_mask):
                sel = ~np.any(nan_mask, axis=1)
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
                f"Expecting '{FV.WEIGHT}' variable with dimensions {(FC.STATE,)} or {(FC.STATE, self._LEVEL)}, got {self.results[FV.WEIGHT].dims}"
            )
        expr = ",".join(inds) + "->" + rhs

        return np.einsum(expr, *fields)

    def reduce_states(
        self, vars_op: dict[str, str | None] | None = None
    ) -> pd.DataFrame:
        """
        Reduces the states dimension by some operation

        Parameters
        ----------
        vars_op
            The operation per variable. The mapping is from variable name
            to reduction mode, with choices: weights, mean_no_weights,
            sum, min, max.

        Returns
        -------
        data
            The results per turbine.

        """

        if vars_op is None:
            vrs = [
                v
                for v, d in self.results.data_vars.items()
                if d.dims == (FC.STATE, self._LEVEL)
            ]
            vars_op = {v: "sum" if v in FV.extensive_state else "weights" for v in vrs}
            vars_op.update(
                {
                    v: None
                    for v, d in self.results.data_vars.items()
                    if d.dims == (self._LEVEL,)
                }
            )

        rdata = {}
        for v, op in vars_op.items():
            vdata = self.results[v].to_numpy()

            try:
                nns = np.sum(np.isnan(vdata))
                assert nns == 0, (
                    f"Found {nns} nan values for variable '{v}' of shape {vdata.shape}"
                )
            except TypeError:
                pass

            if op is None:
                rdata[v] = vdata
            elif op == "weights":
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

        data = pd.DataFrame(index=self.results[self._LEVEL].values, data=rdata)
        data.index.name = self._LEVEL

        return data

    def reduce_turbines(self, vars_op: dict[str, str]) -> pd.DataFrame:
        """
        Reduces the turbine dimension by some operation

        Parameters
        ----------
        vars_op
            The operation per variable. The mapping is from variable name
            to reduction mode, with choices: weights, mean_no_weights,
            sum, min, max.

        Returns
        -------
        data
            The results per state.

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

    def reduce_all(
        self, states_op: dict[str, str | None], turbines_op: dict[str, str]
    ) -> dict:
        """
        Reduces states and turbine dimension by some operation

        Parameters
        ----------
        states_op
            The states contraction operations. The mapping is from variable
            name to reduction mode, with choices: sum, mean, min, max.
        turbines_op
            The turbines contraction operations. The mapping is from variable
            name to reduction mode, with choices: sum, mean, min, max.

        Returns
        -------
        data
            The fully contracted results.

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

    def calc_states_mean(
        self, vars: str | list[str], use_weights: bool = True
    ) -> pd.DataFrame:
        """
        Calculates the mean wrt states.

        Parameters
        ----------
        vars
            The variables
        use_weights
            Flag for using states weights for the mean

        Returns
        -------
        data
            The results per turbine

        """
        r = "weights" if use_weights else "mean_no_weights"
        if isinstance(vars, str):
            return self.reduce_states({vars: r})
        return self.reduce_states({v: r for v in vars})

    def calc_states_sum(self, vars: list[str]) -> pd.DataFrame:
        """
        Calculates the sum wrt states.

        Parameters
        ----------
        vars
            The variables

        Returns
        -------
        data
            The results per turbine

        """
        return self.reduce_states({v: "sum" for v in vars})

    def calc_states_std(self, vars: list[str]) -> pd.DataFrame:
        """
        Calculates the standard deviation wrt states.

        Args:
            vars (_type_): _description_

        Returns:
            _type_: _description_
        """

        return self.reduce_states({v: "std" for v in vars})

    def calc_turbine_mean(self, vars: list[str]) -> pd.DataFrame:
        """
        Calculates the mean wrt turbines.

        Parameters
        ----------
        vars

        Returns
        -------
        data
            The results per state

        """
        return self.reduce_turbines({v: "mean_no_weights" for v in vars})

    def calc_turbine_sum(self, vars: list[str]) -> pd.DataFrame:
        """
        Calculates the sum wrt turbines.

        Parameters
        ----------
        vars
            The variables

        Returns
        -------
        data
            The results per state

        """
        return self.reduce_turbines({v: "sum" for v in vars})

    def calc_farm_mean(self, vars: list[str]) -> dict[str, np.ndarray | float]:
        """
        Calculates the mean over states and turbines.

        Parameters
        ----------
        vars
            The variables

        Returns
        -------
        data
            The fully contracted results

        """
        op_states: dict[str, str | None] = {v: "weights" for v in vars}
        op_turbines: dict[str, str] = {v: "weights" for v in vars}
        return self.reduce_all(states_op=op_states, turbines_op=op_turbines)

    def calc_farm_sum(self, vars: list[str]) -> dict[str, np.ndarray | float]:
        """
        Calculates the sum over states and turbines.

        Parameters
        ----------
        vars
            The variables

        Returns
        -------
        data
            The fully contracted results

        """
        op_states: dict[str, str | None] = {v: "sum" for v in vars}
        op_turbines: dict[str, str] = {v: "sum" for v in vars}
        return self.reduce_all(states_op=op_states, turbines_op=op_turbines)

    def calc_mean_farm_power(self, ambient: bool = False) -> float:
        """
        Calculates the mean total farm power.

        Parameters
        ----------
        ambient
            Flag for ambient power

        Returns
        -------
        data
            The mean wind farm power

        """
        v = FV.P if not ambient else FV.AMB_P
        cdata = self.reduce_all(states_op={v: "weights"}, turbines_op={v: "sum"})
        return cdata[v]

    def get_power_units(self) -> np.ndarray:
        """
        Gets the power units in Watts for all elements

        Returns
        -------
        P_unit_W
            The power units in Watts for all elements, shape: (n_elements,)

        """
        if self.algo is not None:
            turbine_types = self.algo.farm_controller.turbine_types
            assert turbine_types is not None
            P_unit_W: np.ndarray = np.array(
                [FC.P_UNITS[t.P_unit] for t in turbine_types],
                dtype=config.dtype_double,
            )
            return P_unit_W
        else:
            raise KeyError("Algorithm object is required for getting power units")

    def calc_yield(
        self,
        annual: bool = False,
        ambient: bool = False,
        hours: int | None = None,
        delta_t: np.timedelta64 | None = None,
        P_unit_W: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """
        Calculates the yield

        ----------
        annual
            Flag for returning annual results, by default False
        ambient
            Flag for ambient power, by default False
        hours
            The duration time in hours, if not timeseries states
        delta_t: np.datetime64, optional
            The time delta step in case of time series data,
            by default automatically determined
        P_unit_W
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

        if self.algo is not None and P_unit_W is None:
            P_unit_W = self.get_power_units()[:, None]
        elif self.algo is None and P_unit_W is not None:
            pass
        else:
            raise KeyError("Expecting either algorithm or 'P_unit_W'")

        duration_hours: float

        # compute yield per turbine
        if hours is None and annual:
            duration_hours = 8760.0
        elif np.issubdtype(self.results[FC.STATE].dtype, np.datetime64):
            if hours is not None:
                raise KeyError("Unexpected parameter 'hours' for timeseries data")
            times = self.results[FC.STATE].to_numpy()
            if delta_t is None:
                delta_t = times[-1] - times[-2]
            duration = times[-1] - times[0] + delta_t
            duration_seconds = np.int64(duration.astype(np.int64) / 1e9)
            duration_hours = float(duration_seconds) / 3600.0
        elif hours is None:
            raise ValueError(
                "Expecting parameter 'hours' for non-timeseries data, or 'annual=True'"
            )
        else:
            duration_hours = float(hours)

        yld = self.calc_states_mean(var_in) * duration_hours * P_unit_W / 1e9

        if duration_hours != 8760 and annual:
            # convert to annual values
            yld *= 8760 / duration_hours

        yld.rename(columns={var_in: var_out}, inplace=True)
        return yld

    def get_capacity(self) -> np.ndarray:
        """

        Returns
        -------
        capacity_array
            The capacity array (nominal power) for all turbines, shape: (n_turbines,)

        """
        assert self.algo is not None, (
            "Algorithm object is required for adding capacity to farm results"
        )
        Pnom = self.algo.farm.get_capacity_array(self.algo)
        return Pnom

    def add_capacity(self, verbosity: int = 1) -> None:
        """
        Adds capacity to the farm results, equals P_nominal on turbine level

        Parameters
        ----------
        verbosity
            The verbosity level, 0 = silent

        """
        self._results[FV.CAP] = ((self._LEVEL,), self.get_capacity())
        if verbosity > 0:
            print("Capacity added to farm results")

    def add_yield(
        self,
        annual: bool = True,
        ambient: bool = False,
        verbosity: int = 1,
        **kwargs: Any,
    ) -> None:
        """
        Adds yield to the farm results

        Parameters
        ----------
        annual
            Flag for returning annual results
        ambient
        verbosity
            The verbosity level, 0 = silent
        kwargs
            Parameters for calc_yield()

        """
        yld = self.calc_yield(annual=annual, ambient=ambient, **kwargs)
        assert len(yld.columns) == 1, "Expecting single column in yield dataframe"
        v = yld.columns[0]
        self._results[v] = ((self._LEVEL,), yld[v].to_numpy())
        if verbosity > 0:
            s = "Ambient yield" if ambient else "Yield"
            print(f"{s} added to results")

    def add_capacity_factor(
        self,
        capacity: np.ndarray | None = None,
        ambient: bool = False,
        verbosity: int = 1,
    ) -> None:
        """
        Adds capacity factor to the farm results, P / CAP

        Parameters
        ----------
        capacity
            Capacity values for each turbine (nominal power), if algo not given
        ambient
            Flag for calculating ambient capacity, by default False
        verbosity
            The verbosity level, 0 = silent

        """
        if ambient:
            var_in = FV.AMB_P
            var_out = FV.AMB_CAPF
        else:
            var_in = FV.P
            var_out = FV.CAPF

        # compute capacity
        cap = (
            self.results[FV.CAP].to_numpy()
            if FV.CAP in self.results
            else self.get_capacity()
        )

        # add to farm results
        self._results[var_out] = self.results[var_in] / cap[None, :]
        if verbosity > 0:
            if ambient:
                print("Ambient capacity factor added to farm results")
            else:
                print("Capacity factor added to farm results")

    def calc_farm_yield(
        self,
        turbine_yield: pd.DataFrame | None = None,
        power_uncert: float | None = None,
        **kwargs: Any,
    ) -> float | tuple[float, float, float]:
        """
        Calculates yield, P75 and P90 at the farm level

        Parameters
        ----------
        turbine_yield
            Yield values by turbine
        power_uncert
            Uncertainty in the power value. Triggers
            P75 and P90 outputs
        kwargs
            Parameters for calc_yield(). Apply if
            turbine_yield is not given

        Returns
        -------
        farm_yield
            Farm yield result, same unit as turbine yield
        P75
            The P75 value, same unit as turbine yield
        P90
            The P90 value, same unit as turbine yield

        """
        if turbine_yield is None:
            yargs: dict[str, Any] = dict(annual=True)
            yargs.update(kwargs)
            turbine_yield = self.calc_yield(**yargs)
        farm_yield = turbine_yield.sum()

        if power_uncert is not None:
            P75 = farm_yield * (1.0 - (0.675 * power_uncert))
            P90 = farm_yield * (1.0 - (1.282 * power_uncert))
            return farm_yield["YLD"], P75["YLD"], P90["YLD"]

        return farm_yield["YLD"]

    def add_efficiency(self, verbosity: int = 1) -> None:
        """
        Adds efficiency to the farm results

        Parameters
        ----------
        verbosity
            The verbosity level, 0 = silent

        """
        P = self.results[FV.P].to_numpy()
        P0 = np.maximum(self.results[FV.AMB_P].to_numpy(), 1e-12)
        eff = np.minimum(P / P0, 1)
        eff[P < 1e-10] = 0
        self._results[FV.EFF] = (self.results[FV.AMB_P].dims, eff)
        if verbosity > 0:
            print("Efficiency added to farm results")

    def add_full_load_fraction(self, ambient: bool = False, verbosity: int = 1) -> None:
        """
        Adds full load fraction to the farm results

        Parameters
        ----------
        ambient
            Flag for calculating ambient full load fraction, by default False
        verbosity
            The verbosity level, 0 = silent

        """
        if ambient:
            var_in = FV.AMB_P
            var_out = FV.AMB_FLF
        else:
            var_in = FV.P
            var_out = FV.FLF

        # get results data for the vars variable (by state and turbine)
        vdata = self.results[var_in]

        # compute capacity
        cap = (
            self.results[FV.CAP].to_numpy()
            if FV.CAP in self.results
            else self.get_capacity()
        )

        # add to farm results
        self._results[var_out] = (vdata == cap[None, :]).astype(config.dtype_double)
        if verbosity > 0:
            if ambient:
                print("Ambient full load fraction added to farm results")
            else:
                print("Full load fraction added to farm results")

    def calc_farm_efficiency(self) -> float:
        """
        Calculates farm efficiency

        Returns
        -------
        eff
            The farm efficiency

        """
        P = self.calc_mean_farm_power()
        P0 = np.maximum(self.calc_mean_farm_power(ambient=True), 1e-14)
        return np.minimum(P / P0, 1)

    def gen_stdata(
        self,
        turbines: list[int],
        variable: str,
        fig: Figure | None = None,
        ax: Axes | None = None,
        figsize: tuple[float, float] | None = None,
        legloc: str = "lower right",
        animated: bool = True,
        ret_im: bool = True,
    ) -> Iterator[Figure | tuple[Figure, list[Line2D]]]:
        """
        Generates state-turbine data,
        intended to be used in animations

        Parameters
        ----------
        turbines
            The turbines for which to scatter data
        variable
            The variable name
        fig
            The figure object
        ax
            The figure axes
        figsize
            The figsize for plt.Figure
        legloc
            The legend location
        animated
            Flag for animated output
        ret_im
            Flag for image return,

        Yields
        ------
        fig
            The figure object
        im
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

    def write_nc(self, fname: str, **kwargs: Any) -> Dataset:
        """
        Write the results to a netCDF file.

        Parameters
        ----------
        fname
            The file name to write the netCDF file
        kwargs
            Additional parameters for the foxes.utils.write_nc() method

        Returns
        -------
        The evaluated dataset
            The aggregated results that were written to the netCDF file

        """
        fpath = self.get_fpath(fname)
        write_nc(self.results, fpath, **kwargs)
        return self.results
