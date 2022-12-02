import numpy as np
import numbers

from foxes.core import WakeSuperposition
import foxes.variables as FV
import foxes.constants as FC


class LinearSuperposition(WakeSuperposition):
    """
    Linear supersposition of wake model results,
    optionally rescaled.

    Parameters
    ----------
    scalings : dict or number or str
        Scaling rules. If `dict`, key: variable name str,
        value: number or str. If `str`:
        - `source_turbine`: Scale by source turbine value of variable
        - `source_turbine_amb`: Scale by source turbine ambient value of variable
        - `source_turbine_<var>`: Scale by source turbine value of variable <var>
    lim_low : dict, optional
        Lower limits of the final wake deltas. Key: variable str,
        value: float
    lim_high : dict, optional
        Higher limits of the final wake deltas. Key: variable str,
        value: float

    Attributes
    ----------
    scalings : dict or number or str
        The scaling rules
    lim_low : dict
        Lower limits of the final wake deltas. Key: variable str,
        value: float
    lim_high : dict
        Higher limits of the final wake deltas. Key: variable str,
        value: float

    """

    def __init__(self, scalings, lim_low=None, lim_high=None):
        super().__init__()

        self.scalings = scalings
        self.lim_low = lim_low
        self.lim_high = lim_high

    def calc_wakes_plus_wake(
        self,
        algo,
        mdata,
        fdata,
        states_source_turbine,
        sel_sp,
        variable,
        wake_delta,
        wake_model_result,
    ):
        """
        Add a wake delta to previous wake deltas.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
        states_source_turbine : numpy.ndarray
            For each state, one turbine index for the
            wake causing turbine. Shape: (n_states,)
        sel_sp : numpy.ndarray of bool
            The selection of points, shape: (n_states, n_points)
        variable : str
            The variable name for which the wake deltas applies
        wake_delta : numpy.ndarray
            The original wake deltas, shape: (n_states, n_points)
        wake_model_result : numpy.ndarray
            The new wake deltas of the selected points,
            shape: (n_sel_sp,)

        Returns
        -------
        wdelta : numpy.ndarray
            The updated wake deltas, shape: (n_states, n_points)

        """
        if isinstance(self.scalings, dict):
            try:
                scaling = self.scalings[variable]
            except KeyError:
                raise KeyError(
                    f"Model '{self.name}': No scaling found for wake variable '{variable}'"
                )
        else:
            scaling = self.scalings

        if scaling is None:
            wake_delta[sel_sp] += wake_model_result
            return wake_delta

        elif isinstance(scaling, numbers.Number):
            wake_delta[sel_sp] += scaling * wake_model_result
            return wake_delta

        elif (
            isinstance(scaling, str)
            and len(scaling) >= 14
            and (
                scaling == f"source_turbine"
                or scaling == "source_turbine_amb"
                or (len(scaling) > 15 and scaling[14] == "_")
            )
        ):

            if scaling == f"source_turbine":
                var = variable
            elif scaling == "source_turbine_amb":
                var = FV.var2amb[variable]
            else:
                var = scaling[15:]

            try:
                vdata = fdata[var]

            except KeyError:
                raise KeyError(
                    f"Model '{self.name}': Scaling variable '{var}' for wake variable '{variable}' not found in fdata {sorted(list(fdata.keys()))}"
                )

            n_states = mdata.n_states
            n_points = wake_delta.shape[1]
            stsel = (np.arange(n_states), states_source_turbine)
            scale = np.zeros((n_states, n_points), dtype=FC.DTYPE)
            scale[:] = vdata[stsel][:, None]
            scale = scale[sel_sp]

            wake_delta[sel_sp] += scale * wake_model_result

            return wake_delta

        else:
            raise ValueError(
                f"Model '{self.name}': Invalid scaling choice '{scaling}' for wake variable '{variable}', valid choices: None, <scalar>, 'source_turbine', 'source_turbine_amb', 'source_turbine_<var>'"
            )

    def calc_final_wake_delta(
        self, algo, mdata, fdata, variable, amb_results, wake_delta
    ):
        """
        Calculate the final wake delta after adding all
        contributions.

        Parameters
        ----------
        algo : foxes.core.Algorithm
            The calculation algorithm
        mdata : foxes.core.Data
            The model data
        fdata : foxes.core.Data
            The farm data
        variable : str
            The variable name for which the wake deltas applies
        amb_results : numpy.ndarray
            The ambient results, shape: (n_states, n_points)
        wake_delta : numpy.ndarray
            The wake deltas, shape: (n_states, n_points)

        Returns
        -------
        final_wake_delta : numpy.ndarray
            The final wake delta, which will be added to the ambient
            results by simple plus operation. Shape: (n_states, n_points)

        """
        w = wake_delta
        if self.lim_low is not None and variable in self.lim_low:
            w = np.maximum(w, self.lim_low[variable] - amb_results)
        if self.lim_high is not None and variable in self.lim_high:
            w = np.minimum(w, self.lim_high[variable] - amb_results)
        return w
